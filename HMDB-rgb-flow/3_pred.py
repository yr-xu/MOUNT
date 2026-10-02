import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mmaction.apis import init_recognizer
from dataloader_video_flow import EPICDOMAIN

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)

    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--checkpoints', type=str, nargs='+', required=True)
    p.add_argument('--datapath', type=str, required=True)
    p.add_argument('--dataset', type=str, default='HMDB')
    p.add_argument('--near_ood', action='store_true')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--qhat_files', type=str, nargs='+', required=True)
    p.add_argument('--out_all_prefix', type=str, required=True)

    return p.parse_args()

def strip_module_prefix(state_dict):
    if len(state_dict) == 0:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def load_bundle(ckpt_path, device, num_classes, v_dim=2304, f_dim=2048):
    config_file_rgb  = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'

    ckpt = torch.load(ckpt_path, map_location=device)

    model = init_recognizer(config_file_rgb, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_classes).to(device)

    model_flow = init_recognizer(config_file_flow, device=device, use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_classes).to(device)

    mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=num_classes).to(device)

    model.load_state_dict(strip_module_prefix(ckpt["model_state_dict"]), strict=True)
    model_flow.load_state_dict(strip_module_prefix(ckpt["model_flow_state_dict"]), strict=True)
    mlp_cls.load_state_dict(ckpt["mlp_cls_state_dict"], strict=True)

    model.eval()
    model_flow.eval()
    mlp_cls.eval()

    return (model, model_flow, mlp_cls)

@torch.no_grad()
def DPU_probs_bundle(bundle, clip, flow, device):
    model, model_flow, mlp_cls = bundle

    clip = clip['imgs'].to(device).squeeze(1)
    flow = flow['imgs'].to(device).squeeze(1)

    x_slow, x_fast = model.backbone.get_feature(clip)
    v_feat = model.backbone.get_predict((x_slow, x_fast))
    v_logits, v_emd = model.cls_head(v_feat)

    f_feat = model_flow.backbone.get_feature(flow)
    f_feat = model_flow.backbone.get_predict(f_feat)
    f_logits, f_emd = model_flow.cls_head(f_feat)

    logits = mlp_cls(v_emd, f_emd)
    probs = F.softmax(logits, dim=1)
    return probs


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v_dim = 2304
    f_dim = 2048

    taus = []
    tags = []
    for qf in args.qhat_files:
        q_hat = float(np.load(qf)[0])
        tau = 1.0 - q_hat
        taus.append(tau)

        base = qf.split("/")[-1]
        tag = base.replace(".npy", "")
        tags.append(tag)
        
    ckpt0 = torch.load(args.checkpoints[0], map_location=device)
    num_classes = ckpt0['mlp_cls_state_dict']['enc_net.weight'].shape[0]

    bundles = []
    for ckpt_path in args.checkpoints:
        bundle = load_bundle(
            ckpt_path=ckpt_path,
            device=device,
            num_classes=num_classes,
            v_dim=v_dim,
            f_dim=f_dim
        )
        bundles.append(bundle)
    K = len(bundles)
    print(f"Loaded {K} checkpoints for ensemble.")

    model0, model_flow0, mlp0 = bundles[0]

    dataset = EPICDOMAIN(
        split='3_pred',
        eval=False,
        cfg=model0.cfg,
        cfg_flow=model_flow0.cfg,
        sample_dur=10,
        dataset=args.dataset,
        near_ood=args.near_ood,
        far_ood=False,
        datapath=args.datapath,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    @torch.no_grad()
    def ensemble_probs(clip, flow):
        probs_sum = None
        for b in bundles:
            pb = DPU_probs_bundle(b, clip, flow, device)
            probs_sum = pb if probs_sum is None else (probs_sum + pb)
        return probs_sum / float(K)



    all_samples = []
    global_index = 0

    for clip, flow, labels in loader:
        probs = ensemble_probs(clip, flow).detach().cpu()
    
        batch_size = probs.size(0)
        for i in range(batch_size):
            idx = global_index + i
            path = dataset.samples[idx]
    
            p = probs[i]
            y_hat = int(torch.argmax(p).item())
    
            all_samples.append({
                "path": path,
                "y_hat": y_hat,
                "p": p,
            })
    

        global_index += batch_size

    for tag, tau in zip(tags, taus):
        lines = []

        set_sizes = []
    
        for s in all_samples:
            y_hat = s["y_hat"]
            p = s["p"]
    
            pred_set = (p >= tau).nonzero(as_tuple=True)[0].tolist()
            if len(pred_set) == 0:
                pred_set = [y_hat]
    
            set_sizes.append(len(pred_set))
            lines.append(f"{s['path']} {y_hat} {pred_set}\n")

        print(f"\n[{tag}] threshold = {tau:.6f}")
        print(f"Average set size: {float(np.mean(set_sizes)) if set_sizes else 0.0:.4f}")
        print(f"Median set size: {float(np.median(set_sizes)) if set_sizes else 0.0:.4f}")
        print(f"Total ID samples lines : {len(all_samples)}")

        out_all = f"{args.out_all_prefix}_{tag}.txt"
    
        with open(out_all, "w") as f:
            f.writelines(lines)

if __name__ == "__main__":
    main()