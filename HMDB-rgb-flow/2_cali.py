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
        feat = torch.cat((vfeat, afeat), dim = 1)
        return self.enc_net(feat)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=str, nargs='+', required=True)   # fold ckpts
    p.add_argument('--cal_splits', type=str, nargs='+', required=True)    # fold*_cal
    p.add_argument('--datapath', type=str, required=True)
    p.add_argument('--dataset', type=str, default='HMDB')
    p.add_argument('--alphas', type=float, nargs='+', default=[0.1])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--near_ood', action='store_true')
    p.add_argument('--out_prefix', type=str, default='cp_kfold')
    return p.parse_args()

@torch.no_grad()
def dpu_probs(clip, flow, model, model_flow, mlp_cls, device):
    clip = clip['imgs'].to(device).squeeze(1)
    flow = flow['imgs'].to(device).squeeze(1)

    x_slow, x_fast = model.module.backbone.get_feature(clip)
    v_feat = model.module.backbone.get_predict((x_slow, x_fast))
    v_logits, v_emd = model.module.cls_head(v_feat)

    f_feat = model_flow.module.backbone.get_feature(flow)
    f_feat = model_flow.module.backbone.get_predict(f_feat)
    f_logits, f_emd = model_flow.module.cls_head(f_feat)

    logits = mlp_cls(v_emd, f_emd)
    return F.softmax(logits, dim=1)

def main():
    args = parse_args()
    assert len(args.checkpoints) == len(args.cal_splits), "checkpoints and cal_splits must match length"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v_dim = 2304
    f_dim = 2048

    ckpt0 = torch.load(args.checkpoints[0], map_location=device)
    num_classes = ckpt0['mlp_cls_state_dict']['enc_net.weight'].shape[0]

    config_file_rgb = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'

    model = init_recognizer(config_file_rgb, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_classes).to(device)
    model = nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, device=device, use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_classes).to(device)
    model_flow = nn.DataParallel(model_flow)

    mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=num_classes).to(device)

    all_scores = []

    for ckpt_path, split_name in zip(args.checkpoints, args.cal_splits):
        print(f"\n=== Fold calibration: ckpt={ckpt_path} on split={split_name} ===")
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt['model_state_dict'])
        model_flow.load_state_dict(ckpt['model_flow_state_dict'])
        mlp_cls.load_state_dict(ckpt['mlp_cls_state_dict'])

        model.eval()
        model_flow.eval()
        mlp_cls.eval()

        calib_dataset = EPICDOMAIN(
            split=split_name,
            eval=False,
            cfg=model.module.cfg,
            cfg_flow=model_flow.module.cfg,
            sample_dur=10,
            dataset=args.dataset,
            near_ood=args.near_ood,
            far_ood=False,
            datapath=args.datapath,
        )

        calib_loader = DataLoader(
            calib_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )


        fold_scores = []
        n_valid = 0

        for clip, flow, labels in calib_loader:
            probs = dpu_probs(clip, flow, model, model_flow, mlp_cls, device).cpu()
            labels = labels.clone()

            mask = (labels >= 0) & (labels < num_classes)
            if not mask.any():
                continue

            probs = probs[mask]
            labels = labels[mask].long()

            n = labels.shape[0]
            p_true = probs[torch.arange(n), labels]
            scores = 1.0 - p_true
            fold_scores.append(scores)
            n_valid += n

        fold_scores = torch.cat(fold_scores, dim=0)
        print(f"Fold valid labeled samples: {n_valid}")
        print(f"Fold score mean: {fold_scores.mean().item():.6f}")

        all_scores.append(fold_scores)

    all_scores = torch.cat(all_scores, dim=0)
    n = all_scores.numel()
    print(f"\nTOTAL pooled calibration samples: {n}")

    scores_sorted, _ = torch.sort(all_scores)

    for alpha in args.alphas:
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        qhat = scores_sorted[k - 1].item()
        thr = 1.0 - qhat

        print(f"\nalpha={alpha}")
        print(f"qhat={qhat:.6f}")
        print(f"prob threshold={thr:.6f}")

        out = f"{args.out_prefix}_{args.dataset}_alpha{alpha}.npy"
        np.save(out, np.array([qhat, thr], dtype=np.float32))

if __name__ == "__main__":
    main()