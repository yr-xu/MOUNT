import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mmaction.apis import init_recognizer

# PKL dataloader version
from dataloader_video_flow_epic import EPICDOMAIN


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)

    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=str, nargs='+', required=True)   # fold ckpts
    p.add_argument('--cal_splits', type=str, nargs='+', required=True)    # e.g. fold1_cal fold2_cal ...
    p.add_argument('--datapath', type=str, required=True)
    p.add_argument('--alphas', type=float, nargs='+', default=[0.1])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--out_prefix', type=str, default='cp_kfold_epic_pkl')
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

    return model, model_flow, mlp_cls


@torch.no_grad()
def dpu_probs(clip, flow, model, model_flow, mlp_cls, device):
    clip = clip['imgs'].to(device).squeeze(1)
    flow = flow['imgs'].to(device).squeeze(1)

    x_slow, x_fast = model.backbone.get_feature(clip)
    v_feat = model.backbone.get_predict((x_slow, x_fast))
    _, v_emd = model.cls_head(v_feat)

    f_feat = model_flow.backbone.get_feature(flow)
    f_feat = model_flow.backbone.get_predict(f_feat)
    _, f_emd = model_flow.cls_head(f_feat)

    logits = mlp_cls(v_emd, f_emd)
    return F.softmax(logits, dim=1)


def main():
    args = parse_args()
    assert len(args.checkpoints) == len(args.cal_splits), "checkpoints and cal_splits must match length"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt0 = torch.load(args.checkpoints[0], map_location=device)
    num_classes = ckpt0['mlp_cls_state_dict']['enc_net.weight'].shape[0]

    all_scores = []

    for ckpt_path, split_name in zip(args.checkpoints, args.cal_splits):
        print(f"\n=== Fold calibration: ckpt={ckpt_path} on split={split_name} ===")

        model, model_flow, mlp_cls = load_bundle(
            ckpt_path=ckpt_path,
            device=device,
            num_classes=num_classes
        )

        calib_dataset = EPICDOMAIN(
            split=split_name,
            eval=False,
            cfg=model.cfg,
            cfg_flow=model_flow.cfg,
            sample_dur=10,
            far_ood=False,
            datapath=args.datapath,
            labeled_only=True,   # strict labeled-only loading
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
            scores = 1.0 - p_true   # conformal nonconformity score
            fold_scores.append(scores)
            n_valid += n

        if len(fold_scores) == 0:
            raise RuntimeError(f"No valid labeled samples found in calibration split '{split_name}'")

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

        out = f"{args.out_prefix}_alpha{alpha}.npy"
        np.save(out, np.array([qhat, thr], dtype=np.float32))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()