#!/usr/bin/env python3
"""V-JEPA2 ViT-L domain-adaptive continued pretraining (DAPT) — 12GB-friendly.

Reuses the official repo's components (init_video_model / MaskCollator / apply_masks)
and replicates app/vjepa/train.py's train_step exactly, but with a compact single-GPU
loop: bf16 autocast, activation checkpointing, batch 1 + grad accumulation.

Two modes:
  --check-only   build model + load weights from vitl.pt, report key match, exit.
  (default)      run --max-steps continued-pretraining steps on our clips.

Canonical ViT-L config (must match the checkpoint):
  encoder  vit_large img256 patch16 tubelet2 rope wide_silu use_sdpa
  predictor depth12 heads12 embed384 mask_tokens10
"""
import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Project root: override with env var DT_ROOT on other machines (e.g. cloud VM).
ROOT = os.environ.get("DT_ROOT", "/mnt/d/Video/domain_transfer")
REPO = f"{ROOT}/repos/vjepa2"
CKPT = f"{ROOT}/weights/vjepa2_vitl.pt"
DS_FILE = f"{ROOT}/scripts/20_clip_dataset.py"

# standard V-JEPA multiblock masking: one short-range (many small blocks),
# one long-range (few large blocks). spatial_scale = fraction of HxW kept per block.
CFGS_MASK = [
    dict(spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
         aspect_ratio=(0.75, 1.5), num_blocks=8, max_temporal_keep=1.0),
    dict(spatial_scale=(0.70, 0.70), temporal_scale=(1.0, 1.0),
         aspect_ratio=(0.75, 1.5), num_blocks=2, max_temporal_keep=1.0),
]


def load_ds_module():
    spec = importlib.util.spec_from_file_location("clipds", DS_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cycle(dl):
    """Iterate a dataloader indefinitely (multi-epoch) until the caller breaks."""
    while True:
        for batch in dl:
            yield batch


def clean_key(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def build_models(device, frames, ckpt_encoder=True):
    sys.path.insert(0, REPO)
    from app.vjepa.utils import init_video_model
    encoder, predictor = init_video_model(
        device=device, model_name="vit_large", crop_size=256,
        patch_size=16, tubelet_size=2, max_num_frames=64,
        pred_depth=12, pred_num_heads=12, pred_embed_dim=384,
        uniform_power=False, use_mask_tokens=True, num_mask_tokens=10,
        use_sdpa=True, use_rope=True, use_silu=False, use_pred_silu=False,
        wide_silu=True, use_activation_checkpointing=True,
    )
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    report = {}
    if ckpt_encoder:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        enc_sd = clean_key(ck["encoder"])
        tgt_sd = clean_key(ck.get("target_encoder", ck["encoder"]))
        pred_sd = clean_key(ck["predictor"])
        r1 = encoder.backbone.load_state_dict(enc_sd, strict=False)
        r2 = target_encoder.backbone.load_state_dict(tgt_sd, strict=False)
        r3 = predictor.backbone.load_state_dict(pred_sd, strict=False)
        report = dict(
            enc_missing=len(r1.missing_keys), enc_unexpected=len(r1.unexpected_keys),
            tgt_missing=len(r2.missing_keys), tgt_unexpected=len(r2.unexpected_keys),
            pred_missing=len(r3.missing_keys), pred_unexpected=len(r3.unexpected_keys),
            enc_missing_sample=r1.missing_keys[:5], enc_unexpected_sample=r1.unexpected_keys[:5],
            pred_missing_sample=r3.missing_keys[:5], pred_unexpected_sample=r3.unexpected_keys[:5],
        )
    return encoder, predictor, target_encoder, report


def make_collate(frames):
    sys.path.insert(0, REPO)
    from src.masks.multiseq_multiblock3d import MaskCollator
    mc = MaskCollator(cfgs_mask=CFGS_MASK, dataset_fpcs=[frames],
                      crop_size=(256, 256), patch_size=(16, 16), tubelet_size=2)
    gens = mc.mask_generators[frames]

    def collate(batch):  # batch: list of clips [C,T,H,W]
        clip = torch.stack(batch, 0)                      # [B,C,T,H,W]
        masks_enc, masks_pred = [], []
        for g in gens:
            me, mp = g(len(batch))
            masks_enc.append(me)
            masks_pred.append(mp)
        # outer list = per-fpc (single fpc here) -> matches train_step structure
        return [clip], [masks_enc], [masks_pred]

    return collate, mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{ROOT}/manifests/train_videos.txt")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--loss-exp", type=float, default=1.0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--save", default="", help="path to save adapted target_encoder")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    encoder, predictor, target_encoder, report = build_models(device, args.frames)
    if report:
        print("=== weight load report ===")
        for k, v in report.items():
            print(f"  {k}: {v}")
    if args.check_only:
        print("check-only done.")
        return

    ds_mod = load_ds_module()
    ds = ds_mod.VideoClipDataset(args.manifest, num_frames=args.frames,
                                 stride=args.stride, size=256, train=True)
    collate, mc = make_collate(args.frames)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=2, collate_fn=collate, drop_last=True)

    from src.masks.utils import apply_masks
    params = [p for p in encoder.parameters() if p.requires_grad] + \
             [p for p in predictor.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    encoder.train(); predictor.train(); target_encoder.eval()

    step = 0
    opt.zero_grad()
    print(f"=== training: frames={args.frames} batch={args.batch} accum={args.accum} "
          f"max_steps={args.max_steps} ===")
    for clips, masks_enc, masks_pred in cycle(dl):  # loop epochs until max_steps
        clips = [c.to(device, non_blocking=True) for c in clips]
        masks_enc = [[m.to(device) for m in me] for me in masks_enc]
        masks_pred = [[m.to(device) for m in mp] for mp in masks_pred]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                h = target_encoder(clips)
                h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
            z = encoder(clips, masks_enc)
            z = predictor(z, masks_enc, masks_pred)
            h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
            loss, n = 0.0, 0
            for zi, hi in zip(z, h):
                for zij, hij in zip(zi, hi):
                    loss = loss + torch.mean(torch.abs(zij - hij) ** args.loss_exp) / args.loss_exp
                    n += 1
            loss = loss / n

        (loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            opt.step(); opt.zero_grad()
            with torch.no_grad():  # EMA target update
                for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                    pk.mul_(args.ema).add_(pq.detach(), alpha=1 - args.ema)

        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1e9
        print(f"  step {step:3d}  loss={loss.detach().item():.4f}  vram_used={used:.2f}GB")
        step += 1
        mc.step()
        if step >= args.max_steps:
            break

    if args.save:
        torch.save({"target_encoder": target_encoder.state_dict()}, args.save)
        print("saved adapted target_encoder ->", args.save)
    print("DONE.")


if __name__ == "__main__":
    main()
