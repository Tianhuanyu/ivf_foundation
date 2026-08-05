#!/usr/bin/env python3
"""DINOv3 domain-adaptive continued pretraining (DAPT) via DINO self-distillation.

Init a student + EMA teacher from the released DINOv3 weights (HF transformers
format), self-distill on our extracted microscopy frames with multi-crop, and
save the adapted backbone. Compact single-GPU loop (bf16 + grad ckpt), sized for
12GB but scales to remote by raising batch/crops/out-dim/steps.

Modes:
  --check-only   build student/teacher + one multicrop forward, report shapes, exit.
  (default)      run --max-steps DINO self-distillation steps.

Microscopy frames are effectively grayscale -> we drop color jitter/grayscale
augs and keep flip + blur + solarize.
"""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from transformers import AutoModel

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ---------------- multi-crop augmentation ----------------
class GaussianBlur:
    def __init__(self, p=0.5, r=(0.1, 2.0)):
        self.p, self.r = p, r

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        from PIL import ImageFilter
        radius = self.r[0] + torch.rand(1).item() * (self.r[1] - self.r[0])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class Solarize:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        from PIL import ImageOps
        return ImageOps.solarize(img) if torch.rand(1).item() < self.p else img


class MultiCrop:
    def __init__(self, global_crops=2, local_crops=6,
                 global_scale=(0.4, 1.0), local_scale=(0.05, 0.4)):
        norm = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(MEAN, STD)])
        flip = transforms.RandomHorizontalFlip(0.5)
        self.n_global, self.n_local = global_crops, local_crops
        self.g1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_scale, interpolation=Image.BICUBIC),
            flip, GaussianBlur(1.0), norm])
        self.g2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_scale, interpolation=Image.BICUBIC),
            flip, GaussianBlur(0.1), Solarize(0.2), norm])
        self.l = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_scale, interpolation=Image.BICUBIC),
            flip, GaussianBlur(0.5), norm])

    def __call__(self, img):
        img = img.convert("RGB")
        crops = [self.g1(img), self.g2(img)]
        for _ in range(self.n_local):
            crops.append(self.l(img))
        return crops


def multicrop_collate(batch):
    # batch: list of (crops_list, label); regroup into list of batched crop tensors
    n = len(batch[0][0])
    return [torch.stack([b[0][i] for b in batch]) for i in range(n)]


# ---------------- model ----------------
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck))
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last(x)


class Backbone(nn.Module):
    """DINOv3 ViT -> CLS embedding."""
    def __init__(self, path, grad_ckpt=True):
        super().__init__()
        self.vit = AutoModel.from_pretrained(path)
        if grad_ckpt:
            self.vit.gradient_checkpointing_enable()
        self.embed_dim = self.vit.config.hidden_size

    def forward(self, x):
        return self.vit(x).last_hidden_state[:, 0]  # CLS token


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, crops):
        # crops: list of [B,C,h,w] tensors, possibly mixed resolution
        sizes = torch.tensor([c.shape[-1] for c in crops])
        idx = torch.cumsum(torch.unique_consecutive(sizes, return_counts=True)[1], 0)
        start, embs = 0, []
        for end in idx:
            out = self.backbone(torch.cat(crops[start:end]))
            embs.append(out)
            start = end
        return self.head(torch.cat(embs))


# ---------------- DINO loss ----------------
class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_global, n_crops, teacher_temp=0.04,
                 student_temp=0.1, center_m=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_m = center_m
        self.n_global, self.n_crops = n_global, n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        s = (student_out / self.student_temp).chunk(self.n_crops)
        t = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)
        t = t.detach().chunk(self.n_global)
        loss, n = 0.0, 0
        for ti in range(len(t)):
            for si in range(len(s)):
                if si == ti:
                    continue
                loss = loss + torch.sum(-t[ti] * F.log_softmax(s[si], dim=-1), dim=-1).mean()
                n += 1
        self.update_center(teacher_out)
        return loss / n

    @torch.no_grad()
    def update_center(self, teacher_out):
        c = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_m).add_(c, alpha=1 - self.center_m)


def main():
    ap = argparse.ArgumentParser()
    _root = os.environ.get("DT_ROOT", "/mnt/d/Video/domain_transfer")
    ap.add_argument("--frames-dir", default=f"{_root}/frames/train")
    ap.add_argument("--weights", default=f"{_root}/weights/dinov3_vits16")
    ap.add_argument("--global-crops", type=int, default=2)
    ap.add_argument("--local-crops", type=int, default=6)
    ap.add_argument("--out-dim", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    n_crops = args.global_crops + args.local_crops

    student = MultiCropWrapper(Backbone(args.weights),
                               DINOHead(384 if "vits" in args.weights else 768, args.out_dim)).to(device)
    teacher = MultiCropWrapper(Backbone(args.weights, grad_ckpt=False),
                               DINOHead(384 if "vits" in args.weights else 768, args.out_dim)).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"student params (M): {sum(p.numel() for p in student.parameters())/1e6:.1f}")

    tf = MultiCrop(args.global_crops, args.local_crops)
    ds = ImageFolder(args.frames_dir, transform=tf)
    print(f"dataset images: {len(ds)}  classes(stages): {len(ds.classes)}")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=4, collate_fn=multicrop_collate,
                                     drop_last=True, pin_memory=True)

    if args.check_only:
        crops = next(iter(dl))
        crops = [c.to(device) for c in crops]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s_out = student(crops)
            t_out = teacher(crops[:args.global_crops])
        print(f"student_out: {tuple(s_out.shape)} (expect [{n_crops*args.batch}, {args.out_dim}])")
        print(f"teacher_out: {tuple(t_out.shape)} (expect [{args.global_crops*args.batch}, {args.out_dim}])")
        print("check-only done.")
        return

    loss_fn = DINOLoss(args.out_dim, args.global_crops, n_crops,
                       args.teacher_temp, args.student_temp).to(device)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    student.train(); teacher.eval()

    step = 0
    print(f"=== DINOv3 DAPT: crops={args.global_crops}g+{args.local_crops}l "
          f"batch={args.batch} out_dim={args.out_dim} max_steps={args.max_steps} ===")
    while step < args.max_steps:
        for crops in dl:
            crops = [c.to(device, non_blocking=True) for c in crops]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_out = student(crops)
                with torch.no_grad():
                    t_out = teacher(crops[:args.global_crops])
                loss = loss_fn(s_out.float(), t_out.float())
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():  # EMA teacher
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(args.ema).add_(ps.detach(), alpha=1 - args.ema)
            free, total = torch.cuda.mem_get_info()
            print(f"  step {step:3d}  loss={loss.detach().item():.4f}  "
                  f"vram_used={(total-free)/1e9:.2f}GB")
            step += 1
            if step >= args.max_steps:
                break

    if args.save:
        student.backbone.vit.save_pretrained(args.save)
        print("saved adapted DINOv3 backbone ->", args.save)
    print("DONE.")


if __name__ == "__main__":
    main()
