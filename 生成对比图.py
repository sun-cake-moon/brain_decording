"""Three-stage comparison figure: real photo -> low-level canvas -> final reconstruction.

Row 1  original photo the subject actually saw
Row 2  what the brain signal alone decodes to (blurry "sketch", layout right, no detail)
Row 3  after the diffusion model adds detail, conditioned on the same brain signal

The point of the three rows is that the middle row is where the *brain* information
stops, and the bottom row is what a generative model's prior knowledge adds on top.

All three come from the same 982 test images in the same row order, which was
verified: lowlevel_full/align_s0.6/index.csv and canvas_eval/pred/align_s0.6/index.csv
carry identical nsdId sequences and match subj01_preprocessed.npz test_nsdId.

Usage:
    python 生成对比图.py                 # 12 visually clear examples
    python 生成对比图.py --mode spread   # evenly spread, less cherry-picked
    python 生成对比图.py --n 8 --size 480
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FINAL_DIR = os.path.join(ROOT, "lowlevel_full", "align_s0.6")          # 最终生成
CANVAS_DIR = os.path.join(ROOT, "canvas_eval", "pred", "align_s0.6")   # 低层画布
STIMULI = os.path.join(ROOT, "nsd_stimuli", "nsd_stimuli.hdf5")
DATA = os.path.join(ROOT, "data_processed", "subj01_preprocessed.npz")

# Rows picked earlier (RESULTS Run 011) as the clearest reconstructions. Used only
# for `--mode best`; the spread mode ignores them.
BEST_ROWS = [160, 233, 239, 540, 610, 695, 721, 811, 874, 918, 978, 979]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
]

ROW_LABELS = ["原图（被试看到）", "低层重建（仅脑信号）", "最终生成（+扩散补全）"]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size, index=0)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
    print("[warn] no Chinese font found, labels may show as boxes", file=sys.stderr)
    return ImageFont.load_default()


def load_truth(nsdids, size: int) -> torch.Tensor:
    import h5py
    with h5py.File(STIMULI, "r") as f:
        brick = f["imgBrick"]
        arr = np.stack([np.asarray(brick[int(i)]) for i in nsdids])
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return F.interpolate(x, size=(size, size), mode="bilinear",
                         align_corners=False, antialias=True)


def load_png(paths, size: int) -> torch.Tensor:
    x = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                  for p in paths])
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    return F.interpolate(x, size=(size, size), mode="bilinear",
                         align_corners=False, antialias=True)


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()


def compose(rows, size: int, gap: int, label_w: int, font,
            draw_title: bool) -> Image.Image:
    """rows: list of (label, tensor[N,3,H,W]) -> stacked figure."""
    n = rows[0][1].shape[0]
    nrow = len(rows)
    title_h = int(size * 0.14) if draw_title else 0
    W = label_w + n * size + (n - 1) * gap
    H = title_h + nrow * size + (nrow - 1) * gap
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    for ri, (label, t) in enumerate(rows):
        y = title_h + ri * (size + gap)
        for k in range(n):
            x = label_w + k * (size + gap)
            canvas.paste(Image.fromarray(to_u8(t[k])), (x, y))
        # left-hand row label
        dr = ImageDraw.Draw(canvas)
        # right-align the label against the first tile
        bbox = dr.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        dr.text((max(6, label_w - tw - 14), y + size // 2 - th // 2 - bbox[1]),
                label, fill=(30, 30, 30), font=font)

    if draw_title:
        dr = ImageDraw.Draw(canvas)
        big = load_font(max(20, int(size * 0.085)))
        title = "从大脑活动还原人看到的图像"
        bbox = dr.textbbox((0, 0), title, font=big)
        dr.text(((W - (bbox[2] - bbox[0])) // 2, int(title_h * 0.28)),
                title, fill=(20, 20, 20), font=big)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="三段式对比图")
    ap.add_argument("--mode", default="best", choices=["best", "spread"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=352, help="每格像素边长")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    import pandas as pd
    index = pd.read_csv(os.path.join(FINAL_DIR, "index.csv"))
    n_total = len(index)
    nsdids = index["nsdId"].astype(np.int64).values

    if args.mode == "best":
        rows = [r for r in BEST_ROWS if r < n_total][:args.n]
        note = "示例选自效果较好的样本"
    else:
        step = max(1, n_total // args.n)
        rows = [i * step for i in range(args.n)]
        rows = [r for r in rows if r < n_total]
        note = "示例在全测试集上均匀取样"
    n = len(rows)
    print(f"[pick] {n} 个样本（mode={args.mode}）: {rows}", flush=True)

    fin = load_png([os.path.join(FINAL_DIR, f"{r:04d}.png") for r in rows], args.size)
    can = load_png([os.path.join(CANVAS_DIR, f"{r:04d}.png") for r in rows], args.size)
    tru = load_truth([nsdids[r] for r in rows], args.size)
    print(f"[load] 原图 {tuple(tru.shape)} | 低层 {tuple(can.shape)} | 最终 {tuple(fin.shape)}",
          flush=True)

    label_w = int(args.size * 0.78)
    font = load_font(max(16, int(args.size * 0.062)))

    stacked = [(ROW_LABELS[0], tru), (ROW_LABELS[1], can), (ROW_LABELS[2], fin)]

    main_img = compose(stacked, args.size, args.gap, label_w, font, draw_title=True)
    out1 = os.path.join(args.out_dir, f"对比图_{args.mode}_n{n}.png")
    main_img.save(out1)
    print(f"[save] {out1}  ({main_img.width}x{main_img.height})", flush=True)

    # Label-free version: tiles only, handy when the slide already has labels.
    plain = compose(stacked, args.size, args.gap, 0, font, draw_title=False)
    out2 = os.path.join(args.out_dir, f"对比图_{args.mode}_n{n}_无标签.png")
    plain.save(out2)
    print(f"[save] {out2}  ({plain.width}x{plain.height})", flush=True)

    # Three separate full-width strips, one per stage. More useful for slides than
    # a single stacked image: you can show 原图 -> 低层 -> 最终 one row at a time,
    # or animate the reveal.
    stage_names = ["1_原图", "2_低层重建", "3_最终生成"]
    for (_, t), nm in zip(stacked, stage_names):
        W = n * args.size + (n - 1) * args.gap
        strip = Image.new("RGB", (W, args.size), (255, 255, 255))
        for k in range(n):
            strip.paste(Image.fromarray(to_u8(t[k])), (k * (args.size + args.gap), 0))
        p = os.path.join(args.out_dir, f"单行_{nm}_{args.mode}_n{n}.png")
        strip.save(p)
        print(f"[save] {p}  ({strip.width}x{strip.height})", flush=True)

    # Individual pairs/triples, one file per sample.
    tri_dir = os.path.join(args.out_dir, "单例")
    os.makedirs(tri_dir, exist_ok=True)
    sep = 6
    for k, r in enumerate(rows):
        trio = Image.new("RGB", (3 * args.size + 2 * sep, args.size), (255, 255, 255))
        for j, t in enumerate((tru, can, fin)):
            trio.paste(Image.fromarray(to_u8(t[k])), (j * (args.size + sep), 0))
        trio.save(os.path.join(tri_dir, f"样本{r:04d}.png"))
    print(f"[save] {len(rows)} 个单例 -> {tri_dir}", flush=True)

    print(f"\n说明：{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
