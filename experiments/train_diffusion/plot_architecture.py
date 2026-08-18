#!/usr/bin/env python3
"""
plot_architecture.py
Visualise the TemporalUnet diffusion backbone as a publication-quality diagram.

Usage:
    python experiments/train_diffusion/plot_architecture.py
    python experiments/train_diffusion/plot_architecture.py --out figures/architecture.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style

apply_style(grid=False)

# Wong colorblind-safe palette
C_ENC    = "#0072B2"   # blue   — encoder blocks
C_DEC    = "#D55E00"   # orange — decoder blocks
C_NECK   = "#CC79A7"   # pink   — bottleneck
C_COND   = "#009E73"   # green  — conditioning branches
C_SKIP   = "#E69F00"   # amber  — skip-connection arrows
C_FINAL  = "#56B4E9"   # sky    — final conv / output
C_GRAY   = "#BBBBBB"   # light grey — arrows / outlines

ALPHA = 0.85


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, color, label="", sublabel="", alpha=ALPHA, lw=0.8):
    """Draw a rounded rectangle and optional centred text."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03",
        linewidth=lw,
        edgecolor="white",
        facecolor=color,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x, y + (0.10 if sublabel else 0), label,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.13, sublabel,
                ha="center", va="center", fontsize=6.5,
                color="white", alpha=0.9, zorder=4, style="italic")


def arrow(ax, x0, y0, x1, y1, color=C_GRAY, lw=1.0, style="->", zorder=2):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color,
                        lw=lw, connectionstyle="arc3,rad=0.0"),
        zorder=zorder,
    )


def skip_arrow(ax, x0, y0, x1, y1):
    """Curved skip connection arrow."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_SKIP,
            lw=1.3,
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=5,
    )


def label(ax, x, y, text, fontsize=7, color="#333333", ha="center", va="center",
          bold=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=color, zorder=6)


# ---------------------------------------------------------------------------
# Axis setup
# ---------------------------------------------------------------------------

def make_figure():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Draw conditioning section (top panel)
# ---------------------------------------------------------------------------

def draw_conditioning(ax, cx=7.0, top=8.7):
    """Three conditioning branches that merge into a single embedding vector."""

    # --- Section header ---
    label(ax, cx, top, "Conditioning embedding  phi = [t_emb || b_emb || r_emb]  in R^{3d}   (d = 128)",
          fontsize=9, bold=True, color="#222222")

    # branch x-positions
    bx = [cx - 3.8, cx, cx + 3.8]
    by = top - 0.38

    S  = 0.30   # box height
    G  = 0.18   # gap between boxes (arrow span)

    # ── time branch ──────────────────────────────────────────────────────────
    t0 = by - 0.00
    box(ax, bx[0], t0,       1.9, S, C_COND, "Diffusion step  t")
    arrow(ax, bx[0], t0 - S/2, bx[0], t0 - S/2 - G)
    t1 = t0 - S/2 - G - S/2
    box(ax, bx[0], t1,       1.9, S, C_COND, "SinusoidalPosEmb", "(B, d)")
    arrow(ax, bx[0], t1 - S/2, bx[0], t1 - S/2 - G)
    t2 = t1 - S/2 - G - S/2
    box(ax, bx[0], t2,       1.9, S, C_COND, "MLP  d->4d->d", "(B, d)")
    time_bottom = t2 - S/2

    # ── belief branch ─────────────────────────────────────────────────────────
    b0 = by - 0.00
    box(ax, bx[1], b0,       2.1, S, C_COND, "GP belief  [mean, var]  (B, 1600)")
    arrow(ax, bx[1], b0 - S/2, bx[1], b0 - S/2 - G)
    b1 = b0 - S/2 - G - S/2
    box(ax, bx[1], b1,       2.1, S, C_COND, "Reshape  ->  (B, 2, 40, 40)")
    arrow(ax, bx[1], b1 - S/2, bx[1], b1 - S/2 - G)
    b2 = b1 - S/2 - G - S/2
    # Summarise CNN as a single taller block with annotation
    cnn_h = 0.55
    box(ax, bx[1], b2,       2.1, cnn_h, C_COND,
        "BeliefEncoder CNN", "(B, 2, 40,40) -> (B, d)")
    # Annotate CNN internals on the right of the block
    cnn_lines = [
        "Conv2d 3x3  2->16   40x40",
        "Conv2d 3x3  16->32  20x20  s2",
        "Conv2d 3x3  32->64  10x10  s2",
        "Conv2d 3x3  64->128  5x5   s2",
        "AvgPool + Linear -> (B,d)",
    ]
    for li, ln in enumerate(cnn_lines):
        ax.text(bx[1] + 1.10, b2 + cnn_h/2 - 0.06 - li * 0.10,
                ln, ha="left", va="center", fontsize=5.2,
                color="#555555", family="monospace", zorder=5)
    arrow(ax, bx[1], b2 - cnn_h/2, bx[1], b2 - cnn_h/2 - G)
    b3 = b2 - cnn_h/2 - G - S/2
    box(ax, bx[1], b3,       2.1, S, C_COND, "CFG dropout (p=0.1)", "(B, d)")
    belief_bottom = b3 - S/2

    # ── return branch ─────────────────────────────────────────────────────────
    r0 = by - 0.00
    box(ax, bx[2], r0,       1.9, S, C_COND, "Target return  r  (B, 1)")
    arrow(ax, bx[2], r0 - S/2, bx[2], r0 - S/2 - G)
    r1 = r0 - S/2 - G - S/2
    box(ax, bx[2], r1,       1.9, S, C_COND, "MLP  1->d->4d->d", "(B, d)")
    arrow(ax, bx[2], r1 - S/2, bx[2], r1 - S/2 - G)
    r2 = r1 - S/2 - G - S/2
    box(ax, bx[2], r2,       1.9, S, C_COND, "CFG dropout (p=0.1)", "(B, d)")
    return_bottom = r2 - S/2

    # ── merge: place below deepest branch ────────────────────────────────────
    merge_y = min(time_bottom, belief_bottom, return_bottom) - G - S/2
    for bxi, from_y in [(bx[0], time_bottom), (bx[1], belief_bottom),
                        (bx[2], return_bottom)]:
        ax.annotate(
            "", xy=(cx, merge_y + S/2), xytext=(bxi, from_y),
            arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.8,
                            connectionstyle="arc3,rad=0.0"),
            zorder=2,
        )

    box(ax, cx, merge_y, 3.5, S, "#444444",
        "Concat  ->  phi  (3d = 384)", "(B, 384)", alpha=0.88)

    return merge_y - S/2   # bottom of conditioning section


# ---------------------------------------------------------------------------
# Draw U-Net section
# ---------------------------------------------------------------------------

def draw_unet(ax, cond_bottom):
    """Draw the temporal U-Net below the conditioning section."""

    unet_top = cond_bottom - 0.18
    label(ax, 7.0, unet_top,
          "Temporal U-Net  (1D convolutions over horizon h)",
          fontsize=9, bold=True, color="#222222")

    # Column x-positions: input | enc cols | bottleneck | dec cols | output
    #  enc: 4 levels at x = 2.5, 3.7, 4.9, 5.8
    #  bottleneck: x = 7.0
    #  dec: 3 levels at x = 8.2, 9.4, 10.6
    #  input  = 1.2,  output = 11.8

    enc_x = [2.5, 3.7, 4.9, 5.8]
    bot_x = 7.0
    dec_x = [8.2, 9.4, 10.6]
    in_x  = 1.2
    row_h  = 0.50
    base_y = unet_top - 0.30

    # We use a fixed y for the encoder row and a symmetric decoder row
    # Layout (y positions from top):
    #   y0 = base_y            — input / first encoder
    #   y1 = y0 - step         — level 1
    #   y2 = y1 - step         — level 2
    #   y3 = y2 - step         — level 3
    #   y_bot = y3 - step      — bottleneck (level 4 enc + mid blocks)

    step  = 0.62
    y_enc = [base_y - i * step for i in range(4)]   # 4 encoder levels
    y_bot = y_enc[-1] - step                         # bottleneck
    y_dec = list(reversed(y_enc[1:]))                # 3 decoder levels (y_enc[3]..y_enc[1])

    # ── input trajectory ─────────────────────────────────────────────────────
    box(ax, in_x, y_enc[0], 1.4, 0.30, "#555555",
        "Trajectory  x_t", "(B, 16, 2)", alpha=0.8)
    arrow(ax, in_x + 0.7, y_enc[0], in_x + 0.85, y_enc[0], lw=1.0)
    label(ax, in_x + 0.73, y_enc[0] + 0.14, "rearrange\n→(B,2,16)",
          fontsize=5.5, color="#555555")

    # Conditioning injection label (right margin)
    ax.text(12.5, (y_enc[0] + y_bot) / 2,
            "φ injected into\nevery ResBlock\nvia time_mlp",
            ha="center", va="center", fontsize=6.5, color="#009E73",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E6F7F2", ec="#009E73", lw=0.6))

    # Arrows from conditioning to unet
    ax.annotate(
        "", xy=(12.0, (y_enc[0] + y_bot) / 2),
        xytext=(8.6, cond_bottom),
        arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=0.9,
                        connectionstyle="arc3,rad=0.15"),
        zorder=2,
    )

    # ── encoder ──────────────────────────────────────────────────────────────
    enc_label  = ["2→128", "128→256", "256→512", "512→1024"]
    enc_horizon= ["h=16", "h=8", "h=4", "h=2"]

    for i in range(4):
        # ResBlocks
        box(ax, enc_x[i], y_enc[i], 1.1, row_h, C_ENC,
            f"2× ResBlock\n{enc_label[i]}",
            enc_horizon[i])
        # Arrows between levels
        if i < 3:
            # Downsample arrow going right and down
            ax.annotate(
                "", xy=(enc_x[i + 1] - 0.55, y_enc[i + 1] + 0.10),
                xytext=(enc_x[i] + 0.55, y_enc[i] - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_ENC, lw=0.8,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2,
            )
            label(ax, (enc_x[i] + enc_x[i + 1]) / 2,
                  (y_enc[i] + y_enc[i + 1]) / 2 + 0.04,
                  "Downsample", fontsize=5.5, color=C_ENC)
        else:
            # Last encoder → bottleneck
            ax.annotate(
                "", xy=(bot_x - 0.60, y_bot + 0.10),
                xytext=(enc_x[i] + 0.55, y_enc[i] - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_ENC, lw=0.8,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2,
            )

    # ── bottleneck ────────────────────────────────────────────────────────────
    box(ax, bot_x, y_bot, 1.5, row_h, C_NECK,
        "2× ResBlock\n1024→1024", "h=2")

    # bottleneck → first decoder
    ax.annotate(
        "", xy=(dec_x[0] - 0.55, y_dec[0] - 0.10),
        xytext=(bot_x + 0.75, y_bot - 0.10),
        arrowprops=dict(arrowstyle="-|>", color=C_NECK, lw=0.8,
                        connectionstyle="arc3,rad=0.0"),
        zorder=2,
    )

    # ── decoder ──────────────────────────────────────────────────────────────
    dec_label   = ["2048→512", "1024→256", "512→128"]
    dec_horizon = ["h=2→4", "h=4→8", "h=8→16"]
    dec_skip_ch = ["(B,1024,2)", "(B,512,4)", "(B,256,8)"]

    for i in range(3):
        box(ax, dec_x[i], y_dec[i], 1.1, row_h, C_DEC,
            f"2× ResBlock\n{dec_label[i]}",
            dec_horizon[i])
        if i < 2:
            ax.annotate(
                "", xy=(dec_x[i + 1] + 0.55, y_dec[i + 1] + 0.10),
                xytext=(dec_x[i] - 0.55, y_dec[i] - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_DEC, lw=0.8,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2,
            )
            label(ax, (dec_x[i] + dec_x[i + 1]) / 2,
                  (y_dec[i] + y_dec[i + 1]) / 2 + 0.04,
                  "Upsample", fontsize=5.5, color=C_DEC)

    # ── skip connections ─────────────────────────────────────────────────────
    # enc level 3 → dec level 0  (skip3: B,1024,2)
    # enc level 2 → dec level 1  (skip2: B,512,4)
    # enc level 1 → dec level 2  (skip1: B,256,8)
    skip_pairs = [
        (enc_x[3], y_enc[3], dec_x[0], y_dec[0], dec_skip_ch[0]),
        (enc_x[2], y_enc[2], dec_x[1], y_dec[1], dec_skip_ch[1]),
        (enc_x[1], y_enc[1], dec_x[2], y_dec[2], dec_skip_ch[2]),
    ]
    for x0, y0, x1, y1, ch in skip_pairs:
        skip_arrow(ax, x0, y0, x1, y1)
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2 - 0.22
        label(ax, mx, my, ch, fontsize=5.5, color=C_SKIP)

    # Note: skip from enc level 0 (B,128,16) is not consumed in decoder
    ax.text(enc_x[0] - 0.7, y_enc[0] - 0.05,
            "skip₀ unused\n(B,128,16)",
            ha="center", va="center", fontsize=5.5, color="#999999",
            style="italic")

    # ── upsample after last decoder ──────────────────────────────────────────
    up_x = dec_x[2] - 0.55
    up_y = y_dec[2] - 0.50
    arrow(ax, dec_x[2] - 0.55, y_dec[2] - 0.25, dec_x[2] - 0.55, up_y + 0.12, lw=0.8)
    label(ax, up_x - 0.22, y_dec[2] - 0.38, "Upsample\nh→16",
          fontsize=5.5, color=C_DEC)

    # ── final conv ────────────────────────────────────────────────────────────
    final_y = y_dec[2] - 0.62
    final_x = dec_x[2] - 0.55
    box(ax, final_x, final_y, 1.55, 0.30, C_FINAL,
        "Conv1dBlock(128→128)", "(B,128,16)")
    arrow(ax, final_x, final_y - 0.15, final_x, final_y - 0.33)
    box(ax, final_x, final_y - 0.48, 1.55, 0.28, C_FINAL,
        "Conv1d(128→2)", "(B,2,16)")
    arrow(ax, final_x, final_y - 0.62, final_x, final_y - 0.78)
    box(ax, final_x, final_y - 0.90, 1.55, 0.25, "#444444",
        "rearrange → output", "(B,16,2)", alpha=0.8)


# ---------------------------------------------------------------------------
# Draw ResBlock legend box
# ---------------------------------------------------------------------------

def draw_legend(ax):
    legend_x, legend_y = 1.1, 1.95
    box(ax, legend_x, legend_y + 0.20, 1.8, 0.98, "#EEEEEE",
        alpha=0.7, lw=0.5)
    ax.text(legend_x, legend_y + 0.62, "ResBlock internals:",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#333333", zorder=7)

    internals = [
        "Conv1d → GroupNorm → Mish",
        "Conv1d → GroupNorm → Mish",
        "+ time_mlp(φ)  ← additive",
        "+ residual projection",
    ]
    for j, txt in enumerate(internals):
        ax.text(legend_x, legend_y + 0.42 - j * 0.19, txt,
                ha="center", va="center", fontsize=5.8,
                color="#444444", zorder=7,
                family="monospace" if j < 2 else "serif")

    # Colour legend
    items = [
        (C_ENC,   "Encoder"),
        (C_DEC,   "Decoder"),
        (C_NECK,  "Bottleneck"),
        (C_COND,  "Conditioning"),
        (C_FINAL, "Final conv"),
        (C_SKIP,  "Skip connection"),
    ]
    lx, ly = 1.5, 1.35
    for k, (col, txt) in enumerate(items):
        row_k = k // 2
        col_k = k % 2
        px = lx + col_k * 1.8
        py = ly - row_k * 0.22
        ax.add_patch(mpatches.Rectangle(
            (px - 1.55, py - 0.07), 0.22, 0.16,
            facecolor=col, edgecolor="none", zorder=7, alpha=0.85))
        ax.text(px - 1.25, py, txt, ha="left", va="center",
                fontsize=7, color="#333333", zorder=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot TemporalUnet architecture.")
    parser.add_argument("--out", type=Path,
                        default=Path("results/architecture.pdf"),
                        help="Output file path (pdf, png, svg).")
    args = parser.parse_args()

    fig, ax = make_figure()

    cond_bottom = draw_conditioning(ax, cx=7.0, top=8.65)
    draw_unet(ax, cond_bottom)
    draw_legend(ax)

    fig.suptitle("AUV Diffusion Planner — TemporalUnet Architecture",
                 fontsize=11, fontweight="bold", y=0.99)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ext = args.out.suffix.lstrip(".")
    fig.savefig(args.out, format=ext, bbox_inches="tight", dpi=200)
    print(f"Saved -> {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
