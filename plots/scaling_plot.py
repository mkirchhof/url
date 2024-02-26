from plots_constants import *
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2022(family="sans-serif", usetex=False, nrows=1)
plt.rcParams.update(bundles.icml2022())

res = [
    ["ViT-Tiny", 8, 0.352, 0.6109],
    ["ViT-Tiny", 8, 0.352, 0.6103],
    ["ViT-Tiny", 8, 0.352, 0.6095],
    ["ViT-Tiny", 8, 0.352, 0.6091],
    ["ViT-Tiny", 8, 0.352, 0.611],
    ["ViT-Small", 19, 0.5349, 0.6142],
    ["ViT-Small", 19, 0.5349, 0.6181],
    ["ViT-Small", 19, 0.5349, 0.6091],
    ["ViT-Small", 19, 0.5349, 0.6163],
    ["ViT-Small", 19, 0.5349, 0.6145],
    ["ViT-Base", 43, 0.5624, 0.6089],
    ["ViT-Base", 43, 0.5624, 0.6081],
    ["ViT-Base", 43, 0.5624, 0.6088],
    ["ViT-Base", 43, 0.5624, 0.6064],
    ["ViT-Base", 43, 0.5624, 0.6044],
    ["ViT-Large", 66, 0.6594, 0.5855],
    ["ViT-Large", 66, 0.6594, 0.5845],
    ["ViT-Large", 66, 0.6594, 0.5865],
    ["ViT-Large", 66, 0.6594, 0.5828],
    ["ViT-Large", 66, 0.6594, 0.5850],
]

old_best = [
    ["ViT-Medium", 29, 0.49470, 0.56303],
    ["ViT-Medium", 29, 0.52593, 0.57613],
    ["ViT-Medium", 29, 0.55473, 0.56400]
]

old_best_vitbase = [
    ["ViT-Base", 57, 0.5309, 0.5623],
    ["ViT-Base", 57, 0.528, 0.5398],
    ["ViT-Base", 57, 0.548, 0.5757]
]

old_ours_vitbase = [
    ["ViT-Base", 43, 0.6006, 0.5777],
    ["ViT-Base", 43, 0.6006, 0.5904],
    ["ViT-Base", 43, 0.6006, 0.5713]
]

with plt.rc_context(bundles.icml2022(column="full")):
    fig, ax = plt.subplots()
    fig.set_figheight(2.2)

    # Add baselines
    for x, y, color, marker, size in zip(chosen["best_test_avg_downstream_r1"].to_list(),
                                         chosen["best_test_avg_downstream_auroc_correct"].to_list(),
                                         chosen["color"].to_list(),
                                         chosen["marker"].to_list(),
                                         chosen["size"].to_list()):
        if size > 25 and marker != "H":  # Only keep ViT runs
            ax.scatter(x,
                       y,
                       marker="o",
                       c=GREY,
                       s=12,
                       zorder=2,
                       alpha=0.3,
                       edgecolors='none')

    # Highlight old best
    for name, size, x, y in old_best:
        ax.scatter(x,
                   y,
                   c=ORANGE,
                   s=size,
                   zorder=2,
                   marker="*",
                   alpha=0.7,
                   edgecolors='none')

    # Old best reimplementation on ViT base
    for name, size, x, y in old_best_vitbase:
        ax.scatter(x,
                   y,
                   c=GREEN,
                   s=size,
                   zorder=2,
                   marker="*",
                   alpha=0.7,
                   edgecolors='none')

    # Ours reimplementation on ViT Base ImageNet 1k
    for name, size, x, y in old_ours_vitbase:
        ax.scatter(x,
                   y,
                   c=RED,
                   s=size,
                   zorder=2,
                   marker="X",
                   alpha=0.7,
                   edgecolors='none')

    # Ours
    for name, size, x, y in res:
        ax.scatter(x,
                   y,
                   c=BLUE,
                   s=size,
                   zorder=2,
                   marker="X",
                   alpha=0.7,
                   edgecolors='none')

    # Add texts
    ax.text(res[0][2], res[0][3] - 0.008, "ViT-Tiny", ha='center', va='top', color=BLUE)
    ax.text(res[5][2], res[5][3] - 0.009, "ViT-Small", ha='right', va='top', color=BLUE)
    ax.text(res[10][2], res[10][3] - 0.013, "ViT-Base", ha='center', va='top', color=BLUE)
    ax.text(res[15][2] + 0.015, res[15][3] - 0.01, "ViT-Large", ha='right', va='top', color=BLUE)

    # Create legend
    legend_handles = [
        mlines.Line2D([], [], label="Baselines from Kirchhof et al. (2023b),",
                      color=GREY, marker="o", linestyle='None', markersize=4.2, alpha=0.3, markeredgecolor="none"),
        mlines.Line2D(xdata=[], ydata=[], label="ImageNet-1k, ViT-Medium", marker=None, linestyle='None'),
        mlines.Line2D([], [], label="Best Kirchhof et al. (2023b), ImageNet-1k, ViT-Medium",
                      color=ORANGE, marker="*", linestyle='None', markersize=6.5, alpha=0.7, markeredgecolor="none"),
        mlines.Line2D([], [], label="Best Kirchhof et al. (2023b), ImageNet-1k, ViT-Base",
                      color=GREEN, marker="*", linestyle='None', markersize=6.5, alpha=0.7, markeredgecolor="none"),
        mlines.Line2D([], [], label="Ours, ImageNet-1k, ViT-Base",
                      color=RED, marker="X", linestyle='None', markersize=5, alpha=0.7, markeredgecolor="none"),
        mlines.Line2D([], [], label="Ours, ImageNet-21k-W",
                      color=BLUE, marker="X", linestyle='None', markersize=5, alpha=0.7, markeredgecolor="none")
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0, framealpha=0, fontsize=7)

    ax.set_xlabel("Zero-shot Recall@1 (embeddings)")
    ax.set_ylabel("Zero-shot R-AUROC (uncertainties)")
    ax.grid(zorder=-1, color="lightgrey", lw=0.5)
    plt.savefig("scaling.pdf")
    plt.close()
