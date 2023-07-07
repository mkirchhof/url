from plots_constants import *
import matplotlib.transforms as transforms

def remove_architecture_name(text):
    return [t.replace(" ResNet 50", "").replace(" ViT Medium", "") for t in text]

with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    avg = chosen_avg["best_test_avg_downstream_auroc_correct_mean"].to_list()
    yerrs = np.abs(np.stack(((chosen_avg["best_test_avg_downstream_auroc_correct_min"] - chosen_avg["best_test_avg_downstream_auroc_correct_mean"]).to_list(),
                      (chosen_avg["best_test_avg_downstream_auroc_correct_max"] - chosen_avg["best_test_avg_downstream_auroc_correct_mean"]).to_list()),
                     axis=0))
    ax.bar(np.arange(len(chosen_avg)),
           avg,
           color=[id_to_col[sweep] for sweep in chosen_avg["Sweep"].to_list()],
           yerr=yerrs,
           zorder=2
           )

    # Set benchmarks
    #baseline = 0.621
    baseline2 = 0.684
    #ax.axhline(y=baseline, color="black", dashes=(4, 4))
    ax.axhline(y=baseline2, color="black", dashes=(4, 4), linewidth=1)
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    #ax.text(1.01, baseline, "ResNet 50 trained on downstream data, but not test classes", color="black", transform=trans,
    #        ha="right", va="bottom")

    plt.xticks(np.arange(len(chosen_avg)), remove_architecture_name(chosen_avg["Sweep"].to_list()), rotation=50, ha="right")
    font_size = plt.gcf().get_axes()[0].get_xticklabels()[0].get_fontsize()

    ax.text(0.025, baseline2 + 0.003, "ResNet 50 trained on downstream test classes", color="black",
            transform=trans,
            ha="left", va="bottom", fontsize=1.2 * font_size)
    plt.text(4.5, 0.44, "ResNet 50", fontsize=font_size, ha="center")
    plt.text(14.5, 0.44, "ViT Medium", fontsize=font_size, ha="center")
    ax.set_ylabel("Downstream R-AUROC")
    ax.set_ylim(bottom=0.5)
    ax.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.savefig("benchmark.pdf")
    plt.savefig("benchmark.png", dpi=300)
    plt.close()

with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(2.63)
    avg = chosen_avg["best_test_avg_downstream_auroc_correct_mean"].to_list()
    yerrs = np.abs(np.stack(((chosen_avg["best_test_avg_downstream_auroc_correct_min"] - chosen_avg["best_test_avg_downstream_auroc_correct_mean"]).to_list(),
                      (chosen_avg["best_test_avg_downstream_auroc_correct_max"] - chosen_avg["best_test_avg_downstream_auroc_correct_mean"]).to_list()),
                     axis=0))
    ax.bar(np.arange(len(chosen_avg)),
           avg,
           color=[id_to_col[sweep] for sweep in chosen_avg["Sweep"].to_list()],
           yerr=yerrs,
           zorder=2
           )

    # Set benchmarks
    #baseline = 0.621
    baseline2 = 0.684
    #ax.axhline(y=baseline, color="black", dashes=(4, 4))
    ax.axhline(y=baseline2, color="black", dashes=(4, 4), linewidth=1)
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    #ax.text(1.01, baseline, "ResNet 50 trained on downstream data, but not test classes", color="black", transform=trans,
    #        ha="right", va="bottom")

    plt.xticks(np.arange(len(chosen_avg)), remove_architecture_name(chosen_avg["Sweep"].to_list()), rotation=50, ha="right")
    font_size = plt.gcf().get_axes()[0].get_xticklabels()[0].get_fontsize()

    ax.text(0.025, baseline2 + 0.003, "ResNet 50 trained on downstream test classes", color="black",
            transform=trans,
            ha="left", va="bottom", fontsize=1.2 * font_size)
    plt.text(4.5, 0.425, "ResNet 50", fontsize=font_size, ha="center")
    plt.text(14.5, 0.425, "ViT Medium", fontsize=font_size, ha="center")
    ax.set_ylabel("Downstream R-AUROC")
    ax.set_ylim(bottom=0.5)
    ax.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.savefig("benchmark_poster.png", dpi=1200)
    plt.close()

with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    avg = chosen_r1_avg["best_test_avg_downstream_r1_mean"].to_list()
    yerrs = np.abs(np.stack(((chosen_r1_avg["best_test_avg_downstream_r1_min"] - chosen_r1_avg["best_test_avg_downstream_r1_mean"]).to_list(),
                      (chosen_r1_avg["best_test_avg_downstream_r1_max"] - chosen_r1_avg["best_test_avg_downstream_r1_mean"]).to_list()),
                     axis=0))
    ax.bar(np.arange(len(chosen_r1_avg)),
           avg,
           yerr=yerrs,
           color=[id_to_col[s] for s in chosen_r1_avg["Sweep"].to_list()],
           zorder=2
           )

    # Set benchmarks
    #baseline = 0.519
    baseline2 = 0.538
    #ax.axhline(y=baseline, color="black", dashes=(4, 4))
    ax.axhline(y=baseline2, color="black", dashes=(4, 4), linewidth=1)
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    #ax.text(1.01, baseline, "ResNet 50 trained on downstream data, but not test classes", color="black", transform=trans,
    #        ha="right", va="bottom")
    plt.xticks(np.arange(len(chosen_r1_avg)), remove_architecture_name(chosen_r1_avg["Sweep"].to_list()), rotation=50, ha="right")
    font_size = plt.gcf().get_axes()[0].get_xticklabels()[0].get_fontsize()

    ax.text(0.025, baseline2 + 0.01, "ResNet 50 trained on downstream test classes", color="black",
            transform=trans,
            ha="left", va="bottom", fontsize=1.2 * font_size)
    plt.text(4.5, -0.18, "ResNet 50", fontsize=font_size, ha="center")
    plt.text(14.5, -0.18, "ViT Medium", fontsize=font_size, ha="center")
    ax.set_ylabel("Downstream R@1")
    ax.set_ylim(bottom=0)
    ax.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.savefig("benchmark_r1.pdf")
    plt.close()
