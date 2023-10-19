from plots_constants import *

res_ood = pd.read_csv("ood.csv")

# Find out which sweep is which:
res_ood["SweepID"] = res_ood["Sweep"]
res_ood["Sweep"][(res_ood["loss"] == "cross-entropy") & (res_ood["model"] == "resnet50")] = "CE ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "cross-entropy") & (res_ood["model"] == "vit_medium_patch16_gap_256")] = "CE ViT Medium"
res_ood["Sweep"][(res_ood["loss"] == "infonce") & (res_ood["model"] == "resnet50")] = "InfoNCE ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "infonce") & (res_ood["model"] == "vit_medium_patch16_gap_256")] = "InfoNCE ViT Medium"
res_ood["Sweep"][(res_ood["loss"] == "mcinfonce") & (res_ood["model"] == "resnet50")] = "MCInfoNCE ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "mcinfonce") & (res_ood["model"] == "vit_medium_patch16_gap_256")] = "MCInfoNCE ViT Medium"
res_ood["Sweep"][(res_ood["loss"] == "elk") & (res_ood["model"] == "resnet50")] = "ELK ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "elk") & (res_ood["model"] == "vit_medium_patch16_gap_256")] = "ELK ViT Medium"
res_ood["Sweep"][(res_ood["loss"] == "hib") & (res_ood["model"] == "resnet50")] = "HIB ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "hib") & (res_ood["model"] == "vit_medium_patch16_gap_256")] = "HIB ViT Medium"
res_ood["Sweep"][res_ood["model"] == "resnet50hetxl"] = "HET-XL ResNet 50"
res_ood["Sweep"][res_ood["model"] == "vit_medium_patch16_gap_256hetxl"] = "HET-XL ViT Medium"
res_ood["Sweep"][(res_ood["loss"] == "riskpred") & (res_ood["model"] == "resnet50")] = "Losspred ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "riskpred") & (res_ood["model"] == "vit_medium_patch16_gap_256")] = "Losspred ViT Medium"
res_ood["Sweep"][res_ood["model"] == "resnet50dropout"] = "MCDropout ResNet 50"
res_ood["Sweep"][res_ood["model"] == "vit_medium_patch16_gap_256dropout"] = "MCDropout ViT Medium"
res_ood["Sweep"][(res_ood["loss"] == "cross-entropy") & (res_ood["model"] == "resnet50") & (res_ood["num-heads"] > 1)] = "Ensemble ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "cross-entropy") & (res_ood["model"] == "vit_medium_patch16_gap_256") & (res_ood["num-heads"] > 1)] = "Ensemble ViT Medium"
res_ood["Sweep"][res_ood["model"] == "resnet50sngp"] = "SNGP ResNet 50"
res_ood["Sweep"][res_ood["model"] == "vit_medium_patch16_gap_256sngp"] = "SNGP ViT Medium"
res_ood["Sweep"][(res_ood["loss"] == "nivmf") & (res_ood["model"] == "resnet50")] = "nivMF ResNet 50"
res_ood["Sweep"][(res_ood["loss"] == "nivmf") & (res_ood["model"] == "vit_medium_patch16_gap_256")] = "nivMF ViT Medium"

# Add plotting parameters
res_ood["color"] = [id_to_col[i] for i in res_ood["Sweep"].to_list()]
res_ood["marker"] = [id_to_marker[i] for i in res_ood["Sweep"].to_list()]
res_ood["size"] = [id_to_size[i] for i in res_ood["Sweep"].to_list()]
res_ood["is_resnet"] = ["ResNet" in sweep for sweep in res_ood["Sweep"].to_list()]

res_ood = res_ood.groupby('Sweep').agg(
    {'best_test_avg_downstream_auroc_correct': ['mean', 'min', 'max'],
     'best_test_avg_downstream_r1': ['mean', 'min', 'max'],
     'best_test_avg_downstream_auroc_ood': ['mean', 'min', 'max'],
     'best_test_avg_downstream_auroc_correct_mixed': ['mean', 'min', 'max'],
     'eval_time_per_sample': ['mean', 'min', 'max'],
     "is_resnet": ['mean']})
res_ood = res_ood.set_axis(res_ood.columns.map('_'.join), axis=1, inplace=False)
res_ood = res_ood.sort_values(["is_resnet_mean", "best_test_avg_downstream_auroc_ood_mean"], ascending=[False, False])
res_ood = res_ood.reset_index()

# Benchmark of OOD detection
def remove_architecture_name(text):
    return [t.replace(" ResNet 50", "").replace(" ViT Medium", "") for t in text]

with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    avg = res_ood["best_test_avg_downstream_auroc_ood_mean"].to_list()
    yerrs = np.abs(np.stack(((res_ood["best_test_avg_downstream_auroc_ood_min"] - res_ood["best_test_avg_downstream_auroc_ood_mean"]).to_list(),
                      (res_ood["best_test_avg_downstream_auroc_ood_max"] - res_ood["best_test_avg_downstream_auroc_ood_mean"]).to_list()),
                     axis=0))
    ax.bar(np.arange(len(res_ood)),
           avg,
           color=[id_to_col[sweep] for sweep in res_ood["Sweep"].to_list()],
           yerr=yerrs,
           zorder=2
           )

    plt.xticks(np.arange(len(res_ood)), remove_architecture_name(res_ood["Sweep"].to_list()), rotation=50, ha="right")
    font_size = plt.gcf().get_axes()[0].get_xticklabels()[0].get_fontsize()

    plt.text(4.5, -0.22, "ResNet 50", fontsize=font_size, ha="center")
    plt.text(14.5, -0.22, "ViT Medium", fontsize=font_size, ha="center")
    ax.set_ylabel("OOD AUROC")
    ax.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.savefig("benchmark_ood.pdf")
    plt.close()

res_ood = res_ood.sort_values(["is_resnet_mean", "best_test_avg_downstream_auroc_correct_mixed_mean"], ascending=[False, False])
res_ood = res_ood.reset_index()
with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    avg = res_ood["best_test_avg_downstream_auroc_correct_mixed_mean"].to_list()
    yerrs = np.abs(np.stack(((res_ood["best_test_avg_downstream_auroc_correct_mixed_min"] - res_ood["best_test_avg_downstream_auroc_correct_mixed_mean"]).to_list(),
                      (res_ood["best_test_avg_downstream_auroc_correct_mixed_max"] - res_ood["best_test_avg_downstream_auroc_correct_mixed_mean"]).to_list()),
                     axis=0))
    ax.bar(np.arange(len(res_ood)),
           avg,
           color=[id_to_col[sweep] for sweep in res_ood["Sweep"].to_list()],
           yerr=yerrs,
           zorder=2
           )

    plt.xticks(np.arange(len(res_ood)), remove_architecture_name(res_ood["Sweep"].to_list()), rotation=50, ha="right")
    font_size = plt.gcf().get_axes()[0].get_xticklabels()[0].get_fontsize()

    plt.text(4.5, -0.22, "ResNet 50", fontsize=font_size, ha="center")
    plt.text(14.5, -0.22, "ViT Medium", fontsize=font_size, ha="center")
    ax.set_ylabel("R-AUROC on mixed up/downstream data")
    ax.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.savefig("benchmark_mixed.pdf")
    plt.close()

res_ood = res_ood.sort_values(["is_resnet_mean", "best_test_avg_downstream_auroc_correct_mixed_mean"], ascending=[False, False])
res_ood = res_ood.reset_index()
with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    avg = res_ood["best_test_avg_downstream_auroc_correct_mixed_mean"].to_list()
    yerrs = np.abs(np.stack(((res_ood["best_test_avg_downstream_auroc_correct_mixed_min"] - res_ood["best_test_avg_downstream_auroc_correct_mixed_mean"]).to_list(),
                      (res_ood["best_test_avg_downstream_auroc_correct_mixed_max"] - res_ood["best_test_avg_downstream_auroc_correct_mixed_mean"]).to_list()),
                     axis=0))
    ax.bar(np.arange(len(res_ood)),
           avg,
           color=[id_to_col[sweep] for sweep in res_ood["Sweep"].to_list()],
           yerr=yerrs,
           zorder=2
           )

    plt.xticks(np.arange(len(res_ood)), remove_architecture_name(res_ood["Sweep"].to_list()), rotation=50, ha="right")
    font_size = plt.gcf().get_axes()[0].get_xticklabels()[0].get_fontsize()

    plt.text(4.5, -0.22, "ResNet 50", fontsize=font_size, ha="center")
    plt.text(14.5, -0.22, "ViT Medium", fontsize=font_size, ha="center")
    ax.set_ylabel("Eval Time (")
    ax.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.savefig("benchmark_mixed.pdf")
    plt.close()