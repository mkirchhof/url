from plots_constants import *

few_res = pd.read_csv("fewshot.csv")

# Find out which Sweep is which
few_res["Sweep"][few_res["Sweep"] == "dp17rtrh"] = "ELK ViT Medium"
few_res["Sweep"][few_res["Sweep"] == "cgi9q710"] = "Losspred ViT Medium"
few_res["Sweep"][few_res["Sweep"] == "godeg2dk"] = "SNGP ViT Medium"
few_res["Sweep"][few_res["Sweep"] == "8m8z8g7r"] = "CE ViT Medium"
few_res["Sweep"][few_res["Sweep"] == "dz2xkze5"] = "ELK ResNet 50"
few_res["Sweep"][few_res["Sweep"] == "x7ykknom"] = "nivMF ResNet 50"
few_res["Sweep"][few_res["Sweep"] == "xj7hbxe7"] = "MCInfoNCE ResNet 50"
few_res["Sweep"][few_res["Sweep"] == "lvzxtv9q"] = "MCInfoNCE ResNet 50"
few_res["Sweep"][few_res["Sweep"] == "8ugqjhh7"] = "CE ResNet 50"

# Choose whether we want the models trained on CE or the original losses
use_ce_loss = False
if use_ce_loss:
    few_res.drop(few_res[(few_res["Sweep"] == "ELK ViT Medium") & (few_res["loss"] != "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "Losspred ViT Medium") & (few_res["loss"] != "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "SNGP ViT Medium") & (few_res["model"] != "vit_medium_patch16_gap_256")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "ELK ResNet 50") & (few_res["loss"] != "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "nivMF ResNet 50") & (few_res["loss"] != "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "MCInfoNCE ResNet 50") & (few_res["loss"] != "cross-entropy")].index, inplace=True)
else:
    few_res.drop(few_res[(few_res["Sweep"] == "ELK ViT Medium") & (few_res["loss"] == "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "Losspred ViT Medium") & (few_res["loss"] == "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "SNGP ViT Medium") & (few_res["model"] == "vit_medium_patch16_gap_256")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "ELK ResNet 50") & (few_res["loss"] == "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "nivMF ResNet 50") & (few_res["loss"] == "cross-entropy")].index, inplace=True)
    few_res.drop(few_res[(few_res["Sweep"] == "MCInfoNCE ResNet 50") & (few_res["loss"] == "cross-entropy")].index, inplace=True)

# Average datasets
few_res = few_res.groupby(["initial-checkpoint", "n_few_shot", "Sweep"]).agg(["mean"])
few_res.columns = few_res.columns.droplevel(1)
few_res = few_res.reset_index()

# Summary statistics
few_res = few_res.groupby(['Sweep', "n_few_shot"]).agg(
    {'best_test_avg_downstream_auroc_correct': ['mean', 'min', 'max']})
few_res = few_res.set_axis(few_res.columns.map('_'.join), axis=1, inplace=False)
few_res = few_res.sort_values(["n_few_shot", "Sweep"], ascending=[True, True])
few_res = few_res.reset_index()

# Add plotting parameters
few_res["color"] = [id_to_col[i] for i in few_res["Sweep"].to_list()]
few_res["is_resnet"] = ["ResNet" in sweep for sweep in few_res["Sweep"].to_list()]

# Plot resnet and vit next to each other
with plt.rc_context(bundles.neurips2023()):
    fig, (ax, ax2) = plt.subplots(ncols=2)
    fig.set_figheight(2.4)
    for is_resnet, axis in zip([True, False], [ax, ax2]):
        res = few_res[few_res["is_resnet"] == is_resnet]
        for group in res["Sweep"].unique():
            group_data = res[res["Sweep"] == group]
            color = group_data['color'].iloc[0]

            axis.fill_between(group_data['n_few_shot'], group_data['best_test_avg_downstream_auroc_correct_min'], group_data['best_test_avg_downstream_auroc_correct_max'], color=color, alpha=0.2, zorder=3)
            axis.plot(group_data['n_few_shot'], group_data['best_test_avg_downstream_auroc_correct_mean'], label=group, color=color, marker='o', zorder=4)
        axis.set_xlabel(f"k-shot {'(ResNet 50)' if is_resnet else '(ViT Medium)'}")
        axis.set_ylabel("Downstream R-AUROC")
        axis.grid(zorder=-1, color="lightgrey", lw=0.5)
        axis.legend()

    plt.savefig("few_shot.pdf")