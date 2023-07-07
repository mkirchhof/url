import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from tueplots import bundles
bundles.neurips2023(family="sans-serif", usetex=False, nrows=1)
plt.rcParams.update(bundles.neurips2023())

# basic colors
BLUE = "#4878d0"
RED = "#d65f5f"
GREEN = "#6acc64"
ORANGE = "#ee854a"
GREY = "#797979"
BLACK = "#000000"

id_to_col = {
    "CE ResNet 50": "tab:blue",
    "CE ViT Medium": "tab:blue",
    "InfoNCE ResNet 50": "tab:orange",
    "InfoNCE ViT Medium": "tab:orange",
    "MCInfoNCE ResNet 50": "tab:green",
    "MCInfoNCE ViT Medium": "tab:green",
    "ELK ResNet 50": "tab:red",
    "ELK ViT Medium": "tab:red",
    "HIB ResNet 50": "tab:purple",
    "HIB ViT Medium": "tab:purple",
    "HET-XL ResNet 50": "tab:brown",
    "HET-XL ViT Medium": "tab:brown",
    "Riskpred ResNet 50": "tab:pink",
    "Riskpred ViT Medium": "tab:pink",
    "MCDropout ResNet 50": "tab:olive",
    "MCDropout ViT Medium": "tab:olive",
    "Ensemble ResNet 50": "tab:cyan",
    "Ensemble ViT Medium": "tab:cyan",
    "SNGP ResNet 50": "tab:gray",
    "SNGP ViT Medium": "tab:gray",
    "nivMF ResNet 50": "#008080",
    "nivMF ViT Medium": "#008080"
}
id_to_marker = {
    "CE ResNet 50": "^",
    "CE ViT Medium": "^",
    "InfoNCE ResNet 50": "v",
    "InfoNCE ViT Medium": "v",
    "MCInfoNCE ResNet 50": "s",
    "MCInfoNCE ViT Medium": "s",
    "ELK ResNet 50": "p",
    "ELK ViT Medium": "p",
    "HIB ResNet 50": "P",
    "HIB ViT Medium": "P",
    "HET-XL ResNet 50": "*",
    "HET-XL ViT Medium": "*",
    "Riskpred ResNet 50": "H",
    "Riskpred ViT Medium": "H",
    "MCDropout ResNet 50": "X",
    "MCDropout ViT Medium": "X",
    "Ensemble ResNet 50": "D",
    "Ensemble ViT Medium": "D",
    "SNGP ResNet 50": "o",
    "SNGP ViT Medium": "o",
    "nivMF ResNet 50": "h",
    "nivMF ViT Medium": "h"
}
id_to_size = {
    "CE ResNet 50": 12.0,
    "CE ViT Medium": 46.0,
    "InfoNCE ResNet 50": 12.0,
    "InfoNCE ViT Medium": 46.0,
    "MCInfoNCE ResNet 50": 9.0,
    "MCInfoNCE ViT Medium": 39.0,
    "ELK ResNet 50": 15.0,
    "ELK ViT Medium": 49.0,
    "HIB ResNet 50": 11.0,
    "HIB ViT Medium": 46.0,
    "HET-XL ResNet 50": 13.0,
    "HET-XL ViT Medium": 54.0,
    "Riskpred ResNet 50": 14.0,
    "Riskpred ViT Medium": 46.0,
    "MCDropout ResNet 50": 11.0,
    "MCDropout ViT Medium": 43.0,
    "Ensemble ResNet 50": 9.0,
    "Ensemble ViT Medium": 37.0,
    "SNGP ResNet 50": 13.0,
    "SNGP ViT Medium": 39.0,
    "nivMF ResNet 50": 14.0,
    "nivMF ViT Medium": 46.0
}
legend_approaches = ["CE",
                "InfoNCE",
                "MCInfoNCE",
                "ELK",
                "nivMF",
                "HIB",
                "HET-XL",
                "Riskpred",
                "MCDropout",
                "Ensemble",
                "SNGP"]
legend_handles = [mlines.Line2D([], [], label=approach,
                  color=id_to_col[approach + " ResNet 50"],
                  marker=id_to_marker[approach + " ResNet 50"],
                  linestyle='None', markersize=5) for approach in legend_approaches]
legend_handles.append(mlines.Line2D([], [], label="", color="white",linestyle='None', markersize=5))
legend_handles.append(mlines.Line2D([], [], label="", linestyle='None'))
legend_handles.append(mlines.Line2D([], [], label="", linestyle='None'))
#legend_handles.append(mlines.Line2D([], [], label="grey = non-optimal", linestyle='None'))

# Load data
res = pd.read_csv("wandb_export.csv")

# Scale accuracy to [0, 1]
res["best_eval_top1"] = res["best_eval_top1"] / 100

# Find out which sweep is which:
res["SweepID"] = res["Sweep"]
res["Sweep"][(res["loss"] == "cross-entropy") & (res["model"] == "resnet50")] = "CE ResNet 50"
res["Sweep"][(res["loss"] == "cross-entropy") & (res["model"] == "vit_medium_patch16_gap_256")] = "CE ViT Medium"
res["Sweep"][(res["loss"] == "infonce") & (res["model"] == "resnet50")] = "InfoNCE ResNet 50"
res["Sweep"][(res["loss"] == "infonce") & (res["model"] == "vit_medium_patch16_gap_256")] = "InfoNCE ViT Medium"
res["Sweep"][(res["loss"] == "mcinfonce") & (res["model"] == "resnet50")] = "MCInfoNCE ResNet 50"
res["Sweep"][(res["loss"] == "mcinfonce") & (res["model"] == "vit_medium_patch16_gap_256")] = "MCInfoNCE ViT Medium"
res["Sweep"][(res["loss"] == "elk") & (res["model"] == "resnet50")] = "ELK ResNet 50"
res["Sweep"][(res["loss"] == "elk") & (res["model"] == "vit_medium_patch16_gap_256")] = "ELK ViT Medium"
res["Sweep"][(res["loss"] == "hib") & (res["model"] == "resnet50")] = "HIB ResNet 50"
res["Sweep"][(res["loss"] == "hib") & (res["model"] == "vit_medium_patch16_gap_256")] = "HIB ViT Medium"
res["Sweep"][res["model"] == "resnet50hetxl"] = "HET-XL ResNet 50"
res["Sweep"][res["model"] == "vit_medium_patch16_gap_256hetxl"] = "HET-XL ViT Medium"
res["Sweep"][(res["loss"] == "riskpred") & (res["model"] == "resnet50")] = "Riskpred ResNet 50"
res["Sweep"][(res["loss"] == "riskpred") & (res["model"] == "vit_medium_patch16_gap_256")] = "Riskpred ViT Medium"
res["Sweep"][res["model"] == "resnet50dropout"] = "MCDropout ResNet 50"
res["Sweep"][res["model"] == "vit_medium_patch16_gap_256dropout"] = "MCDropout ViT Medium"
res["Sweep"][(res["loss"] == "cross-entropy") & (res["model"] == "resnet50") & (res["num-heads"] > 1)] = "Ensemble ResNet 50"
res["Sweep"][(res["loss"] == "cross-entropy") & (res["model"] == "vit_medium_patch16_gap_256") & (res["num-heads"] > 1)] = "Ensemble ViT Medium"
res["Sweep"][res["model"] == "resnet50sngp"] = "SNGP ResNet 50"
res["Sweep"][res["model"] == "vit_medium_patch16_gap_256sngp"] = "SNGP ViT Medium"
res["Sweep"][(res["loss"] == "nivmf") & (res["model"] == "resnet50")] = "nivMF ResNet 50"
res["Sweep"][(res["loss"] == "nivmf") & (res["model"] == "vit_medium_patch16_gap_256")] = "nivMF ViT Medium"

# Add plotting parameters
res["color"] = [id_to_col[i] for i in res["Sweep"].to_list()]
res["marker"] = [id_to_marker[i] for i in res["Sweep"].to_list()]
res["size"] = [id_to_size[i] for i in res["Sweep"].to_list()]
res["is_resnet"] = ["ResNet" in sweep for sweep in res["Sweep"].to_list()]

# Filter out bad runs
res = res[res["best_eval_avg_downstream_r1"] > 0.1]

# Add final chosen hyperparameters for each setup:
chosen = res[
    ((res["Sweep"] == "CE ResNet 50") & ((res["Name"] == "stellar-sweep-1") | (res["Name"] == "pretty-sweep-2") | (res["Name"] ==  "drawn-sweep-3"))) |
    ((res["Sweep"] == "CE ViT Medium") & ((res["Name"] == "deft-sweep-8") | (res["Name"] == "snowy-sweep-1") | (res["Name"] ==  "noble-sweep-2"))) |
    ((res["Sweep"] == "InfoNCE ResNet 50") & ((res["Name"] == "rural-sweep-8") | (res["Name"] == "dauntless-sweep-1") | (res["Name"] ==  "restful-sweep-2"))) |
    ((res["Sweep"] == "InfoNCE ViT Medium") & ((res["Name"] == "dauntless-sweep-10") | (res["Name"] == "pious-sweep-1") | (res["Name"] ==  "sage-sweep-2"))) |
    ((res["Sweep"] == "MCInfoNCE ResNet 50") & ((res["Name"] == "glamorous-sweep-10") | (res["Name"] == "wandering-sweep-1") | (res["Name"] ==  "comfy-sweep-2"))) |
    ((res["Sweep"] == "MCInfoNCE ViT Medium") & ((res["Name"] == "comic-sweep-8") | (res["Name"] == "resilient-sweep-1") | (res["Name"] ==  "lunar-sweep-2"))) |
    ((res["Sweep"] == "ELK ResNet 50") & ((res["Name"] == "iconic-sweep-4") | (res["Name"] == "sleek-sweep-2") | (res["Name"] ==  "super-sweep-1"))) |
    ((res["Sweep"] == "ELK ViT Medium") & ((res["Name"] == "neat-sweep-7") | (res["Name"] == "twilight-sweep-1") | (res["Name"] ==  "hearty-sweep-2"))) |
    ((res["Sweep"] == "HIB ResNet 50") & ((res["Name"] == "lunar-sweep-14") | (res["Name"] == "crimson-sweep-1") | (res["Name"] ==  "neat-sweep-2"))) |
    ((res["Sweep"] == "HIB ViT Medium") & ((res["Name"] == "efficient-sweep-7") | (res["Name"] == "logical-sweep-1") | (res["Name"] ==  "pious-sweep-2"))) |
    ((res["Sweep"] == "HET-XL ResNet 50") & ((res["Name"] == "winter-sweep-9") | (res["Name"] == "lucky-sweep-1") | (res["Name"] ==  "genial-sweep-2"))) |
    ((res["Sweep"] == "HET-XL ViT Medium") & ((res["Name"] == "expert-sweep-18") | (res["Name"] == "breezy-sweep-1") | (res["Name"] ==  "graceful-sweep-2"))) |
    ((res["Sweep"] == "Riskpred ResNet 50") & ((res["Name"] == "absurd-sweep-6") | (res["Name"] == "generous-sweep-1") | (res["Name"] ==  "clean-sweep-2"))) |
    ((res["Sweep"] == "Riskpred ViT Medium") & ((res["Name"] == "jolly-sweep-9") | (res["Name"] == "smooth-sweep-1") | (res["Name"] ==  "soft-sweep-2"))) |
    ((res["Sweep"] == "MCDropout ResNet 50") & ((res["Name"] == "glorious-sweep-9") | (res["Name"] == "lucky-sweep-1") | (res["Name"] ==  "serene-sweep-2"))) |
    ((res["Sweep"] == "MCDropout ViT Medium") & ((res["Name"] == "deft-sweep-1") | (res["Name"] == "still-sweep-1") | (res["Name"] ==  "distinctive-sweep-2"))) |
    ((res["Sweep"] == "Ensemble ResNet 50") & ((res["Name"] == "desert-sweep-6") | (res["Name"] == "firm-sweep-1") | (res["Name"] ==  "scarlet-sweep-2"))) |
    ((res["Sweep"] == "Ensemble ViT Medium") & ((res["Name"] == "comfy-sweep-10") | (res["Name"] == "pleasant-sweep-1") | (res["Name"] ==  "fluent-sweep-2"))) |
    ((res["Sweep"] == "SNGP ResNet 50") & ((res["Name"] == "chocolate-sweep-6") | (res["Name"] == "crimson-sweep-1") | (res["Name"] ==  "woven-sweep-2"))) |
    ((res["Sweep"] == "SNGP ViT Medium") & ((res["Name"] == "floral-sweep-2") | (res["Name"] == "twilight-sweep-1") | (res["Name"] ==  "classic-sweep-2"))) |
    ((res["Sweep"] == "nivMF ResNet 50") & ((res["Name"] == "rural-sweep-4") | (res["Name"] == "faithful-sweep-1") | (res["Name"] ==  "toasty-sweep-2"))) |
    ((res["Sweep"] == "nivMF ViT Medium") & ((res["Name"] == "giddy-sweep-9") | (res["Name"] == "polished-sweep-1") | (res["Name"] ==  "zesty-sweep-2")))
]
chosen_avg = chosen.groupby('Sweep').agg(
    {'best_test_avg_downstream_auroc_correct': ['mean', 'min', 'max'],
     'best_test_avg_downstream_r1': ['mean', 'min', 'max'],
     "is_resnet":['mean']})
chosen_avg = chosen_avg.set_axis(chosen_avg.columns.map('_'.join), axis=1, inplace=False)
chosen_avg = chosen_avg.sort_values(["is_resnet_mean", "best_test_avg_downstream_auroc_correct_mean"], ascending=[False, False])
chosen_avg = chosen_avg.reset_index()

# Add final chosen hyperparameters for each setup chosen w.r.t. R@1:
chosen_r1 = res[
    ((res["Sweep"] == "CE ResNet 50") & ((res["Name"] == "dainty-sweep-1") | (res["Name"] == "youthful-sweep-2") | (res["Name"] ==  "dashing-sweep-3"))) |
    ((res["Sweep"] == "CE ViT Medium") & ((res["Name"] == "worthy-sweep-2") | (res["Name"] == "chocolate-sweep-1") | (res["Name"] ==  "spring-sweep-2"))) |
    ((res["Sweep"] == "InfoNCE ResNet 50") & ((res["Name"] == "rural-sweep-8") | (res["Name"] == "dauntless-sweep-1") | (res["Name"] == "restful-sweep-2"))) |
    ((res["Sweep"] == "InfoNCE ViT Medium") & ((res["Name"] == "bumbling-sweep-6") | (res["Name"] == "apricot-sweep-1") | (res["Name"] ==  "graceful-sweep-2"))) |
    ((res["Sweep"] == "MCInfoNCE ResNet 50") & ((res["Name"] == "glamorous-sweep-10") | (res["Name"] == "wandering-sweep-1") | (res["Name"] ==  "comfy-sweep-2"))) |
    ((res["Sweep"] == "MCInfoNCE ViT Medium") & ((res["Name"] == "comic-sweep-8") | (res["Name"] == "resilient-sweep-1") | (res["Name"] ==  "lunar-sweep-2"))) |
    ((res["Sweep"] == "ELK ResNet 50") & ((res["Name"] == "breezy-sweep-9") | (res["Name"] == "fallen-sweep-1") | (res["Name"] ==  "gentle-sweep-2"))) |
    ((res["Sweep"] == "ELK ViT Medium") & ((res["Name"] == "neat-sweep-7") | (res["Name"] == "twilight-sweep-1") | (res["Name"] ==  "hearty-sweep-2"))) |
    ((res["Sweep"] == "HIB ResNet 50") & ((res["Name"] == "valiant-sweep-11") | (res["Name"] == "lilac-sweep-1") | (res["Name"] ==  "feasible-sweep-2"))) |
    ((res["Sweep"] == "HIB ViT Medium") & ((res["Name"] == "polar-sweep-4") | (res["Name"] == "polished-sweep-1") | (res["Name"] ==  "northern-sweep-2"))) |
    ((res["Sweep"] == "HET-XL ResNet 50") & ((res["Name"] == "different-sweep-4") | (res["Name"] == "lilac-sweep-1") | (res["Name"] ==  "toasty-sweep-2"))) |
    ((res["Sweep"] == "HET-XL ViT Medium") & ((res["Name"] == "expert-sweep-18") | (res["Name"] == "breezy-sweep-1") | (res["Name"] ==  "graceful-sweep-2"))) |
    ((res["Sweep"] == "Riskpred ResNet 50") & ((res["Name"] == "ethereal-sweep-10") | (res["Name"] == "absurd-sweep-1") | (res["Name"] ==  "hardy-sweep-2"))) |
    ((res["Sweep"] == "Riskpred ViT Medium") & ((res["Name"] == "solar-sweep-10") | (res["Name"] == "driven-sweep-1") | (res["Name"] ==  "rose-sweep-2"))) |
    ((res["Sweep"] == "MCDropout ResNet 50") & ((res["Name"] == "glorious-sweep-9") | (res["Name"] == "lucky-sweep-1") | (res["Name"] ==  "serene-sweep-2"))) |
    ((res["Sweep"] == "MCDropout ViT Medium") & ((res["Name"] == "icy-sweep-3") | (res["Name"] == "hardy-sweep-1") | (res["Name"] ==  "sweepy-sweep-2"))) |
    ((res["Sweep"] == "Ensemble ResNet 50") & ((res["Name"] == "devout-sweep-1") | (res["Name"] == "robust-sweep-1") | (res["Name"] ==  "fluent-sweep-2"))) |
    ((res["Sweep"] == "Ensemble ViT Medium") & ((res["Name"] == "comfy-sweep-10") | (res["Name"] == "pleasant-sweep-1") | (res["Name"] ==  "fluent-sweep-2"))) |
    ((res["Sweep"] == "SNGP ResNet 50") & ((res["Name"] == "stilted-sweep-10") | (res["Name"] == "amber-sweep-1") | (res["Name"] ==  "amber-sweep-2"))) |
    ((res["Sweep"] == "SNGP ViT Medium") & ((res["Name"] == "exalted-sweep-5") | (res["Name"] == "vocal-sweep-1") | (res["Name"] ==  "spring-sweep-2"))) |
    ((res["Sweep"] == "nivMF ResNet 50") & ((res["Name"] == "worthy-sweep-6") | (res["Name"] == "restful-sweep-1") | (res["Name"] ==  "winter-sweep-2"))) |
    ((res["Sweep"] == "nivMF ViT Medium") & ((res["Name"] == "giddy-sweep-9") | (res["Name"] == "polished-sweep-1") | (res["Name"] ==  "zesty-sweep-2")))
]
chosen_r1_avg = chosen_r1.groupby('Sweep').agg(
    {'best_test_avg_downstream_auroc_correct': ['mean', 'min', 'max'],
     'best_test_avg_downstream_r1': ['mean', 'min', 'max'],
     "is_resnet":['mean']})
chosen_r1_avg = chosen_r1_avg.set_axis(chosen_r1_avg.columns.map('_'.join), axis=1, inplace=False)
chosen_r1_avg = chosen_r1_avg.sort_values(["is_resnet_mean", "best_test_avg_downstream_r1_mean"], ascending=[False, False])
chosen_r1_avg = chosen_r1_avg.reset_index()