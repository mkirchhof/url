from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2022(family="sans-serif", usetex=False, nrows=1)
plt.rcParams.update(bundles.icml2022())

BLUE = "#4878d0"
RED = "#d65f5f"
GREEN = "#6acc64"
ORANGE = "#ee854a"

# Data as a dictionary
rauroc = OrderedDict({
    "imagenet21k": [0.7913, 0.7909, 0.7918],
    "caltech101": [0.7579, 0.7517, 0.7628],
    "cifar100": [0.7063, 0.7041, 0.7083],
    "dtd": [0.6492, 0.6434, 0.652],
    "oxford_flowers102": [0.6592, 0.65, 0.6622],
    "oxford_iiit_pet": [0.7403, 0.7316, 0.7479],
    "sun397": [0.6912, 0.6902, 0.6928],
    "svhn": [0.4946, 0.4874, 0.505],
    "cifar10": [0.7385, 0.7369, 0.7394],
    "treeversity": [0.5597, 0.5567, 0.5625],
    "cub": [0.6262, 0.6178, 0.6343],
    "cars": [0.5886, 0.5852, 0.5916],
    "sop": [0.6071, 0.6063, 0.6078]
})
datasets_ordered = ["imagenet21k", "caltech101", "oxford_iiit_pet", "cifar10", "cifar100", "sun397",
                "oxford_flowers102", "dtd", "cub", "sop", "cars", "treeversity", "svhn"]
r1 = OrderedDict({
    "imagenet21k": 0.2279,
    "caltech101": 0.8366,
    "cifar100": 0.4665,
    "dtd": 0.6234,
    "oxford_flowers102": 0.986,
    "oxford_iiit_pet": 0.8353,
    "sun397": 0.5849,
    "svhn": 0.2845,
    "cifar10": 0.7092,
    "treeversity": 0.8363,
    "cub": 0.7308,
    "cars": 0.3782,
    "sop": 0.5781
})

# Define RGB colors for each dataset
dataset_colors = {
    "imagenet21k": BLUE,
    "caltech101": RED,
    "cifar100": RED,
    "dtd": RED,
    "oxford_flowers102": RED,
    "oxford_iiit_pet": RED,
    "sun397": RED,
    "svhn": RED,
    "cifar10": RED,
    "treeversity": RED,
    "cub": RED,
    "cars": RED,
    "sop": RED
}

dataset_names = {
    "imagenet21k": "ImageNet-21k (pretraining)",
    "caltech101": "Caltech 101",
    "cifar100": "CIFAR 100",
    "dtd": "DTD",
    "oxford_flowers102": "Oxford Flowers",
    "oxford_iiit_pet": "Oxford Pets",
    "sun397": "SUN",
    "svhn": "SVHN",
    "cifar10": "CIFAR 10",
    "treeversity": "Treeversity",
    "cub": "CUB 200",
    "cars": "CARS 196",
    "sop": "SOP"
}

with plt.rc_context(bundles.neurips2023(rel_width=1.0)):
    fig, ax = plt.subplots()
    fig.set_figheight(1.7)
    # Create bar plot with specified colors
    bars = plt.bar([dataset_names[ds] for ds in datasets_ordered], [rauroc[ds][0] for ds in datasets_ordered], color=[dataset_colors[ds] for ds in datasets_ordered], zorder=5,
                   yerr = np.transpose(np.array([[rauroc[ds][0] - rauroc[ds][1], rauroc[ds][2] - rauroc[ds][0]] for ds in datasets_ordered]), (1, 0)).tolist(), alpha=0.7)
    for bar, name in zip(bars, [dataset_names[ds] for ds in datasets_ordered]):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2 + 0.07, 0.03, name,
                 ha='center', va='bottom', rotation='vertical', zorder=10)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(([0.0, 0.1, 0.2, 0.3, 0.4, "Random", 0.6, 0.7, 0.8]))
    #ax.axhline(y=0.5, color="black", lw=0.8)#, dashes=(4, 4))
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    plt.ylabel("Zero-shot R-AUROC")
    plt.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.savefig("vtab_zero_shot_rauroc_thesis.pdf")
    plt.close()
