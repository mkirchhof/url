from plots_constants import *
import torch
import seaborn as sns

with plt.rc_context(bundles.neurips2023()):
    fig, (ax, ax2) = plt.subplots(ncols=2)
    fig.set_figheight(2)
    for model, axis in zip(["resnet", "vit"], [ax, ax2]):
        unc_cub = torch.load(f"{model}_uncertainties_torch_imagenet_repr_cub.pt")
        unc_cars = torch.load(f"{model}_uncertainties_torch_imagenet_repr_cars.pt")
        unc_sop = torch.load(f"{model}_uncertainties_torch_imagenet_repr_sop.pt")

        # Plot densities. Use 1/x, because what's saved is the uncertainties, which is 1/norm
        sns.kdeplot(1/torch.concat([unc_cub["ID"], unc_cars["ID"], unc_sop["ID"]]), ax=axis, color=GREY, label='ImageNet (ID)')
        sns.kdeplot(1/unc_cub["OOD"], ax=axis, color=BLUE, label='CUB (OOD)')
        sns.kdeplot(1/unc_cars["OOD"], ax=axis, color=RED, label='CARS (OOD)')
        sns.kdeplot(1/unc_sop["OOD"], ax=axis, color=GREEN, label='SOP (OOD)')

        axis.set_xlabel(f"L2 norm of embedding ({'ResNet 50' if model=='resnet' else 'ViT Medium'})")
        axis.legend()
        axis.set_xlim(left=0.)
        if model == 'resnet':
            axis.set_xlim(right=13)

    plt.savefig("ood_densities.pdf")
    plt.close()