import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from flip_evaluator import evaluate as flip
from rendering.neural.dataset import RenderDataset
from rendering.neural.models import GAN


if __name__ == '__main__':
    dataset = RenderDataset("dataset_normal_max/")
    _, _, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1],
                                                            torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=15)

    ckpt_path = "ckpts/best-model-v14.ckpt"
    model = GAN.load_from_checkpoint(ckpt_path)
    model.eval()
    flip_scores = []
    with torch.no_grad():
        for imgs, conds in test_loader:
            conds = conds.type_as(imgs).to(model.device)
            generated_imgs = model(conds)

            for pred, target in zip(generated_imgs, imgs):
                # FLIP expects numpy RGB, HWC in [0,1], float32
                pred = (pred + 1) / 2  # to [0,1]
                pred_np = pred[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
                target_np = target[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
                flip_scores.append(flip(target_np, pred_np, 'LDR')[1])

    mean_flip = sum(flip_scores) / len(flip_scores)
    print(f"Mean FLIP score: {mean_flip}")

    n = 6
    gen_imgs = (generated_imgs[:n] + 1) / 2
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    axes[0, 0].set_ylabel("Pred", fontsize=12)
    axes[1, 0].set_ylabel("GT", fontsize=12)
    for i, (pred, gt) in enumerate(zip(gen_imgs.cpu(), imgs[:n, :3])):
        pred = pred.permute(1, 2, 0).numpy()
        gt = gt.permute(1, 2, 0).numpy()
        ax = axes[0, i]
        ax.imshow(pred)
        # ax.axis('off')
        ax = axes[1, i]
        ax.imshow(gt)
        # ax.axis('off')
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(left=0.1)
    plt.tight_layout()
    plt.savefig("flip_results.png", dpi=300)


