import torch
import matplotlib.pyplot as plt
from flip_evaluator import evaluate as flip
from rendering.neural.dataset import RenderDataset
from rendering.neural.models import GAN


def evaluate(model, test_loader):
    flip_scores = []
    with torch.no_grad():
        for imgs, conds in test_loader:
            conds = conds.type_as(imgs).to(model.device)
            generated_imgs = model(conds)
            err_maps, last_scores = [], []
            for pred, target in zip(generated_imgs, imgs):
                # FLIP expects numpy RGB, HWC in [0,1], float32
                pred = (pred + 1) / 2  # to [0,1]
                pred_np = pred[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
                target_np = target[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
                err_map, score, _ = flip(target_np, pred_np, 'LDR')
                flip_scores.append(score)
                last_scores.append(score)
                err_maps.append(err_map)

    mean_flip = sum(flip_scores) / len(flip_scores)
    print(f"Mean FLIP score: {mean_flip}")
    with open("flip_scores.csv", "a+") as f:
        f.write(f"{ckpt_path},{mean_flip}\n")
    return generated_imgs, imgs, err_maps, last_scores


def plot(generated_imgs, imgs, err_maps, last_scores):
    n = 10
    gen_imgs = (generated_imgs[:n] + 1) / 2
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 3 * 2))
    axes[0, 0].set_ylabel("Predicted", fontsize=18, )
    axes[1, 0].set_ylabel("Ground Truth", fontsize=18, )
    axes[2, 0].set_ylabel("FLIP Error", fontsize=18, )
    for i, (pred, gt, err, flip_score) in enumerate(zip(gen_imgs.cpu(), imgs[:n, :3], err_maps, last_scores)):
        pred = pred.permute(1, 2, 0).numpy()
        gt = gt.permute(1, 2, 0).numpy()
        ax = axes[0, i]
        ax.imshow(pred)
        # ax.axis('off')
        ax = axes[1, i]
        ax.imshow(gt)
        # ax.axis('off')
        ax = axes[2, i]
        ax.imshow(err)
        # ax.axis('off')
        axes[2, i].set_xlabel(round(flip_score, 4), fontsize=15, )
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.subplots_adjust(left=0.1)
    plt.tight_layout()
    plt.savefig("flip_results.png", dpi=300)


if __name__ == '__main__':
    dataset = RenderDataset("dataset_normal_max/", noisy=False)
    _, _, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1],
                                                   torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=256,
                                              shuffle=True,
                                              generator=torch.Generator().manual_seed(71),
                                              num_workers=15)

    ckpt_path = "ckpts/twilight-butterfly-116/best-model.ckpt"
    model = GAN.load_from_checkpoint(ckpt_path)
    model.eval()

    generated_imgs, imgs, err_maps, last_scores = evaluate(model, test_loader)
    plot(generated_imgs, imgs, err_maps, last_scores)
