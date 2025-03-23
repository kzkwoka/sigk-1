import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm

from torchvision.transforms import transforms, InterpolationMode
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS, PeakSignalNoiseRatio as PSNR, \
    StructuralSimilarityIndexMeasure as SSIM

from model import *
from dataset import ImagePairDataset

runs = pd.read_csv('best_runs.csv')
runs['hash'] = runs['name'].str.split("-").str[-1]

lpips = LPIPS(normalize=True)
psnr = PSNR()
ssim = SSIM()


def get_kwargs(name):
    with open(f"logs/wandb/{name}/files/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    kwargs = dict(model=eval(config.get("model").get("value")),
                  color_space=config.get("color_space").get("value"),
                  model_kwargs={'channels': config.get("channels").get("value"),
                                'upscale_factor': 256 // config.get("input_size").get("value")},
                  lr_kwargs={'lr': config.get("lr").get("value"), 'step_size': config.get("step_size").get("value"),
                             'gamma': config.get("gamma").get("value")})
    return kwargs, config


def load_model(hash, kwargs):
    return SRNet.load_from_checkpoint(checkpoint_path=f"logs/SIGK/{hash}/checkpoints/epoch=499-step=12500.ckpt",
                                      **kwargs)


def rgb_to_ycbcr(image):
    ycbcr = image.convert("YCbCr")
    return ycbcr


def get_transforms(config):
    transform = transforms.Compose([
        # scale the LR image to (shorter side)
        transforms.Resize(config.get("input_size").get("value"), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.get("input_size").get("value")),

        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC) if config.get("use_bicubic").get(
            "value") else transforms.Lambda(lambda x: x),
        transforms.Lambda(rgb_to_ycbcr) if config.get("color_space").get("value") == "ycbcr" else transforms.Lambda(
            lambda x: x),
        transforms.ToTensor()
    ])

    original_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),  # scale the HR image to 256 (shorter side)
        transforms.CenterCrop(256),  # crop the HR image to 256x256
        transforms.Lambda(rgb_to_ycbcr) if config.get("color_space").get("value") == "ycbcr" else transforms.Lambda(
            lambda x: x),
        # target always in RGB
        transforms.ToTensor()
    ])
    return transform, original_transform


def get_validation_dataset(config):
    transform, original_transform = get_transforms(config)
    return ImagePairDataset(image_dir="div2k/DIV2K_valid_HR",
                            input_transform=transform,
                            target_transform=original_transform)


def forward(model, image, config, device="cuda"):
    x = image[0, :, :] if config.get("channels").get("value") == 1 else image
    y = model(x.unsqueeze(0).to(device))
    return y


def get_missing_channels(y, image, config):
    if config.get("channels").get("value") == 1:
        out = torch.cat(
            (y.cpu(), transforms.Resize(256, interpolation=InterpolationMode.BICUBIC)(image[1:, :, :])), dim=0)
    else:
        out = y.squeeze(0).cpu()
    return out


def get_rgb(out, config):
    mode = "YCbCr" if config.get("color_space").get("value") == "ycbcr" else "RGB"
    img = transforms.ToPILImage(mode=mode)(out.squeeze().cpu())
    return img

def get_y_and_rgb(out, target, config):
    if config.get("color_space").get("value") == "ycbcr":
        out_y, target_y = out[0, :, :], target[0, :, :]
        out_rgb = transforms.ToTensor()(transforms.ToPILImage(mode="YCbCr")(out).convert("RGB"))
        target_rgb = transforms.ToTensor()(transforms.ToPILImage(mode="YCbCr")(target).convert("RGB"))
    else:
        out_rgb, target_rgb = out, target
        out_y = transforms.ToTensor()(transforms.ToPILImage(mode="RGB")(out).convert("YCbCr"))[0, :, :]
        target_y = transforms.ToTensor()(transforms.ToPILImage(mode="RGB")(target).convert("YCbCr"))[0, :, :]
    return out_y, target_y, out_rgb, target_rgb


def get_metrics(out_y, target_y, out_rgb, target_rgb):
    out_rgb, target_rgb = out_rgb.unsqueeze(0), target_rgb.unsqueeze(0)
    _lpips = lpips(out_rgb, target_rgb).item()
    _psnr = psnr(out_y, target_y).item()
    _ssim = ssim(out_rgb, target_rgb).item()
    _mse = torch.nn.functional.mse_loss(out_y, target_y).item()
    return _mse, _psnr, _ssim, _lpips


def evaluate(model, dataset, config, device="cuda"):
    scores = {
        'lpips': [],
        'psnr': [],
        'ssim': [],
        'mse': []
    }
    for image, target in tqdm(dataset):
        #image and target are tensors
        y = forward(model, image, config, device)
        out = get_missing_channels(y, image, config)
        # target and out are tensors in the same color space
        out_y, target_y, out_rgb, target_rgb = get_y_and_rgb(out, target, config)
        s = get_metrics(out_y, target_y, out_rgb, target_rgb)
        scores['mse'].append(s[0])
        scores['psnr'].append(s[1])
        scores['ssim'].append(s[2])
        scores['lpips'].append(s[3])
    return scores


if __name__ == '__main__':
    results_df = pd.DataFrame()
    for _, r in runs.iterrows():
        print(r['name'])
        kwargs, cfg = get_kwargs(r['name'])
        model = load_model(r['hash'], kwargs)
        dataset = get_validation_dataset(cfg)
        scores = evaluate(model, dataset, cfg)
        results = {'description': r["desc"]}
        for key, value in scores.items():
            results[key] = np.mean(value)
        results_df = pd.concat([results_df, pd.DataFrame([results])])
        # print(results_df)
        # break
    results_df.to_csv('evaluation_results.csv', mode='a', index=False)
