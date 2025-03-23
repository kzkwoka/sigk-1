import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import restoration
from torch.utils.data import DataLoader
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchvision.transforms import ToTensor
from tqdm import tqdm

from deblurring.dataset import DeblurringDataset
from deblurring.model import CNN, AutoEncoder


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    lpips = LPIPS()
    psnr = PSNR()
    ssim = SSIM()

    scores = {"lpips": [], "psnr": [], "ssim": [], "mse": []}
    criterion = torch.nn.MSELoss()

    total_loss = 0.0

    for masked_images, originals in tqdm(dataloader, desc="Model Evaluation"):
        masked_images = masked_images.to(device)
        outputs = model(masked_images).cpu()

        loss = criterion(outputs, originals)
        total_loss += loss.item()

        for i in range(outputs.size(0)):
            output_img = outputs[i].unsqueeze(0)
            target_img = originals[i].unsqueeze(0)

            scores["lpips"].append(lpips(output_img, target_img).item())
            scores["psnr"].append(psnr(output_img, target_img).item())
            scores["ssim"].append(ssim(output_img, target_img).item())
            scores["mse"].append(F.mse_loss(output_img, target_img).item())

    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    scores["loss"] = total_loss / len(dataloader)
    return scores


@torch.no_grad()
def evaluate_baseline(dataloader: DataLoader, kernel_size: int, sigma: float, num_iter: int) -> dict:
    to_tensor = ToTensor()
    lpips = LPIPS()
    psnr = PSNR()
    ssim = SSIM()

    scores = {"lpips": [], "psnr": [], "ssim": [], "mse": []}

    psf = cv2.getGaussianKernel(kernel_size, sigma)
    psf = psf @ psf.T

    for blur_tensor, sharp_tensor in tqdm(dataloader, desc="Baseline Evaluation"):
        for i in range(blur_tensor.size(0)):
            blur_np = blur_tensor[i].permute(1, 2, 0).numpy().astype(np.float32)
            sharp_np = sharp_tensor[i].permute(1, 2, 0).numpy().astype(np.float32)

            deconvolved = np.zeros_like(blur_np)
            for c in range(3):
                deconvolved[..., c] = restoration.richardson_lucy(blur_np[..., c], psf, num_iter=num_iter)
            deconvolved = np.clip(deconvolved, 0, 1)

            deconvolved_tensor = to_tensor((deconvolved * 255).astype(np.uint8)).unsqueeze(0)
            sharp_tensor_single = sharp_tensor[i].unsqueeze(0)

            scores["lpips"].append(lpips(deconvolved_tensor, sharp_tensor_single).item())
            scores["psnr"].append(psnr(deconvolved_tensor, sharp_tensor_single).item())
            scores["ssim"].append(ssim(deconvolved_tensor, sharp_tensor_single).item())
            scores["mse"].append(F.mse_loss(deconvolved_tensor, sharp_tensor_single).item())

    return {k: sum(v) / len(v) for k, v in scores.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deblurring model or baseline.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset images")
    parser.add_argument("--model_ckpt", type=str, help="Path to trained model weights")
    parser.add_argument("--model_type", type=str, choices=["cnn", "ae"], help="Model architecture")
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="Kernel size for Gaussian blur (baseline)",
    )
    parser.add_argument("--sigma", type=float, default=0.0, help="Sigma for Gaussian PSF (baseline)")
    parser.add_argument(
        "--num_iter",
        type=int,
        default=30,
        help="Number of Richardson-Lucy iterations (baseline)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = DeblurringDataset(args.data_path, kernel_size=args.kernel_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_ckpt:
        assert args.model_type is not None, "You must specify --model_type when using --model_ckpt"
        model = CNN() if args.model_type == "cnn" else AutoEncoder()
        model.load_state_dict(torch.load(args.model_ckpt, map_location="cpu"))
        model.to(device)
        results = evaluate_model(model, dataloader, device)
        print("Model Evaluation:")
    else:
        results = evaluate_baseline(dataloader, args.kernel_size, args.sigma, args.num_iter)
        print("Baseline Evaluation:")

    for k, v in results.items():
        print(f"{k.upper()}: {v:.4f}")


if __name__ == "__main__":
    main()
