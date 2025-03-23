import torch
import torchvision.transforms.functional as TF


def rgb_to_ycbcr(image):
    ycbcr = image.convert("YCbCr")
    return ycbcr


def ycrcb_to_rgb(y, cr, cb):
    ycrcb = torch.cat((y, cr.unsqueeze(1), cb.unsqueeze(1)), dim=1)
    rgb_batch = []
    for img in ycrcb:
        rgb = TF.to_pil_image(img.cpu(), mode="YCbCr").convert("RGB")
        rgb_batch.append(TF.to_tensor(rgb).to(y.device))
    return torch.stack(rgb_batch)
