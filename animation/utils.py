import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


def images_to_video(input_dir, output_path, fps=30, ext='jpg'):
    input_dir = Path(input_dir)
    images = sorted([img for img in input_dir.glob(f'*.{ext}')])
    if not images:
        raise ValueError(f"No .{ext} images found in {input_dir}")

    first_frame = cv2.imread(str(images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        video_writer.write(frame)

    video_writer.release()


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)  # from [-1, 1] to [0, 1]


def visualize_model_outputs(model, dataset, num_examples=3):
    model.eval()
    device = next(model.parameters()).device  # get model device

    for i in range(num_examples):
        x0, x1, x2 = dataset[i]

        # Add batch dimension and send to model's device
        input_pair = torch.cat([x0, x2], dim=0).unsqueeze(0).to(device)  # [1, 6, H, W]

        with torch.no_grad():
            pred = model(input_pair)[0].cpu()  # remove batch, move to CPU

        # Prepare plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(TF.to_pil_image(denorm(x0.cpu())))
        axs[0].set_title("Frame t-1")

        axs[1].imshow(TF.to_pil_image(denorm(pred)))
        axs[1].set_title("Predicted Frame t")

        axs[2].imshow(TF.to_pil_image(denorm(x1.cpu())))
        axs[2].set_title("Ground Truth Frame t")

        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
