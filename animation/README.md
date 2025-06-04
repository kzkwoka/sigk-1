# Animation Frame Interpolation with U-Net

This project implements a frame interpolation model using a custom U-Net architecture trained on the IPN Hand dataset. The model predicts intermediate animation frames based on the surrounding ones.

---

## 📦 Project Structure

```
scripts/
  ├── download_ipn_frames.sh     # Download and extract IPN Hand frame data
  ├── split_dataset.py           # Split sequences into train/valid/test folders
  ├── train_model.py             # Train U-Net using PyTorch Lightning
  └── test_model.py              # Evaluate the trained model

src/
  ├── dataset.py                 # PyTorch Dataset for loading triplet frame sequences
  └── model.py                   # U-Net model + LightningModule wrapper
```

---

## 🛠 Setup

```bash
pip install -r requirements.txt
wandb login
```

---

## 📥 Download Data

```bash
bash scripts/download_ipn_frames.sh
```

Then split into train/valid/test:

```bash
python scripts/split_dataset.py
```

---

## 🚀 Train the Model

```bash
python scripts/train_model.py \
  --train_dir frames_merged/train \
  --val_dir frames_merged/valid \
  --batch_size 16 \
  --epochs 50 \
  --run_name baseline_unet
```

---

## 🧪 Test the Model

```bash
python scripts/test_model.py \
  --test_dir frames_merged/test \
  --ckpt_path checkpoints/xx.ckpt \
  --batch_size 16 \
  --run_name test_unet
```

---

## 📊 Logging

All metrics (loss, PSNR, SSIM) are logged automatically to [Weights & Biases](https://wandb.ai/) when configured.

---

## 📁 Output Example

Each input triplet:  
```
[Frame t-1, → predict → Frame t (target), Frame t+1]
```

Predictions can be visualized with `visualize_model_outputs()` for debugging.

---

## ✍️ Citation & Acknowledgements

This project is based on the [IPN Hand Dataset](https://gibranbenitez.github.io/IPN_Hand/).
