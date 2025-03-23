# Resizing experiments

## Training
python resizing/train.py --model srcnn --use_bicubic --channels 3  --epochs 500 --step_size 200 --input_size 64

## Evaluation
python resizing/eval.py

## Visualizations
For upsizing single images and visualizations see [Jupyter notebook](resizing/evaluation.ipynb)