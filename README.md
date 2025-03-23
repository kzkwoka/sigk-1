# Image Modification

## Virtual Environment
### Create python environment
```
conda create -p <ENV_PATH> python=3.11
```

```
conda activate <ENV_PATH>
```

### Install requirements
```
pip install -r requirements.txt
```

### Set PYTHONPATH
```
export PYTHONPATH=$(pwd)
```

## Debluring experiments
### Run example training
```
python deblurring/train.py --data_path DIV2K_train_HR_resized/ --img_size 256 --batch_size 16 --epochs 30 --lr 0.0001 --model cnn --save_path /results/debluring_kernel_5/model.pth
```
### Run baseline evaluation
```
python deblurring/eval.py --data_path DIV2K_valid_HR_resized --kernel_size 5
```
### Run model checkpoint evaluation
```
python deblurring/eval.py --data_path DIV2K_valid_HR_resized --model_ckpt model.pth --model_type cnn --kernel_size 5
```