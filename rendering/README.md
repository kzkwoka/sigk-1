# Rendering

## Dataset
To create the dataset, run the following command:
```bash
cd rendering/renderer
python src/main.py
```
By default, the dataset will be saved in `rendering/renderer/dataset`.
You can change the output directory by modifying the `output_dir` parameter in `src/main.py`.

## Training
```bash
cd rendering
python3 neural/train_gan.py
```

## Evaluation
To evaluate the model, change the model path in the file and run the following command:
```bash
cd rendering
python3 neural/evaluate.py
```

## Connecting to renderer
The trained model can be then used to render the images. In the `renderer/src/main.py` file, 
change the model path to the trained model in `renderer/src/phong_neural_window.py`
Then run the following command:
```bash
cd rendering/renderer
python src/main.py --neural
```
It will generate the images into the `rendering/renderer/dataset` folder.

### Rendering given scenes
To render a given scene, you can place the scene parameters in the `rendering/renderer/resources/params.csv`
and then run the following command:
```bash
cd rendering/renderer
python src/main.py --neural --reference
```