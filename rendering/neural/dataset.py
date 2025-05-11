import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class RenderDataset(Dataset):
    def __init__(self, data_dir, param_file='params.csv'):
        self.data_dir = data_dir
        self.images = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])

        self.camera_pos = (5.0, 5.0, 15.0)
        self._load_conditions(param_file)

        self.tensor = ToTensor()

    def _load_conditions(self, fname):
        self.params_df = pd.read_csv(os.path.join(self.data_dir, fname), header=None)
        self.params_df.columns = ['id',
                                  'trans_x', 'trans_y', 'trans_z',
                                  'diffuse_r', 'diffuse_g', 'diffuse_b',
                                  'shine',
                                  'light_x', 'light_y', 'light_z']
        self.params_df['id'] = self.params_df['id'].astype(int)
        self.params_df = self.params_df.set_index('id')
        self._translate_conditions()
        self._normalize_conditions()

        other_columns = ['shine', 'diffuse_r', 'diffuse_g', 'diffuse_b']
        self.relative = [c for c in self.params_df.columns if 'rel_' in c] + other_columns
        self.absolute = [c for c in self.params_df.columns if 'rel_' not in c] + other_columns

    def _translate_conditions(self):
        self.params_df['rel_trans_x'] = self.params_df['trans_x'] - self.camera_pos[0]
        self.params_df['rel_trans_y'] = self.params_df['trans_y'] - self.camera_pos[1]
        self.params_df['rel_trans_z'] = self.params_df['trans_z'] - self.camera_pos[2]

        self.params_df['rel_light_x'] = self.params_df['light_x'] - self.camera_pos[0]
        self.params_df['rel_light_y'] = self.params_df['light_y'] - self.camera_pos[1]
        self.params_df['rel_light_z'] = self.params_df['light_z'] - self.camera_pos[2]
    
    def _normalize_conditions(self):
        column_ranges = {
            'trans_x': (-20.0, 20.0),
            'trans_y': (-20.0, 20.0),
            'trans_z': (-20.0, 20.0),
            'shine': (3.0, 20.0),
            'light_x': (-20.0, 20.0),
            'light_y': (-20.0, 20.0),
            'light_z': (-20.0, 20.0),
            'rel_trans_x': (-20.0 - self.camera_pos[0], 20.0 - self.camera_pos[0]),
            'rel_trans_y': (-20.0 - self.camera_pos[1], 20.0 - self.camera_pos[1]),
            'rel_trans_z': (-20.0 - self.camera_pos[2], 20.0 - self.camera_pos[2]),

            'rel_light_x': (-20.0 - self.camera_pos[0], 20.0 - self.camera_pos[0]),
            'rel_light_y': (-20.0 - self.camera_pos[1], 20.0 - self.camera_pos[1]),
            'rel_light_z': (-20.0 - self.camera_pos[2], 20.0 - self.camera_pos[2]),
        }

        for column, (min_val, max_val) in column_ranges.items():
            self.params_df[column] = (self.params_df[column] - min_val) / (max_val - min_val)
    
    def _get_conditions(self, idx, relative=False):
        if relative:
            return self.params_df.loc[idx, self.relative].values.astype(np.float32)
        return self.params_df.loc[idx, self.absolute].values.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.images[idx])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found.")

        img = Image.open(image_path).convert('RGBA')
        return self.tensor(img), self._get_conditions(idx, relative=True)


if __name__ == '__main__':
    dataset = RenderDataset('dataset')
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][1])
    