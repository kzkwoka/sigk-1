import os.path

import moderngl
import numpy as np
import torch
from PIL import Image
from pyrr import Matrix44
from scipy.stats import truncnorm

from base_window import BaseWindow
from rendering.neural.models import GAN


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frames = self.argv.frames
        self.frame = 0

        self.scene_params = np.loadtxt('resources/params.csv', delimiter=',')

        self.nn_texture = None

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        # self.model_matrix = self.program["model_matrix"]
        self.nn_texture = self.program["nn_texture"]

    @classmethod
    def add_arguments(cls, parser):
        super(PhongWindow, cls).add_arguments(parser)
        parser.add_argument('--frames', type=int, required=True, help='Number of frames to render')

    def normalize_params(self, params):
        # obj xyz, light xyz, shininess, diffuse rgb
        min_vals = np.array([-25.0, -25.0, -35.0, -25.0, -25.0, -35.0, 3.0, 0.0, 0.0, 0.0])
        max_vals = np.array([15.0, 15.0, 5.0, 15.0, 15.0, 5.0, 20.0, 1.0, 1.0, 1.0])
        # [-1, 1]
        normalized_params = 2 * ((params - min_vals) / (max_vals - min_vals)) - 1
        return normalized_params

    def get_relative_params(self, params, camera_position):
        # obj xyz, light xyz, shininess, diffuse rgb
        camera_position = np.array(camera_position)
        relative_params = np.concatenate([params[:3] - camera_position, params[3:6] - camera_position, params[6:]])
        return relative_params

    def generate_texture(self, scene_params, camera_position):
        params = self.get_relative_params(scene_params, camera_position)
        params = self.normalize_params(params)

        ckpt_path = "../ckpts/crisp-rain-103/best-model.ckpt"
        model = GAN.load_from_checkpoint(ckpt_path)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(params).float().to("cuda"))[0].cpu().permute(1, 2, 0).numpy()
        out = np.flipud((out + 1) / 2)
        return out

    def on_render(self, time: float, frame_time: float):
        if self.frame >= self.frames:
            self.wnd.close()
            return

        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # model_translation = np.random.uniform(-20.0, 20.0, size=3)
        # a_, b_ = -20 / 7, 20 / 7
        # model_translation = truncnorm.rvs(a_, b_, loc=0, scale=7, size=3)
        # material_diffuse = np.random.uniform(0.0, 1.0, size=3)
        # material_shininess = np.random.uniform(3.0, 20.0)
        # light_position = np.random.uniform(-20.0, 20.0, size=3)
        # scene_params = np.concatenate(
        #     [model_translation, light_position, [material_shininess], material_diffuse]
        # )

        scene_params = self.scene_params[self.frame]
        model_translation, light_position, [material_shininess], material_diffuse = np.split(scene_params, [3, 6, 7])

        camera_position = [5.0, 5.0, 15.0]
        nn_data = self.generate_texture(scene_params, camera_position)
        nn_data = nn_data.astype('f4')
        nn_data = np.ascontiguousarray(nn_data)

        # Create a texture from the neural network output (128x128)
        self.nn_texture = self.ctx.texture((128, 128), 3, nn_data, dtype='f4')
        self.nn_texture.use(location=0)  # Bind texture to texture unit 0

        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_position,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )


        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        # self.model_matrix.write(model_matrix.astype('f4').tobytes())


        self.vao.render()
        if self.output_path:
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(self.output_path, f'image_{self.frame:04}.png'))
            params = [self.frame] + model_translation.tolist() + material_diffuse.tolist() + [
                material_shininess] + light_position.tolist()
            with open(os.path.join(self.output_path, f'params.csv'), 'a+') as f:
                f.write(','.join(map(str, params)) + '\n')
            self.frame += 1
            if self.frame % 100 == 0:
                print(f"Saved {self.frame} images.")
