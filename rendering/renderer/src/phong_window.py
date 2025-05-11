import os.path

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44
from scipy.stats import truncnorm

from base_window import BaseWindow


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frames = self.argv.frames
        self.frame = 0

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    @classmethod
    def add_arguments(cls, parser):
        super(PhongWindow, cls).add_arguments(parser)
        parser.add_argument('--frames', type=int, required=True, help='Number of frames to render')

    def on_render(self, time: float, frame_time: float):
        if self.frame >= self.frames:
            self.wnd.close()
            return

        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # model_translation = np.random.uniform(-20.0, 20.0, size=3)
        a_, b_ = -20 / 7, 20 / 7
        model_translation = truncnorm.rvs(a_, b_, loc=0, scale=7, size=3)
        material_diffuse = np.random.uniform(0.0, 1.0, size=3)
        material_shininess = np.random.uniform(3.0, 20.0)
        light_position = np.random.uniform(-20.0, 20.0, size=3)

        camera_position = [5.0, 5.0, 15.0]
        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_position,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

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
