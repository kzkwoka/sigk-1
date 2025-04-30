from pathlib import Path

import os
import moderngl
from moderngl_window import WindowConfig, geometry

from utils.shader_utils import get_shaders
from utils.config import GL_VERSION, WINDOW_TITLE, WINDOW_SIZE


class BaseWindow(WindowConfig):
    window_size = WINDOW_SIZE
    aspect_ratio = 1
    gl_version = GL_VERSION
    title = WINDOW_TITLE
    resource_dir = (Path(__file__).parent.parent / 'resources' / 'models').resolve()

    def __init__(self, **kwargs):
        super(BaseWindow, self).__init__(**kwargs)
        self.output_path = self.argv.output_path
        if self.argv.output_path:
            os.makedirs(name=self.output_path, exist_ok=True)

        shaders = get_shaders(self.argv.shaders_dir_path)
        self.program = self.ctx.program(vertex_shader=shaders[self.argv.shader_name].vertex_shader,
                                        fragment_shader=shaders[self.argv.shader_name].fragment_shader)

        self.load_models()
        self.init_shaders_variables()

    def load_models(self):
        self.obj_color = None
        if self.argv.model_name:
            self.obj = self.load_scene(self.argv.model_name)
            if self.obj.materials:
                self.obj_color = self.obj.materials[0].color
            self.vao = self.obj.root_nodes[0].mesh.vao.instance(self.program)
        else:
            self.vao = geometry.quad_2d().instance(self.program)

    def init_shaders_variables(self):
        pass

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--shaders_dir_path', type=str, required=True, help='Path to the directory with shaders')
        parser.add_argument('--shader_name', type=str, required=True,
                            help='Name of the shader to look for in the shader_path directory')
        parser.add_argument('--model_name', type=str, required=False, help='Name of the model to load')
        parser.add_argument('--output_path', type=str, required=False, help='Where to save an image')

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.1, 0.2, 0.3, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
