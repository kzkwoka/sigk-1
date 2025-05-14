import argparse
from collections import namedtuple
from enum import Enum

import moderngl_window

from phong_window import PhongWindow
from phong_neural_window import PhongWindow as NPhongWindow  # Choose the correct import

Task = namedtuple('Task', ['window_args', 'window_cls'])


class TaskType(Enum):
    @property
    def window_args(self):
        return self.value.window_args

    @property
    def window_cls(self):
        return self.value.window_cls

    PHONG = Task(
        [
            "--shaders_dir_path=./resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../dataset/",
            "--frames=10000"
        ],
        PhongWindow
    )

    PHONG_REF = Task(
        [
            "--shaders_dir_path=./resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../dataset_ref/",
            "--frames=12",
            "--ref_path=resources/params.csv"
        ],
        PhongWindow
    )
    PHONG_NEURAL = Task(
        [
            "--shaders_dir_path=./resources/shaders/phong_neural",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../dataset/",
            "--frames=10000",
            "--ckpt_path=../ckpts/twilight-butterfly-116/best-model.ckpt"
        ],
        NPhongWindow
    )

    PHONG_NEURAL_REF = Task(
        [
            "--shaders_dir_path=./resources/shaders/phong_neural",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../dataset_n/",
            "--frames=12",
            "--ckpt_path=../ckpts/twilight-butterfly-116/best-model.ckpt",
            "--ref_path=resources/params.csv"
        ],
        NPhongWindow
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Render a scene with different shaders.")
    parser.add_argument('--neural', action='store_true', default=False,
                        help='Render using neural shading.')
    parser.add_argument('--reference', action='store_true', default=False,
                        help='Generate scenes from reference file.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.neural:
        if args.reference:
            task = TaskType.PHONG_NEURAL_REF
        else:
            task = TaskType.PHONG_NEURAL
    else:
        if args.reference:
            task = TaskType.PHONG_REF
        else:
            task = TaskType.PHONG
    moderngl_window.run_window_config(task.window_cls, args=task.window_args)
