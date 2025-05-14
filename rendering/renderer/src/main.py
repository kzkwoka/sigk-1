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
            "--output_path=../dataset_ref/",
            "--frames=12"
        ],
        PhongWindow
    )
    PHONG_NEURAL = Task(
        [
            "--shaders_dir_path=./resources/shaders/phong_neural",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../dataset_n/",
            "--frames=12"
        ],
        NPhongWindow
    )


if __name__ == '__main__':
    task = TaskType.PHONG_NEURAL
    moderngl_window.run_window_config(task.window_cls, args=task.window_args)
