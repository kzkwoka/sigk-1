#version 330 core

uniform mat4 model_view_projection;
uniform mat4 model_matrix;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_position;
out vec3 v_normal;

void main() {
    v_position = (model_matrix * vec4(in_position, 1.0)).xyz;
    v_normal = normalize((model_matrix * vec4(in_normal, 0.0)).xyz);
    gl_Position = model_view_projection * vec4(in_position, 1.0);
}
