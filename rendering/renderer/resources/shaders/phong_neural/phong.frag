#version 330 core

in vec3 v_position;
in vec3 v_normal;

out vec4 f_color;


uniform sampler2D nn_texture;

void main() {
    vec2 texture_coords = gl_FragCoord.xy / vec2(128.0, 128.0);
    vec3 nn_color = texture(nn_texture, texture_coords).rgb;
    f_color = vec4(nn_color, 1.0);
}
