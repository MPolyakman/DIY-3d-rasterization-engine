#pragma once

#include "vectors.h"
#include <cstdint>
#include <utility>
#include <vector>

struct GLFWwindow;

vec3 operator-(vec3 a , vec3 b);

vec3 operator+(vec3 a , vec3 b);

vec3 operator*(float a, vec3 b);

vec4 operator-(vec4 a , vec4 b);

vec4 operator+(vec4 a , vec4 b);

vec3 cross(vec3 a, vec3 b);

vec3 normalize(vec3 a);

float dot(vec3 a, vec3 b);

float dot(vec4 a, vec4 b);

void glfw_error_callback(int error, const char* description);

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/);

float clamp01(float x);

std::uint8_t to_u8(float x);

// Цвет радуги по оттенку hue из [0, 1)
void hue_to_rgb(float hue, float rgb[3]);

struct Matrix
{
    unsigned i, j;

    std::vector<std::vector<float>> data;

    Matrix(unsigned x, unsigned y, std::vector<std::vector<float>> mat);
    Matrix& transpose();
    Matrix inverse() const;
    Matrix multiply(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(float s) const;
};

Matrix operator*(float s, const Matrix& m);

class Polygon
{

    public:
        std::pair<float, float>p1;
        std::pair<float, float>p2;
        std::pair<float, float>p3;

        Polygon(std::pair<float, float>fp, std::pair<float, float>sp, std::pair<float, float>tp);

        bool inside(std::pair<float, float> pixel) const;
};

struct Polygon3D {
    vec3 p1, p2, p3;
    float b1, b2, b3;
    vec2 t1, t2, t3;
    float inv_w1, inv_w2, inv_w3;
    vec3 n1, n2, n3;
    Polygon3D(vec3 v1, vec3 v2, vec3 v3, float br1, float br2, float br3, vec2 tc1, vec2 tc2, vec2 tc3, float iw1, float iw2, float iw3, vec3 no1, vec3 no2, vec3 no3);
};

struct Mat4 {
    float m[4][4];

    static Mat4 identity();

    Mat4 operator*(const Mat4& b) const;

    vec4 operator*(const vec4& v) const;

    static Mat4 lookAt(vec3 eye, vec3 target, vec3 up);

    static Mat4 perspective(float fovy, float aspect, float n, float f);
};

struct Object3D
{
    unsigned mesh_id;
    unsigned tex_id;
    vec3 where;

    Object3D(unsigned m, unsigned t, vec3 v);
};

struct MipLevel {
    int width;
    int height;
    std::vector<float> pixels;
};
