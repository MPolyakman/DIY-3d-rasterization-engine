#include "geomath.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

vec3 operator-(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 operator+(vec3 a, vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 operator*(float a, vec3 b) {
    return {a * b.x, a * b.y, a * b.z};
}

vec4 operator-(vec4 a, vec4 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

vec4 operator+(vec4 a, vec4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

vec3 cross(vec3 a, vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

vec3 normalize(vec3 a) {
    float length = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    a.x /= length;
    a.y /= length;
    a.z /= length;
    return a;
}

float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float dot(vec4 a, vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void key_callback(GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

float clamp01(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

std::uint8_t to_u8(float x) {
    return static_cast<std::uint8_t>(clamp01(x) * 255.0f + 0.5f);
}

void hue_to_rgb(float hue, float rgb[3]) {
    const float h = hue * 6.0f;
    rgb[0] = clamp01(std::fabs(h - 3.0f) - 1.0f);
    rgb[1] = clamp01(2.0f - std::fabs(h - 2.0f));
    rgb[2] = clamp01(2.0f - std::fabs(h - 4.0f));
}

Matrix::Matrix(unsigned x, unsigned y, std::vector<std::vector<float>> mat) : i(x), j(y) {
    if (i > mat.size())
        throw std::invalid_argument("i <= size of row");

    for (const auto& row : mat) {
        if (j > row.size())
            throw std::invalid_argument("j <= size of column");
    }
    data = std::move(mat);
}

Matrix& Matrix::transpose() {
    for (unsigned y = 0; y < j; ++y) {
        for (unsigned x = y; x < i; ++x)
            std::swap(data[x][y], data[y][x]);
    }
    return *this;
}

Matrix Matrix::inverse() const {
    if (i != j)
        throw std::invalid_argument("cannot invert a non-square matrix");
    const unsigned n = i;

    std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0.0));
    for (unsigned row = 0; row < n; ++row) {
        for (unsigned col = 0; col < n; ++col)
            augmented[row][col] = static_cast<double>(data[row][col]);
        augmented[row][n + row] = 1.0;
    }

    for (unsigned col = 0; col < n; ++col) {
        unsigned pivot = col;
        double max_value = std::abs(augmented[col][col]);
        for (unsigned row = col + 1; row < n; ++row) {
            if (std::abs(augmented[row][col]) > max_value) {
                max_value = std::abs(augmented[row][col]);
                pivot = row;
            }
        }
        if (max_value < 1e-12)
            throw std::runtime_error("matrix is singular and cannot be inverted");
        if (pivot != col)
            std::swap(augmented[pivot], augmented[col]);

        const double pivot_value = augmented[col][col];
        for (unsigned c = 0; c < 2 * n; ++c)
            augmented[col][c] /= pivot_value;

        for (unsigned row = 0; row < n; ++row) {
            if (row == col)
                continue;
            const double factor = augmented[row][col];
            if (factor == 0.0)
                continue;
            for (unsigned c = 0; c < 2 * n; ++c)
                augmented[row][c] -= factor * augmented[col][c];
        }
    }

    std::vector<std::vector<float>> inverse_data(n, std::vector<float>(n));
    for (unsigned row = 0; row < n; ++row) {
        for (unsigned col = 0; col < n; ++col)
            inverse_data[row][col] = static_cast<float>(augmented[row][n + col]);
    }
    return Matrix(n, n, std::move(inverse_data));
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (j != other.i) {
        throw std::invalid_argument(
            "dimension mismatch: A is " + std::to_string(i) + "x" + std::to_string(j) +
            ", B is " + std::to_string(other.i) + "x" + std::to_string(other.j));
    }

    std::vector<std::vector<float>> result(i, std::vector<float>(other.j, 0.0f));
    for (unsigned row = 0; row < i; ++row) {
        for (unsigned k = 0; k < j; ++k) {
            const double value = static_cast<double>(data[row][k]);
            if (value == 0.0)
                continue;
            for (unsigned col = 0; col < other.j; ++col)
                result[row][col] += static_cast<float>(value * other.data[k][col]);
        }
    }
    return Matrix(i, other.j, std::move(result));
}

Matrix Matrix::operator*(const Matrix& other) const {
    return multiply(other);
}

Matrix Matrix::operator*(float s) const {
    auto result = data;
    for (auto& row : result) {
        for (auto& value : row)
            value *= s;
    }
    return Matrix(i, j, std::move(result));
}

Matrix operator*(float s, const Matrix& m) {
    return m * s;
}

Polygon::Polygon(std::pair<float, float> fp, std::pair<float, float> sp, std::pair<float, float> tp)
    : p1(fp), p2(sp), p3(tp) {}

bool Polygon::inside(std::pair<float, float> pixel) const {
    const float px = pixel.first;
    const float py = pixel.second;
    const float v1x = p2.first - p1.first;
    const float v1y = p2.second - p1.second;
    const float v2x = p3.first - p2.first;
    const float v2y = p3.second - p2.second;
    const float v3x = p1.first - p3.first;
    const float v3y = p1.second - p3.second;
    const float c1 = v1x * (py - p1.second) - v1y * (px - p1.first);
    const float c2 = v2x * (py - p2.second) - v2y * (px - p2.first);
    const float c3 = v3x * (py - p3.second) - v3y * (px - p3.first);
    return (c1 >= 0 && c2 >= 0 && c3 >= 0) || (c1 <= 0 && c2 <= 0 && c3 <= 0);
}

Polygon3D::Polygon3D(vec3 v1, vec3 v2, vec3 v3, float br1, float br2, float br3,
                     vec2 tc1, vec2 tc2, vec2 tc3, float iw1, float iw2, float iw3,
                     vec3 no1, vec3 no2, vec3 no3)
    : p1(v1), p2(v2), p3(v3), b1(br1), b2(br2), b3(br3), t1(tc1), t2(tc2), t3(tc3),
      inv_w1(iw1), inv_w2(iw2), inv_w3(iw3), n1(no1), n2(no2), n3(no3) {}

Mat4 Mat4::identity() {
    Mat4 result{};
    for (int i = 0; i < 4; ++i)
        result.m[i][i] = 1.0f;
    return result;
}

Mat4 Mat4::operator*(const Mat4& b) const {
    Mat4 result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            for (int k = 0; k < 4; ++k)
                result.m[row][col] += m[row][k] * b.m[k][col];
        }
    }
    return result;
}

vec4 Mat4::operator*(const vec4& v) const {
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
    };
}

Mat4 Mat4::lookAt(vec3 eye, vec3 target, vec3 up) {
    const vec3 forward = normalize(target - eye);
    const vec3 side = normalize(cross(forward, up));
    const vec3 corrected_up = cross(side, forward);
    Mat4 result = identity();
    result.m[0][0] = side.x; result.m[0][1] = side.y; result.m[0][2] = side.z; result.m[0][3] = -dot(side, eye);
    result.m[1][0] = corrected_up.x; result.m[1][1] = corrected_up.y; result.m[1][2] = corrected_up.z; result.m[1][3] = -dot(corrected_up, eye);
    result.m[2][0] = -forward.x; result.m[2][1] = -forward.y; result.m[2][2] = -forward.z; result.m[2][3] = dot(forward, eye);
    return result;
}

Mat4 Mat4::perspective(float fovy, float aspect, float n, float f) {
    const float t = 1.0f / std::tan(fovy * 0.5f);
    Mat4 result{};
    result.m[0][0] = t / aspect;
    result.m[1][1] = t;
    result.m[2][2] = (f + n) / (n - f);
    result.m[2][3] = (2.0f * f * n) / (n - f);
    result.m[3][2] = -1.0f;
    return result;
}

Object3D::Object3D(unsigned m, unsigned t, vec3 v) : mesh_id(m), tex_id(t), where(v) {}
