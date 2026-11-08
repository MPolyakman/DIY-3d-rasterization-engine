#include <GLFW/glfw3.h>

#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <random>

#include "image_utils.h"
#include "mesh_utils.h"
#include "vectors.h"

constexpr float FARFARAWAY = std::numeric_limits<float>::max();

vec3 operator-(vec3 a , vec3 b) {
    vec3 out;
    out.x = a.x - b.x;
    out.y = a.y - b.y;
    out.z = a.z - b.z;
    return out;
}

vec3 operator+(vec3 a , vec3 b) {
    vec3 out;
    out.x = a.x + b.x;
    out.y = a.y + b.y;
    out.z = a.z + b.z;
    return out;
}

vec3 operator*(float a, vec3 b) {
    b.x *= a;
    b.y *= a;
    b.z *= a;
    return b;
}

vec4 operator-(vec4 a , vec4 b) {
    vec4 out;
    out.x = a.x - b.x;
    out.y = a.y - b.y;
    out.z = a.z - b.z;
    out.w = a.w - b.w;
    return out;
}

vec4 operator+(vec4 a , vec4 b) {
    vec4 out;
    out.x = a.x + b.x;
    out.y = a.y + b.y;
    out.z = a.z + b.z;
    out.w = a.w + b.w;
    return out;
}

vec3 cross(vec3 a, vec3 b) {
    vec3 out;
    out.x = a.y * b.z - a.z * b.y;
    out.y = a.z * b.x - a.x * b.z;
    out.z = a.x * b.y - a.y * b.x;
    return out;
}

vec3 normalize(vec3 a) {
    float l = std::sqrt(a.x *a.x + a.y * a.y + a.z * a.z);
    a.x /= l;
    a.y /= l;
    a.z /= l;
    return a;
}

float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float dot(vec4 a, vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

namespace {

void glfw_error_callback(int error, const char* description)
{
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

constexpr float clamp01(float x)
{
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

constexpr std::uint8_t to_u8(float x)
{
    return static_cast<std::uint8_t>(clamp01(x) * 255.0f + 0.5f);
}

// Цвет радуги по оттенку hue из [0, 1)
void hue_to_rgb(float hue, float rgb[3])
{
    const float h = hue * 6.0f;
    rgb[0] = clamp01(std::fabs(h - 3.0f) - 1.0f);
    rgb[1] = clamp01(2.0f - std::fabs(h - 2.0f));
    rgb[2] = clamp01(2.0f - std::fabs(h - 4.0f));
}

} // namespace

struct Matrix
{
    unsigned i, j;

    std::vector<std::vector<float>> data;

    Matrix(unsigned x, unsigned y, std::vector<std::vector<float>> mat) : i(x), j(y) {
        if (i > mat.size()) {
            throw std::invalid_argument("i <= size of row");
        }

        for (auto col  : mat) {
            if (j > col.size()) {
                throw std::invalid_argument("j <= sixe of collumn");
            }
        }
        this->data = mat;
    }

    Matrix& transpose() {
        for (int y = 0; y < j; y++) {
            for (int x = y; x < i; x++) {
                std::swap(data[x][y], data[y][x]);
            }
        }

        return *this;
    }

    Matrix inverse() const {
        if (i != j) {
            throw std::invalid_argument("cannot invert a non-square matrix");
        }
        const unsigned n = i;

        std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
        for (unsigned r = 0; r < n; ++r) {
            for (unsigned c = 0; c < n; ++c) {
                aug[r][c] = static_cast<double>(data[r][c]);
            }
            aug[r][n + r] = 1.0;
        }

        for (unsigned col = 0; col < n; ++col) {
            // partial pivot
            unsigned pivot = col;
            double maxVal = std::abs(aug[col][col]);
            for (unsigned r = col + 1; r < n; ++r) {
                if (std::abs(aug[r][col]) > maxVal) {
                    maxVal = std::abs(aug[r][col]);
                    pivot = r;
                }
            }
            if (maxVal < 1e-12) {
                throw std::runtime_error("matrix is singular and cannot be inverted");
            }
            if (pivot != col) {
                std::swap(aug[pivot], aug[col]);
            }

            double pv = aug[col][col];
            for (unsigned c = 0; c < 2 * n; ++c) {
                aug[col][c] /= pv;
            }

            for (unsigned r = 0; r < n; ++r) {
                if (r == col) continue;
                double f = aug[r][col];
                if (f == 0.0) continue;
                for (unsigned c = 0; c < 2 * n; ++c) {
                    aug[r][c] -= f * aug[col][c];
                }
            }
        }

        std::vector<std::vector<float>> inv(n, std::vector<float>(n));
        for (unsigned r = 0; r < n; ++r) {
            for (unsigned c = 0; c < n; ++c) {
                inv[r][c] = static_cast<float>(aug[r][n + c]);
            }
        }
        return Matrix(n, n, inv);

    }

    Matrix multiply(const Matrix& other) const {
        if (j != other.i) {
            throw std::invalid_argument(
                "dimension mismatch: A is " + std::to_string(i) + "x" + std::to_string(j) +
                ", B is " + std::to_string(other.i) + "x" + std::to_string(other.j));
        }

        std::vector<std::vector<float>> result(i, std::vector<float>(other.j, 0.0f));

        for (unsigned r = 0; r < i; ++r) {
            for (unsigned k = 0; k < j; ++k) {
                double a = static_cast<double>(data[r][k]);
                if (a == 0.0) continue;
                for (unsigned c = 0; c < other.j; ++c) {
                    result[r][c] += static_cast<float>(a * other.data[k][c]);
                }
            }
        }

        return Matrix(i, other.j, result);
    }

    Matrix operator*(const Matrix& other) const {
        return multiply(other);
    }

    Matrix operator*(float s) const {
        auto result = data;
        for (auto& row : result)
            for (auto& v : row)
                v *= s;
        return Matrix(i, j, result);
    }

    friend Matrix operator*(float s, const Matrix& m) {
        return m * s;
    }
};

class Polygon
{
    public:
        std::pair<float, float>p1;
        std::pair<float, float>p2;
        std::pair<float, float>p3;

        Polygon(std::pair<float, float>fp, std::pair<float, float>sp, std::pair<float, float>tp) : p1(fp), p2(sp), p3(tp) {}

        bool inside(std::pair<float, float> pixel) const
        {
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

            return (c1 >= 0 && c2 >= 0 && c3 >= 0)
                || (c1 <= 0 && c2 <= 0 && c3 <= 0);
        }
};

struct Polygon3D {
    vec3 p1, p2, p3;
    float b1, b2, b3;
    vec2 t1, t2, t3;
    float inv_w1, inv_w2, inv_w3;
    vec3 n1, n2, n3;
    Polygon3D(vec3 v1, vec3 v2, vec3 v3, float br1, float br2, float br3, vec2 tc1, vec2 tc2, vec2 tc3, float iw1, float iw2, float iw3, vec3 no1, vec3 no2, vec3 no3)
    : p1(v1), p2(v2), p3(v3), b1(br1), b2(br2), b3(br3), t1(tc1), t2(tc2), t3(tc3), inv_w1(iw1), inv_w2(iw2), inv_w3(iw3), n1(no1), n2(no2), n3(no3) {}
};

struct Mat4 {
    float m[4][4];

    static Mat4 identity() {
        Mat4 res{};
        for(int i=0; i<4; ++i) res.m[i][i] = 1.0f;
        return res;
    }

    Mat4 operator*(const Mat4& b) const {
        Mat4 res{};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                res.m[r][c] = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    res.m[r][c] += m[r][k] * b.m[k][c];
                }
            }
        }
        return res;
    }

    vec4 operator*(const vec4& v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
            m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w
        };
    }

    static Mat4 lookAt(vec3 eye, vec3 target, vec3 up) {
        vec3 f = normalize(target - eye);
        vec3 s = normalize(cross(f, up));
        vec3 u = cross(s, f);
        Mat4 res = identity();

        res.m[0][0] = s.x; res.m[0][1] = s.y; res.m[0][2] = s.z; res.m[0][3] = -dot(s, eye);
        res.m[1][0] = u.x; res.m[1][1] = u.y; res.m[1][2] = u.z; res.m[1][3] = -dot(u, eye);
        res.m[2][0] = -f.x; res.m[2][1] = -f.y; res.m[2][2] = -f.z; res.m[2][3] = dot(f, eye);
        return res;
    }

    static Mat4 perspective(float fovy, float aspect, float n, float f) {
        float t = 1.0f / std::tan(fovy * 0.5f);
        Mat4 res{};
        res.m[0][0] = t / aspect;
        res.m[1][1] = t;
        res.m[2][2] = (f + n) / (n - f);
        res.m[2][3] = (2.0f * f * n) / (n - f);
        res.m[3][2] = -1.0f;
        return res;
    }
};

struct Object3D
{
    unsigned mesh_id;
    unsigned tex_id;
    vec3 where;

    Object3D(unsigned m, unsigned t, vec3 v) : mesh_id(m), tex_id(t), where(v) {}
};

class Render
{
    public:
        int task;
        int height;
        int width;
        std::vector<std::uint8_t>pixels;
        std::vector<float>z_buffer;

        vec3 light_dir;

        std::vector<mesh>meshes;
        std::vector<float*>original_textures;
        std::vector<std::pair<int, int>>img_sizes;
        std::vector<Object3D>objects;

        Render(int todo, int w, int h,  std::vector<std::string>meshes_paths, std::vector<std::string>texture_paths, vec3 l = {1, 0, 0}) : task(todo), width(w), height(h),
            pixels(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4),  z_buffer(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), FARFARAWAY) {
                light_dir = (1 / std::sqrt(l.x * l.x + l.y * l.y + l.z * l.z)) * l;
                std::unordered_map<std::string, int>mesh_mapa;
                std::unordered_map<std::string, int>tex_mapa;
                unsigned obj_counter = 0;
                unsigned tex_counter = 0;
                unsigned obj_mesh_id = 0;
                unsigned obj_tex_id = 0;

                meshes.resize(meshes_paths.size());
                original_textures.resize(meshes_paths.size(), nullptr);
                img_sizes.resize(meshes_paths.size());
                // objects.resize(meshes_paths.size());

                for (int i = 0; i < (int)meshes_paths.size(); ++i) {
                    meshes[i] = load_obj(meshes_paths[i].c_str());

                    int wi = 0, hi = 0;
                    original_textures[i] = load_image_f32_rgb(texture_paths[i].c_str(), &wi, &hi, 1.0f);
                    img_sizes[i] = { wi, hi };

                    if (!mesh_mapa.contains(meshes_paths[i]))
                        mesh_mapa[meshes_paths[i]] = i;
                    unsigned obj_mesh_id = (unsigned)mesh_mapa[meshes_paths[i]];

                    if (!tex_mapa.contains(texture_paths[i]))
                        tex_mapa[texture_paths[i]] = i;
                    unsigned obj_tex_id = (unsigned)tex_mapa[texture_paths[i]];

                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> dist(-0.9f, 0.9f);
                    vec3 place{ dist(gen), dist(gen), dist(gen) - 0.5f };

                    objects.push_back(Object3D(obj_mesh_id, obj_tex_id, place));
                }

                for (auto& m : meshes) {
                    mesh welded = weld_and_compute_normals(m);
                    if (welded.num_vertices > 0 && welded.num_triangles > 0) {
                        free_mesh(&m);
                        m = welded;
                    } else {
                        free_mesh(&welded);
                    }
                    make_normals(m);
                    mesh scaled = scale_mesh(m);
                    free_mesh(&m);
                    m = scaled;
                }
            }

        ~Render() {
            for (float* t : original_textures) std::free(t);
            for (auto& m : meshes) free_mesh(&m);
        }

        std::vector<std::uint8_t> render_default() {
            int fb_width = 0, fb_height = 0;
            const std::size_t n_pixels = static_cast<std::size_t>(fb_width) * static_cast<std::size_t>(fb_height);
            std::vector<std::uint8_t> pixels(4 * n_pixels);
            for (std::size_t i = 0; i < n_pixels; ++i)
            {
                pixels[4 * i + 0] = 200;
                pixels[4 * i + 1] = 200;
                pixels[4 * i + 2] = 200;
                pixels[4 * i + 3] = 255;
            }
            return pixels;
        }

        mesh weld_and_compute_normals(const mesh& src, float eps = 1e-5f) {
            mesh dst{};
            dst.num_vertices  = 0;
            dst.num_triangles = src.num_triangles;
            dst.verts = dst.normals = nullptr;
            dst.tcs = nullptr;
            dst.indices = nullptr;

            if (src.num_triangles <= 0 || !src.indices || !src.verts) return dst;

            struct Key { int x, y, z, u, v; };
            auto key_of = [&](int old_idx) {
                const vec3& p = src.verts[old_idx];
                float u = 0.f, v = 0.f;
                if (src.tcs) { u = src.tcs[old_idx].x; v = src.tcs[old_idx].y; }
                return Key{
                    (int)std::lround(p.x / eps),
                    (int)std::lround(p.y / eps),
                    (int)std::lround(p.z / eps),
                    (int)std::lround(u / eps),
                    (int)std::lround(v / eps)
                };
            };
            struct KH { size_t operator()(const Key& k) const noexcept {
                return ((size_t)(uint32_t)k.x * 73856093u)
                     ^ ((size_t)(uint32_t)k.y * 19349663u)
                     ^ ((size_t)(uint32_t)k.z * 83492791u)
                     ^ ((size_t)(uint32_t)k.u * 2654435761u)
                     ^ ((size_t)(uint32_t)k.v * 40503u);
            }};
            struct KE { bool operator()(const Key& a, const Key& b) const noexcept {
                return a.x==b.x && a.y==b.y && a.z==b.z && a.u==b.u && a.v==b.v;
            }};

            std::unordered_map<Key, int, KH, KE> remap;
            remap.reserve(src.num_triangles * 3);

            std::vector<vec3> new_verts;
            std::vector<vec2> new_tcs;
            std::vector<int>  new_indices(src.num_triangles * 3);
            new_verts.reserve(src.num_triangles * 3);
            if (src.tcs) new_tcs.reserve(src.num_triangles * 3);

            for (int i = 0; i < src.num_triangles * 3; ++i) {
                int old = src.indices[i];
                Key k = key_of(old);
                auto it = remap.find(k);
                int ni;
                if (it == remap.end()) {
                    ni = (int)new_verts.size();
                    remap.emplace(k, ni);
                    new_verts.push_back(src.verts[old]);
                    if (src.tcs) new_tcs.push_back(src.tcs[old]);
                } else {
                    ni = it->second;
                }
                new_indices[i] = ni;
            }

            dst.num_vertices = (int)new_verts.size();
            dst.verts   = (vec3*)std::malloc(new_verts.size()   * sizeof(vec3));
            dst.indices = (int* )std::malloc(new_indices.size() * sizeof(int));
            if (!dst.verts || !dst.indices) {
                std::free(dst.verts);
                std::free(dst.indices);
                return mesh{};
            }
            std::memcpy(dst.verts,   new_verts.data(),   new_verts.size()   * sizeof(vec3));
            std::memcpy(dst.indices, new_indices.data(), new_indices.size() * sizeof(int));

            if (src.tcs) {
                dst.tcs = (vec2*)std::malloc(new_tcs.size() * sizeof(vec2));
                if (!dst.tcs) {
                    std::free(dst.verts);
                    std::free(dst.indices);
                    return mesh{};
                }
                std::memcpy(dst.tcs, new_tcs.data(), new_tcs.size() * sizeof(vec2));
            }

            return dst;
        }

        std::vector<float> cross_product(std::pair<float, float> pixel, Polygon trig) {
            std::pair<float,  float> p1 = trig.p1;
            std::pair<float,  float> p2 = trig.p2;
            std::pair<float,  float> p3 = trig.p3;


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

            return std::vector<float> {c1, c2, c3};
        }

        std::vector<std::uint8_t> render_triangle(Polygon poly) {
            std::pair<float, float> p1 = poly.p1;
            std::pair<float, float> p2 = poly.p2;
            std::pair<float, float> p3 = poly.p3;

            std::vector<std::uint8_t> pixels(width * height * 4);
            float max_x = -1;
            float min_x = INFINITY;
            float max_y = -1;
            float min_y = INFINITY;

            max_x = std::max(max_x, p1.first);
            max_x = std::max(max_x, p2.first);
            max_x = std::max(max_x, p3.first);
            min_x = std::min(min_x, p1.first);
            min_x = std::min(min_x, p2.first);
            min_x = std::min(min_x, p3.first);

            max_y = std::max(max_y, p1.second);
            max_y = std::max(max_y, p2.second);
            max_y = std::max(max_y, p3.second);
            min_y = std::min(min_y, p1.second);
            min_y = std::min(min_y, p2.second);
            min_y = std::min(min_y, p3.second);

            min_x = std::max(0.0f, min_x);
            min_y = std::max(0.0f, min_y);
            max_x = std::min(static_cast<float>(width), max_x);
            max_y = std::min(static_cast<float>(height), max_y);

            const float v1x = p2.first - p1.first;
            const float v1y = p2.second - p1.second;
            const float v2x = p3.first - p2.first;
            const float v2y = p3.second - p2.second;
            const float v3x = p1.first - p3.first;
            const float v3y = p1.second - p3.second;

            const float area2 = std::fabs((poly.p2.first  - poly.p1.first) * (poly.p3.second - poly.p1.second) - (poly.p2.second - poly.p1.second) * (poly.p3.first  - poly.p1.first));
            if (area2 == 0.0f) return pixels;


            for (int x = min_x; x <= max_x; x++) {
                for (int y = min_y; y <= max_y; y++) {
                    std::pair<float, float> pix(x + 0.5, y + 0.5);

                    float px = pix.first;
                    float py = pix.second;

                    float c1 = v1x * (py - p1.second) - v1y * (px - p1.first);
                    float c2 = v2x * (py - p2.second) - v2y * (px - p2.first);
                    float c3 = v3x * (py - p3.second) - v3y * (px - p3.first);

                    bool in_trig = (c1 < 0 && c2 < 0 && c3 < 0) || (c1 > 0 && c2 > 0 && c3 > 0);

                    if (in_trig) {
                        pixels[(x + y*width) * 4 ]     = static_cast<std::uint8_t>(std::fabs(c1) * 255.0f / area2);
                        pixels[(x + y*width) * 4 + 1]  = static_cast<uint8_t>(std::abs(c2) * 255.0f / area2);
                        pixels[(x + y*width) * 4 + 2]  = static_cast<uint8_t>(std::abs(c3) * 255.0f / area2);
                        pixels[(x + y*width) * 4 + 3 ] = 255;
                    }
                }
            }
            return pixels;
        }

    mesh scale_mesh(const mesh &src)
    {
        mesh dst{};
        dst.num_vertices  = 0;
        dst.num_triangles = 0;
        dst.verts   = nullptr;
        dst.normals = nullptr;
        dst.tcs     = nullptr;
        dst.indices = nullptr;

        if (src.num_vertices <= 0 || src.verts == nullptr) {
            return dst;
        }

        float minx =  std::numeric_limits<float>::infinity();
        float miny =  std::numeric_limits<float>::infinity();
        float minz =  std::numeric_limits<float>::infinity();
        float maxx = -std::numeric_limits<float>::infinity();
        float maxy = -std::numeric_limits<float>::infinity();
        float maxz = -std::numeric_limits<float>::infinity();

        for (int k = 0; k < src.num_vertices; ++k) {
            const vec3 &v = src.verts[k];
            minx = std::min(minx, v.x);  maxx = std::max(maxx, v.x);
            miny = std::min(miny, v.y);  maxy = std::max(maxy, v.y);
            minz = std::min(minz, v.z);  maxz = std::max(maxz, v.z);
        }

        const float cx = 0.5f * (minx + maxx);
        const float cy = 0.5f * (miny + maxy);
        const float cz = 0.5f * (minz + maxz);

        const float ex = maxx - minx;
        const float ey = maxy - miny;
        const float ez = maxz - minz;
        const float maxExtent = std::max(ex, std::max(ey, ez));

        const float s = (maxExtent > 0.0f) ? (1.0f / maxExtent) : 1.0f;

        const size_t nv = static_cast<size_t>(src.num_vertices);
        const size_t nt = static_cast<size_t>(src.num_triangles);

        dst.verts = static_cast<vec3*>(std::malloc(nv * sizeof(vec3)));
        if (!dst.verts) return mesh{};

        if (src.normals) {
            dst.normals = static_cast<vec3*>(std::malloc(nv * sizeof(vec3)));
            if (!dst.normals) { std::free(dst.verts); return mesh{}; }
        }
        if (src.tcs) {
            dst.tcs = static_cast<vec2*>(std::malloc(nv * sizeof(vec2)));
            if (!dst.tcs) { std::free(dst.verts); std::free(dst.normals); return mesh{}; }
        }
        if (nt > 0) {
            dst.indices = static_cast<int*>(std::malloc(nt * 3 * sizeof(int)));
            if (!dst.indices) {
                std::free(dst.verts); std::free(dst.normals); std::free(dst.tcs);
                return mesh{};
            }
        }

        dst.num_vertices  = src.num_vertices;
        dst.num_triangles = src.num_triangles;

        for (int k = 0; k < src.num_vertices; ++k) {
            const vec3 &v = src.verts[k];
            dst.verts[k] = vec3{(v.x - cx) * s, (v.y - cy) * s, (v.z - cz) * s };
        }

        if (src.normals) {
            std::memcpy(dst.normals, src.normals, nv * sizeof(vec3));
        }

        if (src.tcs) {
            std::memcpy(dst.tcs, src.tcs, nv * sizeof(vec2));
        }

        if (nt > 0 && src.indices) {
            std::memcpy(dst.indices, src.indices, nt * 3 * sizeof(int));
        }

        return dst;
    }

    mesh place(const mesh& instance, vec3 shift) {
        mesh out{};
        out.num_vertices  = instance.num_vertices;
        out.num_triangles = instance.num_triangles;

        out.verts   = (vec3*)std::malloc(instance.num_vertices * sizeof(vec3));
        out.normals = instance.normals ? (vec3*)std::malloc(instance.num_vertices * sizeof(vec3)) : nullptr;
        out.tcs     = instance.tcs     ? (vec2*)std::malloc(instance.num_vertices * sizeof(vec2)) : nullptr;
        out.indices = instance.indices ? (int* )std::malloc(instance.num_triangles * 3 * sizeof(int)) : nullptr;

        for (int i = 0; i < instance.num_vertices; ++i) {
            out.verts[i] = instance.verts[i] + shift;
            if (out.normals) out.normals[i] = instance.normals[i];
            if (out.tcs)     out.tcs[i]     = instance.tcs[i];
        }
        if (out.indices && instance.indices) {
            std::memcpy(out.indices, instance.indices,
                        instance.num_triangles * 3 * sizeof(int));
        }
        return out;
    }

    void render_triangle_3d(const Polygon3D& poly, float* img, int w, int h) {
        const vec3& v1 = poly.p1;
        const vec3& v2 = poly.p2;
        const vec3& v3 = poly.p3;

        int min_x = std::max(0, (int)std::floor(std::min({v1.x, v2.x, v3.x})));
        int max_x = std::min(width - 1, (int)std::ceil(std::max({v1.x, v2.x, v3.x})));
        int min_y = std::max(0, (int)std::floor(std::min({v1.y, v2.y, v3.y})));
        int max_y = std::min(height - 1, (int)std::ceil(std::max({v1.y, v2.y, v3.y})));

        float area2 = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);

        if (area2 <= 0.0001f) return;

        float inv_area2 = 1.0f / area2;


        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                float px = x + 0.5f;
                float py = y + 0.5f;

                float b1 = ((v2.x - px) * (v3.y - py) - (v2.y - py) * (v3.x - px)) * inv_area2;
                float b2 = ((v3.x - px) * (v1.y - py) - (v3.y - py) * (v1.x - px)) * inv_area2;
                float b3 = 1.0f - b1 - b2;

                if (b1 >= 0.0f && b2 >= 0.0f && b3 >= 0.0f) {

                    float z_ndc = b1 * v1.z + b2 * v2.z + b3 * v3.z;

                    int idx = y * width + x;

                    if (z_ndc < z_buffer[idx]) {
                        z_buffer[idx] = z_ndc;


                        if (task == 3) {
                            pixels[idx * 4 + 0] = 200; // R
                            pixels[idx * 4 + 1] = 200; // G
                            pixels[idx * 4 + 2] = 200; // B
                            pixels[idx * 4 + 3] = 255; // A
                        }
                        else if (task == 4) {
                            // float br = b1*poly.b1 + b2*poly.b2 + b3*poly.b3;
                            // br = std::max(br, 0.07f);

                            vec3 n = normalize(b1*poly.n1 + b2*poly.n2 + b3*poly.n3);
                            float br = std::max(dot(n, (-1)*light_dir), 0.0f) + 0.07f;

                            std::uint8_t value = static_cast<std::uint8_t>(std::clamp(br * 255.0f, 0.0f, 255.0f));
                            pixels[idx * 4 + 0] = value; // R
                            pixels[idx * 4 + 1] = value; // G
                            pixels[idx * 4 + 2] = value; // B
                            pixels[idx * 4 + 3] = 255; // A
                        }
                        else if (task == 99) {
                            vec3 n = normalize(b1*poly.n1 + b2*poly.n2 + b3*poly.n3);
                            pixels[idx*4+0] = (std::uint8_t)std::clamp((n.x*0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
                            pixels[idx*4+1] = (std::uint8_t)std::clamp((n.y*0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
                            pixels[idx*4+2] = (std::uint8_t)std::clamp((n.z*0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
                            pixels[idx*4+3] = 255;
                        }

                        else if (task == 5 || task == 6) {
                            vec3 n = normalize(b1*poly.n1 + b2*poly.n2 + b3*poly.n3);
                            float br = std::max(dot(n, (-1)*light_dir), 0.0f) + 0.07f;

                            if (img == nullptr || w <= 0 || h <= 0) {
                                std::uint8_t v = static_cast<std::uint8_t>(br * 255.0f);
                                pixels[idx*4+0] = pixels[idx*4+1] = pixels[idx*4+2] = v;
                                pixels[idx*4+3] = 255;
                            } else {

                                float inv_w_p = b1*poly.inv_w1 + b2*poly.inv_w2 + b3*poly.inv_w3;

                                float u = ( b1*poly.t1.x*poly.inv_w1
                                          + b2*poly.t2.x*poly.inv_w2
                                          + b3*poly.t3.x*poly.inv_w3 ) / inv_w_p;

                                float v = ( b1*poly.t1.y*poly.inv_w1
                                          + b2*poly.t2.y*poly.inv_w2
                                          + b3*poly.t3.y*poly.inv_w3 ) / inv_w_p;

                                float ti = u*w - 0.5f, tj = v*h - 0.5f;
                                int tx = (int)std::floor(ti), ty = (int)std::floor(tj);
                                float alpha = ti - tx, beta = tj - ty;
                                int x0 = std::clamp(tx, 0, w-1), x1 = std::clamp(tx+1, 0, w-1);
                                int y0 = std::clamp(ty, 0, h-1), y1 = std::clamp(ty+1, 0, h-1);
                                auto samp = [&](int c) {
                                    return (1-alpha)*(1-beta)*img[(y0*w + x0)*3 + c]
                                         +      alpha *(1-beta)*img[(y0*w + x1)*3 + c]
                                         + (1-alpha)*     beta *img[(y1*w + x0)*3 + c]
                                         +      alpha *     beta *img[(y1*w + x1)*3 + c];
                                };
                                pixels[idx*4+0] = (std::uint8_t)std::clamp(samp(0)*255.0f * br, 0.0f, 255.0f);
                                pixels[idx*4+1] = (std::uint8_t)std::clamp(samp(1)*255.0f * br, 0.0f, 255.0f);
                                pixels[idx*4+2] = (std::uint8_t)std::clamp(samp(2)*255.0f * br, 0.0f, 255.0f);
                                pixels[idx*4+3] = 255;
                            }

                        }
                    }
                }
            }
        }
    }

    void visualize_depth() {
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < width * height; ++i) {
            if (z_buffer[i] < FARFARAWAY) {
                min_z = std::min(min_z, z_buffer[i]);
                max_z = std::max(max_z, z_buffer[i]);
            }
        }

        float range = (max_z > min_z) ? (max_z - min_z) : 1.0f;

        for (int i = 0; i < width * height; ++i) {
            if (z_buffer[i] >= FARFARAWAY) {
                pixels[i * 4 + 0] = 0;
                pixels[i * 4 + 1] = 0;
                pixels[i * 4 + 2] = 0;
            } else {
                float t = (z_buffer[i] - min_z) / range;
                std::uint8_t c = static_cast<std::uint8_t>((1.0f - t) * 255.0f);
                pixels[i * 4 + 0] = c;
                pixels[i * 4 + 1] = c;
                pixels[i * 4 + 2] = c;
            }
            pixels[i * 4 + 3] = 255;
        }
    }

    // void shade_lambert(mesh obj) {
    //     unsigned s = sizeof(*obj.normals) / sizeof(obj.normals[0]);
    //     std::vector<float>bright(s);
    //     for (int i = 0; i < s; i++) {
    //         bright[i] = dot(obj.normals[i], (-1 * std::sqrt(light_dir.x * light_dir.x + light_dir.y * light_dir.y + light_dir.z * light_dir.z) * light_dir));
    //     }
    // }

    void make_normals(mesh& m) {
        if (m.normals) {
            std::free(m.normals);
            m.normals = nullptr;
        }

        if (m.num_vertices <= 0 || m.verts == nullptr) {
            return;
        }

        vec3 *acc = static_cast<vec3*>(std::calloc(m.num_vertices, sizeof(vec3)));
        if (!acc) return;

        for (int t = 0; t < m.num_triangles; t++) {
            int i0 = m.indices[3 * t];
            int i1 = m.indices[3 * t + 1];
            int i2 = m.indices[3 * t + 2];

            vec3 e1 = m.verts[i1] - m.verts[i0];
            vec3 e2 = m.verts[i2] - m.verts[i0];
            vec3 n = cross(e1, e2);

            acc[i0] = acc[i0] + n;
            acc[i1] = acc[i1] + n;
            acc[i2] = acc[i2] + n;
        }

        for (int i = 0; i < m.num_vertices; i++) {
            float len_sq = acc[i].x * acc[i].x +
                           acc[i].y * acc[i].y +
                           acc[i].z * acc[i].z;

            if (len_sq > 1e-12f) {
                acc[i] = normalize(acc[i]);
            } else {
                acc[i] = make3(0.0f, 1.0f, 0.0f);             }
        }

        m.normals = acc;
    }

    std::vector<std::uint8_t> render_mesh(mesh& mesh_o, float* texture, int w, int h, const Mat4& model = Mat4::identity()) {
        if (mesh_o.normals == NULL) {
            make_normals(mesh_o);
        }

        float aspect = static_cast<float>(width) / static_cast<float>(height);
        float fov = 60.0f * 3.14159265f / 180.0f; // 60 градусов
        float near_p = 0.1f;
        float far_p = 100.0f;

        Mat4 proj = Mat4::perspective(fov, aspect, near_p, far_p);
        Mat4 view = Mat4::lookAt({0.0f, 0.0f, 2.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});

        Mat4 mvp = proj * view * model;

        float *img = texture;

        if (mesh_o.normals) {
            int flat = 0, smooth = 0;
            for (int t = 0; t < mesh_o.num_triangles; ++t) {
                vec3 a = mesh_o.normals[mesh_o.indices[3*t+0]];
                vec3 b = mesh_o.normals[mesh_o.indices[3*t+1]];
                vec3 c = mesh_o.normals[mesh_o.indices[3*t+2]];
                if (dot(a, b) > 0.9999f && dot(a, c) > 0.9999f) flat++;
                else smooth++;
            }
            bool split = (mesh_o.num_vertices == 3 * mesh_o.num_triangles);
            std::fprintf(stderr,
                "mesh v=%d t=%d split=%s flat_faces=%d smooth_faces=%d\n",
                mesh_o.num_vertices, mesh_o.num_triangles,
                split ? "YES" : "no", flat, smooth);
        }

        for (int i = 0; i < mesh_o.num_triangles; ++i) {
            int i0 = mesh_o.indices[i * 3 + 0];
            int i1 = mesh_o.indices[i * 3 + 1];
            int i2 = mesh_o.indices[i * 3 + 2];

            vec3 n1 = mesh_o.normals[i0];
            vec3 n2 = mesh_o.normals[i1];
            vec3 n3 = mesh_o.normals[i2];

            vec2 tc1{}, tc2{}, tc3{};
            if (mesh_o.tcs) {
                tc1 = mesh_o.tcs[i0];
                tc2 = mesh_o.tcs[i1];
                tc3 = mesh_o.tcs[i2];
            }

            vec3 verts[3] = { mesh_o.verts[i0], mesh_o.verts[i1], mesh_o.verts[i2] };
            vec4 clip_coords[3];
            bool culled = false;

            for (int v = 0; v < 3; ++v) {
                vec4 p_local = { verts[v].x, verts[v].y, verts[v].z, 1.0f };
                clip_coords[v] = mvp * p_local;

                if (clip_coords[v].w <= 0.001f) {
                    culled = true;
                    break;
                }
            }

            if (culled) continue;

            vec3 screen_coords[3];
            for (int v = 0; v < 3; ++v) {
                float inv_w = 1.0f / clip_coords[v].w;
                float ndc_x = clip_coords[v].x * inv_w;
                float ndc_y = clip_coords[v].y * inv_w;
                float ndc_z = clip_coords[v].z * inv_w;

                screen_coords[v].x = (ndc_x * 0.5f + 0.5f) * width;
                screen_coords[v].y = (ndc_y * 0.5f + 0.5f) * height;
                screen_coords[v].z = ndc_z;
            }

            float bright1 = dot(n1, (-1 * light_dir));
            float bright2 = dot(n2, (-1 * light_dir));
            float bright3 = dot(n3, (-1 * light_dir));

            float iw0 = 1.0f / clip_coords[0].w;
            float iw1 = 1.0f / clip_coords[1].w;
            float iw2 = 1.0f / clip_coords[2].w;

            Polygon3D poly(screen_coords[0], screen_coords[1], screen_coords[2], bright1, bright2, bright3, tc1, tc2, tc3, iw0, iw1, iw2, n1, n2, n3);
            render_triangle_3d(poly, img, w, h);

        }


        if (task == 3) visualize_depth();

        return pixels;
    }

    std::vector<std::uint8_t> forward(Polygon& content) {
        if (this->task == 1) {
            return render_triangle(content);
        }
        return std::vector<std::uint8_t>(width * height * 4, 255);
    }

    std::vector<std::uint8_t> forward(mesh& content) {
        if (this->task == 3 || this->task == 4) {
            return render_mesh(content, nullptr, 0, 0);
        }
        if (this->task == 5) {
            if (!original_textures.empty() && original_textures[0]) {
                return render_mesh(content,
                                   original_textures[0],
                                   img_sizes[0].first,
                                   img_sizes[0].second);
            }
            return render_mesh(content, nullptr, 0, 0);
        }
        return std::vector<std::uint8_t>(width * height * 4, 255);
    }

    std::vector<std::uint8_t> inner_forward() {
        std::fill(pixels.begin(), pixels.end(), 0);
        std::fill(z_buffer.begin(), z_buffer.end(), FARFARAWAY);

        std::fprintf(stderr,
                "task6: objects=%zu | mesh0: verts=%d tris=%d tcs=%p | tex0: %p %dx%d\n",
                objects.size(),
                objects.empty() ? 0 : meshes[objects[0].mesh_id].num_vertices,
                objects.empty() ? 0 : meshes[objects[0].mesh_id].num_triangles,
                objects.empty() ? nullptr : (void*)meshes[objects[0].mesh_id].tcs,
                objects.empty() ? nullptr : (void*)original_textures[objects[0].tex_id],
                objects.empty() ? 0 : img_sizes[objects[0].tex_id].first,
                objects.empty() ? 0 : img_sizes[objects[0].tex_id].second);

        for (size_t i = 0; i < objects.size(); ++i) {
            const Object3D& obj = objects[i];

            // mesh placed = place(meshes[obj.mesh_id], obj.where);
            Mat4 model = Mat4::identity();
            model.m[0][3] = obj.where.x;
            model.m[1][3] = obj.where.y;
            model.m[2][3] = obj.where.z;

            render_mesh(meshes[obj.mesh_id],
                        original_textures[obj.tex_id],
                        img_sizes[obj.tex_id].first,
                        img_sizes[obj.tex_id].second,
                        model);

        }
        return pixels;
    }

};

int main(int argc, char** argv)
{
    const int width  = argc > 1 ? std::atoi(argv[1]) : 640;
    const int height = argc > 2 ? std::atoi(argv[2]) : 480;
    const int task = argc > 3 ? std::atoi(argv[3]) : 1;
    const int num_of_obj = argc > 4 ? std::atoi(argv[4]) : 1;
    const std::string path_to_mesh  = argc > 5 ? argv[5] : "resources/Bunny.obj";
    const std::string path_to_texture  = argc > 6 ? argv[6] : "resources/textures/porcelain.jpg";
    if (width <= 0 || height <= 0)
    {
        std::printf("Usage: %s [width] [height]\n", argv[0]);
        return 1;
    }

    if (argc == 4) {
        if (std::atoi(argv[3]) < 1 || std::atoi(argv[3]) > 6) {
            return 1;
        }
    }

    std::vector<std::string>objects(num_of_obj);
    std::vector<std::string>texture_paths(num_of_obj);

    if (argc >= 5 + 2*num_of_obj) {
        for (int k = 0; k < num_of_obj; ++k) {
            objects[k]       = argv[5 + 2*k];
            texture_paths[k] = argv[5 + 2*k + 1];
        }
    } else {
        objects[0]       = path_to_mesh;
        texture_paths[0] = path_to_texture;
        for (int k = 1; k < num_of_obj; ++k) {
            objects[k]       = path_to_mesh;
            texture_paths[k] = path_to_texture;
        }
    }


    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(width, height, "Software Renderer", nullptr, nullptr);

    // GLFWmonitor* mon = glfwGetPrimaryMonitor();
    // const GLFWvidmode* mode = glfwGetVideoMode(mon);
    // glfwSetWindowMonitor(window, mon, 0, 0, mode->width, mode->height, mode->refreshRate);

    if (!window)
    {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSwapInterval(0);

    int fb_width = 0, fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);

    const std::size_t n_pixels = static_cast<std::size_t>(fb_width) * static_cast<std::size_t>(fb_height);
    std::vector<std::uint8_t> pixels(4 * n_pixels);

    constexpr float hue_speed = 0.2f;
    float hue = 0.0f;

    double prev_time = glfwGetTime();
    double report_time = prev_time;
    int report_frames = 0;
    vec3 sun;
    sun.x = 0.3; sun.y = -1; sun.z = -0.7;
    Render engine(task, fb_width, fb_height, objects, texture_paths, sun);

    mesh obj = load_obj(path_to_mesh.c_str());
    obj = engine.scale_mesh(obj);

    while (!glfwWindowShouldClose(window))
    {
        const double now = glfwGetTime();
        const auto dt = static_cast<float>(now - prev_time);
        prev_time = now;

        hue = std::fmod(hue + hue_speed * dt, 1.0f);

        float rgb[3];
        hue_to_rgb(hue, rgb);
        const std::uint8_t r = to_u8(rgb[0]);
        const std::uint8_t g = to_u8(rgb[1]);
        const std::uint8_t b = to_u8(rgb[2]);

        // for (std::size_t i = 0; i < n_pixels; ++i)
        // {
        //     pixels[4 * i + 0] = r;
        //     pixels[4 * i + 1] = g;
        //     pixels[4 * i + 2] = b;
        //     pixels[4 * i + 3] = 255;
        // }

        std::pair<float, float> p1(0.99 * fb_width, 0.95 * fb_height);
        std::pair<float, float> p2(0.02 * fb_width, 0.95 * fb_height);
        std::pair<float, float> p3(0.50 * fb_width, 0.05 * fb_height);

        // std::cout << p1.first << " " << p1.second << std::endl;

        if (task == 1) {
            Polygon default_p(p1, p2, p3);
            pixels = engine.forward(default_p);
        }
        else if (task == 3) {
            pixels = engine.forward(obj);
        }
        else if (task == 4) {
            pixels = engine.forward(obj);
        }
        else if (task == 5 and obj.tcs != NULL) {
            pixels = engine.forward(obj);
        }
        else if (task == 6) {
            pixels = engine.inner_forward();
        }
        else if (task == 99) {
            pixels = engine.inner_forward();
        }
        else {
            std::cout << "wtf task " << task << std::endl;
        }

        glDrawPixels(fb_width, fb_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        glfwSwapBuffers(window);
        glfwPollEvents();

        ++report_frames;
        const double elapsed = glfwGetTime() - report_time;
        if (elapsed >= 1.0)
        {
            std::printf("frame time: %.3f ms (%.1f FPS)\n",
                        1000.0 * elapsed / report_frames,
                        report_frames / elapsed);
            std::fflush(stdout);
            report_time += elapsed;
            report_frames = 0;
        }
    }

    free_mesh(&obj);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
