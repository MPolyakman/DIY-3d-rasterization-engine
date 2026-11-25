#pragma once

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
#include <string>

#include "mesh_utils.h"
#include "vectors.h"
#include "image_utils.h"

#include "geomath.hpp"

class Render
{
    public:
        int task;
        int height;
        int width;
        std::vector<std::uint8_t>pixels;
        std::vector<std::uint8_t>scaled_pixels;
        std::vector<float>z_buffer;

        vec3 light_dir;

        std::vector<mesh>meshes;
        std::vector<float*>original_textures;
        std::vector<std::pair<int, int>>img_sizes;
        std::vector<Object3D>objects;
        std::vector<std::vector<MipLevel>>texture_mips;
        vec3 camera_position{0.0f, 0.0f, 2.5f};
        float camera_yaw = -1.57079632679f;
        float camera_pitch = 0.0f;
        bool mip_debug = false;
        bool derivative_mips = false;
        bool terrain_camera = false;
        std::vector<float> heightmap;
        int heightmap_size = 256;
        int terrain_grid_size = 65;
        std::uint32_t terrain_seed = 1337;
        mesh terrain_mesh{};
        std::vector<vec2> terrain_grid;
        std::vector<float*> skybox_images;
        std::vector<std::pair<int, int>> skybox_sizes;
        std::vector<MipLevel> grass_mips;
        std::vector<MipLevel> rock_mips;

        unsigned scale_factor = 1;


        Render(int todo, int w, int h,  std::vector<std::string>meshes_paths, std::vector<std::string>texture_paths, vec3 l = {1, 0, 0});

        ~Render();

        std::vector<MipLevel> build_mips(const float* image, int w, int h);

        float noise_hash(int x, int y, std::uint32_t seed, int period) const;

        float value_noise(float x, float y, std::uint32_t seed, int period) const;

        float terrain_noise(float x, float y, std::uint32_t seed) const;

        void generate_terrain(std::uint32_t seed);

        float height_at(float x, float z) const;

        void update_terrain();

        std::vector<std::uint8_t> render_default();

        mesh weld_and_compute_normals(const mesh& src, float eps = 1e-5f);

        std::vector<float> cross_product(std::pair<float, float> pixel, Polygon trig);

        std::vector<std::uint8_t> render_triangle(Polygon poly);

    mesh scale_mesh(const mesh &src);

    mesh place(const mesh& instance, vec3 shift);

    vec3 sample_level(const MipLevel& level, float u, float v, bool repeat) const;

    vec3 sample_mips(const std::vector<MipLevel>& levels, float u, float v, float lod, bool repeat) const;

    void render_triangle_3d(const Polygon3D& poly, float* img, int w, int h,
                            const std::vector<MipLevel>* mips = nullptr,
                            const std::vector<MipLevel>* rock = nullptr,
                            bool terrain = false);

    void visualize_depth();

    void make_normals(mesh& m);

    std::vector<std::uint8_t> render_mesh(mesh& mesh_o, float* texture, int w, int h,
                                          const Mat4& model = Mat4::identity(),
                                          const std::vector<MipLevel>* mips = nullptr,
                                          const std::vector<MipLevel>* rock = nullptr,
                                          bool terrain = false);

    void draw_skybox();

    std::vector<std::uint8_t> forward(Polygon& content);

    std::vector<std::uint8_t> forward(mesh& content);

        const std::vector<std::uint8_t>& inner_forward();

};
