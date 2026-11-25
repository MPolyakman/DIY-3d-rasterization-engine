#include "render.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_map>

namespace {
constexpr float FARFARAWAY = std::numeric_limits<float>::max();
}

Render::Render(int todo, int w, int h,  std::vector<std::string>meshes_paths, std::vector<std::string>texture_paths, vec3 l) : task(todo), height(h), width(w),
            pixels(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4),  z_buffer(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), FARFARAWAY) {
                light_dir = (1 / std::sqrt(l.x * l.x + l.y * l.y + l.z * l.z)) * l;
                std::unordered_map<std::string, int>mesh_mapa;
                std::unordered_map<std::string, int>tex_mapa;
                meshes.resize(meshes_paths.size());
                original_textures.resize(meshes_paths.size(), nullptr);
                img_sizes.resize(meshes_paths.size());
                texture_mips.resize(meshes_paths.size());
                // objects.resize(meshes_paths.size());

                for (int i = 0; i < (int)meshes_paths.size(); ++i) {
                    meshes[i] = load_obj(meshes_paths[i].c_str());

                    int wi = 0, hi = 0;
                    original_textures[i] = load_image_f32_rgb(texture_paths[i].c_str(), &wi, &hi, 1.0f);
                    img_sizes[i] = { wi, hi };
                    if (original_textures[i])
                        texture_mips[i] = build_mips(original_textures[i], wi, hi);

                    if (!mesh_mapa.contains(meshes_paths[i]))
                        mesh_mapa[meshes_paths[i]] = i;
                    unsigned obj_mesh_id = (unsigned)mesh_mapa[meshes_paths[i]];

                    if (!tex_mapa.contains(texture_paths[i]))
                        tex_mapa[texture_paths[i]] = i;
                    unsigned obj_tex_id = (unsigned)tex_mapa[texture_paths[i]];

                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> dist(-0.9f, 0.9f);
                    vec3 place{ dist(gen), dist(gen) + 1.5f, dist(gen) - 0.5f };

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

                if (task == 6) {
                    generate_terrain(terrain_seed);
                    int w = 0, h = 0;
                    float* grass = load_image_f32_rgb("resources/textures/Grass_09-256x256.png", &w, &h, 1.0f);
                    if (grass) grass_mips = build_mips(grass, w, h);
                    std::free(grass);
                    float* rock = load_image_f32_rgb("resources/textures/Grass_13-256x256.png", &w, &h, 1.0f);
                    if (rock) rock_mips = build_mips(rock, w, h);
                    std::free(rock);
                    const char* faces[] = {
                        "resources/textures/skybox/hills_rt.png",
                        "resources/textures/skybox/hills_lf.png",
                        "resources/textures/skybox/hills_up.jpg",
                        "resources/textures/skybox/hills_dn.png",
                        "resources/textures/skybox/hills_ft.png",
                        "resources/textures/skybox/hills_bk.png"
                    };
                    skybox_images.resize(6, nullptr);
                    skybox_sizes.resize(6);
                    for (int i = 0; i < 6; ++i)
                        skybox_images[i] = load_image_f32_rgb(faces[i], &skybox_sizes[i].first, &skybox_sizes[i].second, 1.0f);
                    camera_position = {0.0f, height_at(0.0f, 4.0f) + 2.0f, 4.0f};
                    camera_pitch = -0.2f;
                }
            }

Render::~Render() {
            for (float* t : original_textures) std::free(t);
            for (float* t : skybox_images) std::free(t);
            for (auto& m : meshes) free_mesh(&m);
            free_mesh(&terrain_mesh);
        }

std::vector<MipLevel> Render::build_mips(const float* image, int w, int h) {
            std::vector<MipLevel> levels;
            if (!image || w <= 0 || h <= 0) return levels;
            MipLevel base{w, h, std::vector<float>(image, image + static_cast<std::size_t>(w) * h * 3)};
            levels.push_back(std::move(base));
            while (w > 1 || h > 1) {
                const MipLevel& src = levels.back();
                int nw = std::max(1, (w + 1) / 2);
                int nh = std::max(1, (h + 1) / 2);
                MipLevel dst{nw, nh, std::vector<float>(static_cast<std::size_t>(nw) * nh * 3)};
                for (int y = 0; y < nh; ++y) {
                    for (int x = 0; x < nw; ++x) {
                        int count = 0;
                        for (int oy = 0; oy < 2; ++oy) {
                            for (int ox = 0; ox < 2; ++ox) {
                                int sx = 2 * x + ox, sy = 2 * y + oy;
                                if (sx >= w || sy >= h) continue;
                                for (int c = 0; c < 3; ++c)
                                    dst.pixels[(static_cast<std::size_t>(y) * nw + x) * 3 + c] += src.pixels[(static_cast<std::size_t>(sy) * w + sx) * 3 + c];
                                ++count;
                            }
                        }
                        for (int c = 0; c < 3; ++c)
                            dst.pixels[(static_cast<std::size_t>(y) * nw + x) * 3 + c] /= count;
                    }
                }
                levels.push_back(std::move(dst));
                w = nw;
                h = nh;
            }
            return levels;
        }

float Render::noise_hash(int x, int y, std::uint32_t seed, int period) const {
            x = (x % period + period) % period;
            y = (y % period + period) % period;
            std::uint32_t n = static_cast<std::uint32_t>(x) * 374761393u + static_cast<std::uint32_t>(y) * 668265263u + static_cast<std::uint32_t>(seed) * 1442695041u;
            n = (n ^ (n >> 13)) * 1274126177u;
            n ^= n >> 16;
            return static_cast<float>(n) / static_cast<float>(UINT32_MAX) * 2.0f - 1.0f;
        }

float Render::value_noise(float x, float y, std::uint32_t seed, int period) const {
            int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
            float fx = x - ix, fy = y - iy;
            fx = fx * fx * (3.0f - 2.0f * fx);
            fy = fy * fy * (3.0f - 2.0f * fy);
            float a = noise_hash(ix, iy, seed, period);
            float b = noise_hash(ix + 1, iy, seed, period);
            float c = noise_hash(ix, iy + 1, seed, period);
            float d = noise_hash(ix + 1, iy + 1, seed, period);
            return (1.0f - fy) * ((1.0f - fx) * a + fx * b) + fy * ((1.0f - fx) * c + fx * d);
        }

float Render::terrain_noise(float x, float y, std::uint32_t seed) const {
            float sum = 0.0f, amplitude = 1.0f, total = 0.0f;
            for (int octave = 0; octave < 5; ++octave) {
                int frequency = 1 << octave;
                sum += amplitude * value_noise(x * frequency / 32.0f, y * frequency / 32.0f, seed + octave, 8 * frequency);
                total += amplitude;
                amplitude *= 0.5f;
            }
            return sum / total;
        }

void Render::generate_terrain(std::uint32_t seed) {
            heightmap.resize(static_cast<std::size_t>(heightmap_size) * heightmap_size);
            for (int y = 0; y < heightmap_size; ++y)
                for (int x = 0; x < heightmap_size; ++x)
                    heightmap[static_cast<std::size_t>(y) * heightmap_size + x] = 0.5f + 0.5f * terrain_noise(static_cast<float>(x), static_cast<float>(y), seed);

            const int side = terrain_grid_size;
            const float radius = (side - 1) * 0.5f;
            terrain_mesh.num_vertices = side * side;
            terrain_mesh.num_triangles = (side - 1) * (side - 1) * 2;
            terrain_mesh.verts = static_cast<vec3*>(std::malloc(terrain_mesh.num_vertices * sizeof(vec3)));
            terrain_mesh.normals = static_cast<vec3*>(std::malloc(terrain_mesh.num_vertices * sizeof(vec3)));
            terrain_mesh.tcs = static_cast<vec2*>(std::malloc(terrain_mesh.num_vertices * sizeof(vec2)));
            terrain_mesh.indices = static_cast<int*>(std::malloc(terrain_mesh.num_triangles * 3 * sizeof(int)));
            terrain_grid.resize(terrain_mesh.num_vertices);
            for (int z = 0; z < side; ++z) {
                for (int x = 0; x < side; ++x) {
                    int i = z * side + x;
                    terrain_grid[i] = {x - radius, z - radius};
                    terrain_mesh.verts[i] = {x - radius, 0.0f, z - radius};
                    terrain_mesh.normals[i] = {0.0f, 1.0f, 0.0f};
                    terrain_mesh.tcs[i] = {0.0f, 0.0f};
                }
            }
            int t = 0;
            for (int z = 0; z < side - 1; ++z) {
                for (int x = 0; x < side - 1; ++x) {
                    int a = z * side + x, b = a + side, c = a + 1, d = b + 1;
                    terrain_mesh.indices[t++] = a; terrain_mesh.indices[t++] = b; terrain_mesh.indices[t++] = c;
                    terrain_mesh.indices[t++] = c; terrain_mesh.indices[t++] = b; terrain_mesh.indices[t++] = d;
                }
            }
        }

float Render::height_at(float x, float z) const {
            if (heightmap.empty()) return 0.0f;
            float fx = x * 2.0f + heightmap_size * 0.5f;
            float fy = z * 2.0f + heightmap_size * 0.5f;
            int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
            float ax = fx - x0, ay = fy - y0;
            auto at = [&](int px, int py) {
                px = (px % heightmap_size + heightmap_size) % heightmap_size;
                py = (py % heightmap_size + heightmap_size) % heightmap_size;
                return heightmap[static_cast<std::size_t>(py) * heightmap_size + px];
            };
            float h0 = at(x0, y0) * (1.0f - ax) + at(x0 + 1, y0) * ax;
            float h1 = at(x0, y0 + 1) * (1.0f - ax) + at(x0 + 1, y0 + 1) * ax;
            return (h0 * (1.0f - ay) + h1 * ay) * 5.0f;
        }

void Render::update_terrain() {
            float origin_x = std::round(camera_position.x);
            float origin_z = std::round(camera_position.z);
            for (int i = 0; i < terrain_mesh.num_vertices; ++i) {
                float x = origin_x + terrain_grid[i].x;
                float z = origin_z + terrain_grid[i].y;
                terrain_mesh.verts[i] = {x, height_at(x, z), z};
                terrain_mesh.normals[i] = normalize({height_at(x - 0.5f, z) - height_at(x + 0.5f, z), 1.0f, height_at(x, z - 0.5f) - height_at(x, z + 0.5f)});
                terrain_mesh.tcs[i] = {x * 0.08f, z * 0.08f};
            }
        }

std::vector<std::uint8_t> Render::render_default() {
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

mesh Render::weld_and_compute_normals(const mesh& src, float eps) {
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

std::vector<float> Render::cross_product(std::pair<float, float> pixel, Polygon trig) {
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

std::vector<std::uint8_t> Render::render_triangle(Polygon poly) {
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

mesh Render::scale_mesh(const mesh &src)
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

mesh Render::place(const mesh& instance, vec3 shift) {
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

vec3 Render::sample_level(const MipLevel& level, float u, float v, bool repeat) const {
        float fx = u * level.width - 0.5f, fy = v * level.height - 0.5f;
        int ix = static_cast<int>(std::floor(fx)), iy = static_cast<int>(std::floor(fy));
        float ax = fx - ix, ay = fy - iy;
        auto coordinate = [&](int p, int size) {
            return repeat ? (p % size + size) % size : std::clamp(p, 0, size - 1);
        };
        auto sample = [&](int x, int y, int c) {
            x = coordinate(x, level.width);
            y = coordinate(y, level.height);
            return level.pixels[(static_cast<std::size_t>(y) * level.width + x) * 3 + c];
        };
        float rgb[3];
        for (int c = 0; c < 3; ++c) {
            float a = sample(ix, iy, c) * (1.0f - ax) + sample(ix + 1, iy, c) * ax;
            float b = sample(ix, iy + 1, c) * (1.0f - ax) + sample(ix + 1, iy + 1, c) * ax;
            rgb[c] = a * (1.0f - ay) + b * ay;
        }
        return {rgb[0], rgb[1], rgb[2]};
    }

vec3 Render::sample_mips(const std::vector<MipLevel>& levels, float u, float v, float lod, bool repeat) const {
        if (levels.empty()) return {1.0f, 1.0f, 1.0f};
        lod = std::clamp(lod, 0.0f, static_cast<float>(levels.size() - 1));
        int lo = static_cast<int>(std::floor(lod));
        int hi = std::min(lo + 1, static_cast<int>(levels.size()) - 1);
        float blend = lod - lo;
        vec3 a = sample_level(levels[lo], u, v, repeat);
        vec3 b = sample_level(levels[hi], u, v, repeat);
        return (1.0f - blend) * a + blend * b;
    }

void Render::render_triangle_3d(const Polygon3D& poly, float* img, int w, int h,
                            const std::vector<MipLevel>* mips,
                            const std::vector<MipLevel>* rock,
                            bool terrain) {
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

                            if ((!mips || mips->empty()) && (img == nullptr || w <= 0 || h <= 0)) {
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

                                float lod = 0.0f;
                                const int tw = mips && !mips->empty() ? mips->front().width : w;
                                const int th = mips && !mips->empty() ? mips->front().height : h;
                                if (mips && !mips->empty()) {
                                    if (derivative_mips) {
                                        float dbdx[3] = {(v3.y - v2.y) / area2, (v1.y - v3.y) / area2, (v2.y - v1.y) / area2};
                                        float dbdy[3] = {(v3.x - v2.x) / area2, (v1.x - v3.x) / area2, (v2.x - v1.x) / area2};
                                        float invws[3] = {poly.inv_w1, poly.inv_w2, poly.inv_w3};
                                        float us[3] = {poly.t1.x, poly.t2.x, poly.t3.x};
                                        float vs[3] = {poly.t1.y, poly.t2.y, poly.t3.y};
                                        float bs[3] = {b1, b2, b3};
                                        float ddenx = 0.0f, ddeny = 0.0f, dunx = 0.0f, duny = 0.0f, dvnx = 0.0f, dvny = 0.0f;
                                        float un = 0.0f, vn = 0.0f;
                                        for (int k = 0; k < 3; ++k) {
                                            ddenx += dbdx[k] * invws[k];
                                            ddeny += dbdy[k] * invws[k];
                                            dunx += dbdx[k] * us[k] * invws[k];
                                            duny += dbdy[k] * us[k] * invws[k];
                                            dvnx += dbdx[k] * vs[k] * invws[k];
                                            dvny += dbdy[k] * vs[k] * invws[k];
                                            un += bs[k] * us[k] * invws[k];
                                            vn += bs[k] * vs[k] * invws[k];
                                        }
                                        float du_dx = (dunx * inv_w_p - un * ddenx) / (inv_w_p * inv_w_p);
                                        float du_dy = (duny * inv_w_p - un * ddeny) / (inv_w_p * inv_w_p);
                                        float dv_dx = (dvnx * inv_w_p - vn * ddenx) / (inv_w_p * inv_w_p);
                                        float dv_dy = (dvny * inv_w_p - vn * ddeny) / (inv_w_p * inv_w_p);
                                        float rho_x = std::sqrt(du_dx * du_dx * tw * tw + dv_dx * dv_dx * th * th);
                                        float rho_y = std::sqrt(du_dy * du_dy * tw * tw + dv_dy * dv_dy * th * th);
                                        lod = std::log2(std::max({rho_x, rho_y, 1.0f}));
                                    } else {
                                        float near_p = 0.1f, far_p = 100.0f;
                                        float distance = (2.0f * near_p * far_p) / (far_p + near_p - z_ndc * (far_p - near_p));
                                        lod = std::log2(std::max(1.0f, distance * 0.25f));
                                    }
                                    lod = std::clamp(lod, 0.0f, static_cast<float>(mips->size() - 1));
                                }

                                if (mip_debug && mips && !mips->empty()) {
                                    int level = std::clamp(static_cast<int>(std::lround(lod)), 0, static_cast<int>(mips->size()) - 1);
                                    float color[3];
                                    hue_to_rgb(static_cast<float>(level) / mips->size(), color);
                                    for (int c = 0; c < 3; ++c)
                                        pixels[idx * 4 + c] = static_cast<std::uint8_t>(color[c] * 255.0f);
                                } else if (mips && !mips->empty()) {
                                    vec3 color = sample_mips(*mips, u, v, lod, terrain);
                                    if (terrain && rock && !rock->empty()) {
                                        vec3 stone = sample_mips(*rock, u, v, lod, true);
                                        float amount = std::clamp((0.9f - n.y) / 0.4f, 0.0f, 1.0f);
                                        color = (1.0f - amount) * color + amount * stone;
                                    }
                                    pixels[idx*4+0] = (std::uint8_t)std::clamp(color.x * 255.0f * br, 0.0f, 255.0f);
                                    pixels[idx*4+1] = (std::uint8_t)std::clamp(color.y * 255.0f * br, 0.0f, 255.0f);
                                    pixels[idx*4+2] = (std::uint8_t)std::clamp(color.z * 255.0f * br, 0.0f, 255.0f);
                                } else {
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
                                }
                                pixels[idx*4+3] = 255;
                            }

                        }
                    }
                }
            }
        }
    }

void Render::visualize_depth() {
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

void Render::make_normals(mesh& m) {
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

std::vector<std::uint8_t> Render::render_mesh(mesh& mesh_o, float* texture, int w, int h,
                                          const Mat4& model,
                                          const std::vector<MipLevel>* mips,
                                          const std::vector<MipLevel>* rock,
                                          bool terrain) {
        if (mesh_o.normals == NULL) {
            make_normals(mesh_o);
        }

        float aspect = static_cast<float>(width) / static_cast<float>(height);
        float fov = 60.0f * 3.14159265f / 180.0f; // 60 градусов
        float near_p = 0.1f;
        float far_p = 100.0f;

        Mat4 proj = Mat4::perspective(fov, aspect, near_p, far_p);
        vec3 camera_direction{
            std::cos(camera_yaw) * std::cos(camera_pitch),
            std::sin(camera_pitch),
            std::sin(camera_yaw) * std::cos(camera_pitch)
        };
        Mat4 view = Mat4::lookAt(camera_position, camera_position + camera_direction, {0.0f, 1.0f, 0.0f});

        Mat4 mvp = proj * view * model;

        float *img = texture;

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
            render_triangle_3d(poly, img, w, h, mips, rock, terrain);

        }


        if (task == 3) visualize_depth();

        return pixels;
    }

void Render::draw_skybox() {
        if (skybox_images.size() != 6) return;
        vec3 forward{
            std::cos(camera_yaw) * std::cos(camera_pitch),
            std::sin(camera_pitch),
            std::sin(camera_yaw) * std::cos(camera_pitch)
        };
        vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
        vec3 up = normalize(cross(right, forward));
        float tan_half = std::tan(60.0f * 3.14159265f / 360.0f);
        float aspect = static_cast<float>(width) / height;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (z_buffer[idx] < FARFARAWAY) continue;
                float sx = (2.0f * (x + 0.5f) / width - 1.0f) * tan_half * aspect;
                float sy = (2.0f * (y + 0.5f) / height - 1.0f) * tan_half;
                vec3 ray = forward + sx * right + sy * up;
                float ax = std::fabs(ray.x), ay = std::fabs(ray.y), az = std::fabs(ray.z);
                int face;
                float u, v, major;
                if (ax >= ay && ax >= az) {
                    major = ax;
                    if (ray.x >= 0.0f) { face = 0; u = -ray.z / major; }
                    else { face = 1; u = ray.z / major; }
                    v = ray.y / major;
                } else if (ay >= ax && ay >= az) {
                    major = ay;
                    face = ray.y >= 0.0f ? 2 : 3;
                    u = ray.x / major;
                    v = (ray.y >= 0.0f ? -ray.z : ray.z) / major;
                } else {
                    major = az;
                    if (ray.z >= 0.0f) { face = 4; u = ray.x / major; }
                    else { face = 5; u = -ray.x / major; }
                    v = ray.y / major;
                }
                if (!skybox_images[face]) continue;
                u = u * 0.5f + 0.5f;
                v = v * 0.5f + 0.5f;
                int iw = skybox_sizes[face].first, ih = skybox_sizes[face].second;
                float fx = u * iw - 0.5f, fy = v * ih - 0.5f;
                int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, iw - 1);
                int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, ih - 1);
                int x1 = std::min(x0 + 1, iw - 1), y1 = std::min(y0 + 1, ih - 1);
                float a = fx - std::floor(fx), b = fy - std::floor(fy);
                for (int c = 0; c < 3; ++c) {
                    float c00 = skybox_images[face][(y0 * iw + x0) * 3 + c];
                    float c10 = skybox_images[face][(y0 * iw + x1) * 3 + c];
                    float c01 = skybox_images[face][(y1 * iw + x0) * 3 + c];
                    float c11 = skybox_images[face][(y1 * iw + x1) * 3 + c];
                    float value = (1.0f - b) * ((1.0f - a) * c00 + a * c10) + b * ((1.0f - a) * c01 + a * c11);
                    pixels[idx * 4 + c] = static_cast<std::uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
                }
                pixels[idx * 4 + 3] = 255;
            }
        }
    }

std::vector<std::uint8_t> Render::forward(Polygon& content) {
        if (this->task == 1) {
            return render_triangle(content);
        }
        return std::vector<std::uint8_t>(width * height * 4, 255);
    }

std::vector<std::uint8_t> Render::forward(mesh& content) {
        if (this->task == 3 || this->task == 4) {
            return render_mesh(content, nullptr, 0, 0);
        }
        if (this->task == 5) {
            if (!original_textures.empty() && original_textures[0]) {
                return render_mesh(content,
                                   original_textures[0],
                                   img_sizes[0].first,
                                   img_sizes[0].second,
                                   Mat4::identity(),
                                   texture_mips.empty() ? nullptr : &texture_mips[0]);
            }
            return render_mesh(content, nullptr, 0, 0);
        }
        return std::vector<std::uint8_t>(width * height * 4, 255);
    }

const std::vector<std::uint8_t>& Render::inner_forward() {
        std::fill(pixels.begin(), pixels.end(), 0);
        std::fill(z_buffer.begin(), z_buffer.end(), FARFARAWAY);

        if (task == 6) {
            update_terrain();
            render_mesh(terrain_mesh, nullptr, 0, 0,
                        Mat4::identity(), &grass_mips, &rock_mips, true);
        }

        // std::fprintf(stderr,
        //         "task6: objects=%zu | mesh0: verts=%d tris=%d tcs=%p | tex0: %p %dx%d\n",
        //         objects.size(),
        //         objects.empty() ? 0 : meshes[objects[0].mesh_id].num_vertices,
        //         objects.empty() ? 0 : meshes[objects[0].mesh_id].num_triangles,
        //         objects.empty() ? nullptr : (void*)meshes[objects[0].mesh_id].tcs,
        //         objects.empty() ? nullptr : (void*)original_textures[objects[0].tex_id],
        //         objects.empty() ? 0 : img_sizes[objects[0].tex_id].first,
        //         objects.empty() ? 0 : img_sizes[objects[0].tex_id].second);

        for (size_t i = 0; i < objects.size(); ++i) {
            const Object3D& obj = objects[i];

            // mesh placed = place(meshes[obj.mesh_id], obj.where);
            Mat4 model = Mat4::identity();
            model.m[0][3] = obj.where.x;
            model.m[1][3] = obj.where.y + (task == 6 ? height_at(obj.where.x, obj.where.z) : 0.0f);
            model.m[2][3] = obj.where.z;

            render_mesh(meshes[obj.mesh_id],
                        original_textures[obj.tex_id],
                        img_sizes[obj.tex_id].first,
                        img_sizes[obj.tex_id].second,
                        model,
                        obj.tex_id < texture_mips.size() ? &texture_mips[obj.tex_id] : nullptr);

        }

        if (task == 6) draw_skybox();

        if (scale_factor == 1)
            return pixels;

        const int src_w = width;
        const int src_h = height;
        const int dst_w = width  * scale_factor;
        const int dst_h = height * scale_factor;
        const float inv_s = 1.0f / static_cast<float>(scale_factor);

        scaled_pixels.resize(4 * dst_w * dst_h);

        for (int dy = 0; dy < dst_h; ++dy) {
            // координата центра dst-строки в src-пространстве
            float sy    = (dy + 0.5f) * inv_s - 0.5f;
            int   y0    = (int)std::floor(sy);
            float beta  = sy - y0;
            int   y1    = y0 + 1;

            y0 = std::clamp(y0, 0, src_h - 1);
            y1 = std::clamp(y1, 0, src_h - 1);

            const int row0 = y0 * src_w;
            const int row1 = y1 * src_w;

            for (int dx = 0; dx < dst_w; ++dx) {
                float sx    = (dx + 0.5f) * inv_s - 0.5f;
                int   x0    = (int)std::floor(sx);
                float alpha = sx - x0;
                int   x1    = x0 + 1;

                x0 = std::clamp(x0, 0, src_w - 1);
                x1 = std::clamp(x1, 0, src_w - 1);

                const int i00 = (row0 + x0) * 4;
                const int i10 = (row0 + x1) * 4;
                const int i01 = (row1 + x0) * 4;
                const int i11 = (row1 + x1) * 4;

                const float w00 = (1 - alpha) * (1 - beta);
                const float w10 =      alpha  * (1 - beta);
                const float w01 = (1 - alpha) *      beta;
                const float w11 =      alpha  *      beta;

                const int o = (dy * dst_w + dx) * 4;

                for (int c = 0; c < 4; ++c) {
                    float v = w00 * pixels[i00 + c]
                            + w10 * pixels[i10 + c]
                            + w01 * pixels[i01 + c]
                            + w11 * pixels[i11 + c];
                    scaled_pixels[o + c] = (std::uint8_t)std::clamp(v + 0.5f, 0.0f, 255.0f);
                }
            }
        }
        return scaled_pixels;
    }
