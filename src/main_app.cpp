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
#include "render.hpp"



void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    static bool first_mouse = true;
    static double last_x = 0.0, last_y = 0.0;
    if (first_mouse) {
        last_x = xpos;
        last_y = ypos;
        first_mouse = false;
        return;
    }

    Render* engine = static_cast<Render*>(glfwGetWindowUserPointer(window));
    if (!engine) return;

    const float sensitivity = 0.0025f;
    engine->camera_yaw += static_cast<float>(xpos - last_x) * sensitivity;
    engine->camera_pitch -= static_cast<float>(ypos - last_y) * sensitivity;
    engine->camera_pitch = std::clamp(engine->camera_pitch, -1.553343f, 1.553343f);
    last_x = xpos;
    last_y = ypos;
}

int main(int argc, char** argv)
{
    int width  = argc > 1 ? std::atoi(argv[1]) : 640;
    int height = argc > 2 ? std::atoi(argv[2]) : 480;
    const int task = argc > 3 ? std::atoi(argv[3]) : 1;
    const int num_of_obj = argc > 4 ? std::atoi(argv[4]) : 1;
    const std::string path_to_mesh  = argc > 5 ? argv[5] : "resources/Bunny.obj";
    const std::string path_to_texture  = argc > 6 ? argv[6] : "resources/textures/porcelain.jpg";

    const unsigned scale_factor = task == 6 ? 4 : 1;

    // width /= scale_factor;
    // height /= scale_factor;

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
    Render engine(task, fb_width / scale_factor, fb_height / scale_factor, objects, texture_paths, sun);
    engine.scale_factor = scale_factor;
    glfwSetWindowUserPointer(window, &engine);
    if (task == 6) {
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    mesh obj = load_obj(path_to_mesh.c_str());
    obj = engine.scale_mesh(obj);
    bool m_was_down = false, b_was_down = false, t_was_down = false;

    while (!glfwWindowShouldClose(window))
    {
        const std::vector<std::uint8_t>* frame_pixels = &pixels;
        const double now = glfwGetTime();
        const auto dt = static_cast<float>(now - prev_time);
        prev_time = now;

        bool m_down = glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS;
        bool b_down = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
        bool t_down = glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS;
        if (m_down && !m_was_down) engine.mip_debug = !engine.mip_debug;
        if (b_down && !b_was_down) engine.derivative_mips = !engine.derivative_mips;
        if (t_down && !t_was_down) engine.terrain_camera = !engine.terrain_camera;
        m_was_down = m_down;
        b_was_down = b_down;
        t_was_down = t_down;

        if (task == 6) {
            vec3 forward{
                std::cos(engine.camera_yaw) * std::cos(engine.camera_pitch),
                std::sin(engine.camera_pitch),
                std::sin(engine.camera_yaw) * std::cos(engine.camera_pitch)
            };
            vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
            const float speed = 2.5f * dt;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) engine.camera_position = engine.camera_position + speed * forward;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) engine.camera_position = engine.camera_position - speed * forward;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) engine.camera_position = engine.camera_position - speed * right;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) engine.camera_position = engine.camera_position + speed * right;
            if (engine.terrain_camera)
                engine.camera_position.y = engine.height_at(engine.camera_position.x, engine.camera_position.z) + 1.8f;
        }

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
            frame_pixels = &engine.inner_forward();
        }
        else if (task == 99) {
            frame_pixels = &engine.inner_forward();
        }
        else {
            std::cout << "wtf task " << task << std::endl;
        }

        glDrawPixels(fb_width, fb_height, GL_RGBA, GL_UNSIGNED_BYTE, frame_pixels->data());
        // glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
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
