#pragma once

// Owns the full application lifecycle: GLFW window, OpenGL context,
// ImGui, the main loop, and all GLFW input callbacks.
// Composes Camera, NeuralNetwork, and Renderer — each in its own class.

#include "Camera.h"
#include "Network.h"
#include "Renderer.h"

// Forward-declare GLFWwindow so consumers of this header don't need GLFW.
struct GLFWwindow;

class Application
{
public:
    // Creates the window and initialises every subsystem.
    // If any step fails m_ready is left false and run() returns EXIT_FAILURE.
    Application(int width, int height, const char* title);
    ~Application();

    // Non-copyable / non-movable (owns a raw GLFWwindow* and GL objects).
    Application(const Application&)            = delete;
    Application& operator=(const Application&) = delete;

    // Runs the main loop until the window is closed.
    // Returns EXIT_SUCCESS or EXIT_FAILURE.
    int run();

private:
    // Subsystems
    GLFWwindow*   m_window  = nullptr;
    Camera        m_camera;       // orbital camera
    NeuralNetwork m_network;      // MLP data model + animation
    Renderer      m_renderer;     // OpenGL draw pipeline + ImGui panel
    bool          m_ready   = false;

    // Init phases (each returns false on failure)
    bool initGLFW  (int w, int h, const char* title);
    bool initGLAD  ();
    bool initImGui ();
    bool initScene ();  // builds the network and calls Renderer::init()

    // Per-frame
    void update(float dt);
    void render();
    void rebuildNetwork();   // re-creates the NeuralNetwork from netConfig

    // GLFW static callback trampolines
    // Each retrieves the Application* stored via glfwSetWindowUserPointer and
    // delegates to the appropriate member, after guarding against ImGui capture.
    static void cbError      (int code, const char* msg);
    static void cbMouseButton(GLFWwindow*, int button, int action, int mods);
    static void cbCursorPos  (GLFWwindow*, double x, double y);
    static void cbScroll     (GLFWwindow*, double dx, double dy);
};
