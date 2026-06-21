#pragma once

// Owns the full application lifecycle: GLFW window, OpenGL context,
// ImGui, the main loop, and all GLFW input callbacks.
// Composes Camera, NeuralNetwork, Renderer, and UI — each in its own class.
// All shared UI/simulation state lives in AppState.

#include "Camera.h"
#include "Network.h"
#include "Renderer.h"
#include "UI.h"
#include "AppState.h"

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
    GLFWwindow*   m_window   = nullptr;
    Camera        m_camera;       // orbital camera
    NeuralNetwork m_network;      // MLP data model + animation
    Renderer      m_renderer;     // OpenGL 3D draw pipeline
    UI            m_ui;           // ImGui side-panel
    AppState      m_state;        // all UI-editable shared state
    bool          m_ready    = false;

    // Init phases (each returns false on failure)
    bool initGLFW  (int w, int h, const char* title);
    bool initGLAD  ();
    bool initImGui ();
    bool initScene ();  // builds the network and calls Renderer::init()

    // Per-frame
    void update(float dt);
    void render();
    void rebuildNetwork();   // re-creates the NeuralNetwork from m_state.netConfig

    // GLFW static callback trampolines
    static void cbError      (int code, const char* msg);
    static void cbMouseButton(GLFWwindow*, int button, int action, int mods);
    static void cbCursorPos  (GLFWwindow*, double x, double y);
    static void cbScroll     (GLFWwindow*, double dx, double dy);
};
