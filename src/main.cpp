// Entry point for the Neural Network Visualizer.

#include "Camera.h"
#include "Network.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstdlib>

// Constants
static constexpr int  kInitWidth    = 1280;
static constexpr int  kInitHeight   = 720;
static constexpr char kWindowTitle[]  = "Neural Net Visualizer";
static constexpr char kGlslVersion[] = "#version 330 core";

// Application state
// Stored on the stack in main() and shared with GLFW callbacks via
// glfwSetWindowUserPointer — avoids any global mutable state.
struct AppState
{
    Camera        camera;
    NeuralNetwork network;
    // Milestone 4 will add: Renderer renderer;
};

// GLFW callbacks
static void glfwErrorCallback(int error, const char* description)
{
    std::fprintf(stderr, "[GLFW Error %d] %s\n", error, description);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    // Let ImGui handle its own clicks first
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    state->camera.onMouseButton(button, action, x, y);
}

static void cursorPosCallback(GLFWwindow* window, double x, double y)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    state->camera.onMouseMove(x, y);
}

static void scrollCallback(GLFWwindow* window, double /*offsetX*/, double offsetY)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    state->camera.onScroll(offsetY);
}

// ImGui theme
static void applyImGuiTheme()
{
    ImGui::StyleColorsDark();

    ImGuiStyle& style       = ImGui::GetStyle();
    style.WindowRounding    = 6.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding      = 4.0f;

    // Deep violet / electric blue accent palette
    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg]         = ImVec4(0.07f, 0.07f, 0.10f, 0.95f);
    c[ImGuiCol_Header]           = ImVec4(0.26f, 0.16f, 0.55f, 0.80f);
    c[ImGuiCol_HeaderHovered]    = ImVec4(0.36f, 0.22f, 0.75f, 0.85f);
    c[ImGuiCol_HeaderActive]     = ImVec4(0.40f, 0.26f, 0.85f, 1.00f);
    c[ImGuiCol_Button]           = ImVec4(0.20f, 0.12f, 0.50f, 0.85f);
    c[ImGuiCol_ButtonHovered]    = ImVec4(0.30f, 0.18f, 0.70f, 0.90f);
    c[ImGuiCol_ButtonActive]     = ImVec4(0.38f, 0.24f, 0.85f, 1.00f);
    c[ImGuiCol_FrameBg]          = ImVec4(0.12f, 0.10f, 0.20f, 0.80f);
    c[ImGuiCol_FrameBgHovered]   = ImVec4(0.20f, 0.16f, 0.35f, 0.80f);
    c[ImGuiCol_FrameBgActive]    = ImVec4(0.25f, 0.20f, 0.45f, 1.00f);
    c[ImGuiCol_TitleBg]          = ImVec4(0.05f, 0.05f, 0.09f, 1.00f);
    c[ImGuiCol_TitleBgActive]    = ImVec4(0.18f, 0.10f, 0.42f, 1.00f);
    c[ImGuiCol_SliderGrab]       = ImVec4(0.45f, 0.28f, 0.90f, 1.00f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.55f, 0.38f, 1.00f, 1.00f);
    c[ImGuiCol_CheckMark]        = ImVec4(0.55f, 0.38f, 1.00f, 1.00f);
    c[ImGuiCol_Tab]              = ImVec4(0.12f, 0.08f, 0.25f, 0.90f);
    c[ImGuiCol_TabHovered]       = ImVec4(0.30f, 0.18f, 0.65f, 0.90f);
    c[ImGuiCol_TabActive]        = ImVec4(0.22f, 0.13f, 0.50f, 1.00f);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.45f, 0.28f, 0.90f, 0.80f);
    c[ImGuiCol_SeparatorActive]  = ImVec4(0.55f, 0.38f, 1.00f, 1.00f);
}

// main
int main()
{
    // GLFW init
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        std::fprintf(stderr, "Failed to initialise GLFW.\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(kInitWidth, kInitHeight,
                                          kWindowTitle, nullptr, nullptr);
    if (!window)
    {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // GLAD
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::fprintf(stderr, "Failed to initialise GLAD.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    // Application state
    AppState state{ Camera(glm::vec3(0.0f), 45.0f, 25.0f, 14.0f) };

    // Build the MLP: Input(4) → Hidden1(8) → Hidden2(6) → Output(3)
    state.network.build({
        { 4, "Input"    },
        { 8, "Hidden 1" },
        { 6, "Hidden 2" },
        { 3, "Output"   }
    }, /*layerSpacing=*/3.0f, /*neuronSpacing=*/1.2f);

    // Share state with GLFW callbacks via user pointer
    glfwSetWindowUserPointer(window, &state);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback  (window, cursorPosCallback);
    glfwSetScrollCallback     (window, scrollCallback);

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    applyImGuiTheme();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(kGlslVersion);

    // OpenGL global state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // Main loop
    double prevTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Delta time
        const double now = glfwGetTime();
        const float  dt  = static_cast<float>(now - prevTime);
        prevTime = now;

        // Advance network simulation
        state.network.tick(dt);

        int fbWidth = 0, fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);

        glClearColor(0.04f, 0.04f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Camera matrices (will feed the renderer in Milestone 4)
        const glm::mat4 view = state.camera.getViewMatrix();
        const float aspect   = (fbHeight > 0)
                               ? static_cast<float>(fbWidth) / static_cast<float>(fbHeight)
                               : 1.0f;
        const glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 200.0f);

        // Suppress unused-variable warnings until the renderer is in place
        (void)view;
        (void)proj;

        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Info / controls panel
        ImGui::SetNextWindowPos (ImVec2(20.0f, 20.0f),  ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(300.0f, 0.0f),  ImGuiCond_Always);
        ImGui::Begin("Neural Net Visualizer", nullptr,
                     ImGuiWindowFlags_NoResize   |
                     ImGuiWindowFlags_NoMove     |
                     ImGuiWindowFlags_NoCollapse);

        ImGui::TextColored(ImVec4(0.55f, 0.38f, 1.00f, 1.0f),
                           "3D Neural Network Visualizer");
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextWrapped(
            "An interactive 3D architecture visualizer.\n"
            "Use mouse/keyboard controls to navigate the model.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Camera readout
        const glm::vec3 camPos = state.camera.getPosition();
        ImGui::Text("Camera");
        ImGui::TextDisabled("  Eye  (%.2f, %.2f, %.2f)",
                            camPos.x, camPos.y, camPos.z);
        ImGui::Spacing();
        ImGui::TextDisabled("LMB drag  -> orbit");
        ImGui::TextDisabled("Scroll    -> zoom");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Network stats
        ImGui::Text("Network  (%d layers  |  %d neurons  |  %d synapses)",
                    state.network.getLayerCount(),
                    state.network.getNeuronCount(),
                    state.network.getSynapseCount());
        ImGui::Spacing();

        const auto& layers  = state.network.getLayers();
        const auto& neurons = state.network.getNeurons();

        for (int li = 0; li < state.network.getLayerCount(); ++li)
        {
            const auto&    desc  = layers[li];
            const uint32_t base  = state.network.layerStart(li);
            const int      count = desc.neuronCount;

            // Average activation for this layer
            float avg = 0.0f;
            for (int ni = 0; ni < count; ++ni)
                avg += neurons[base + ni].activation;
            avg /= static_cast<float>(count);

            ImGui::TextDisabled("  %-10s  n=%d  avg=%.2f",
                                desc.label.c_str(), count, avg);

            // Activation bar
            ImGui::SameLine();
            const ImVec4 barCol =
                (li == 0)                                    ? ImVec4(0.30f, 0.80f, 0.40f, 1.0f) // input  – green
                : (li == state.network.getLayerCount() - 1) ? ImVec4(0.90f, 0.40f, 0.20f, 1.0f) // output – orange
                :                                             ImVec4(0.45f, 0.28f, 0.90f, 1.0f); // hidden – violet
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barCol);
            ImGui::ProgressBar(avg, ImVec2(-1.0f, 6.0f), "");
            ImGui::PopStyleColor();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("Renderer : %s", glGetString(GL_RENDERER));
        ImGui::TextDisabled("GL ver.  : %s", glGetString(GL_VERSION));

        ImGui::End();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}
