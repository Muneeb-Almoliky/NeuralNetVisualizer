// Full application lifecycle: GLFW window, OpenGL context, ImGui, main loop.

#include "Application.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Platform helpers
#ifdef _WIN32
// GLAD must be included before windows.h to avoid WGL redefinition warnings.
#include <windows.h>
#include <commdlg.h>

// Opens the Windows "Open File" dialog and returns the chosen path,
// or an empty string if the user cancels.
static std::string openFileDialog()
{
    char path[MAX_PATH] = {};
    OPENFILENAMEA ofn   = {};
    ofn.lStructSize     = sizeof(ofn);
    ofn.lpstrFilter     = "CSV Files\0*.csv\0All Files\0*.*\0";
    ofn.lpstrFile       = path;
    ofn.nMaxFile        = MAX_PATH;
    ofn.Flags           = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    return GetOpenFileNameA(&ofn) ? std::string(path) : std::string{};
}
#else
static std::string openFileDialog() { return {}; } // stub for non-Windows
#endif

static constexpr char kGlslVersion[] = "#version 330 core";

// Construction / Destruction
Application::Application(int width, int height, const char* title)
    : m_camera(glm::vec3(0.0f), 45.0f, 25.0f, 14.0f)
{
    if (!initGLFW  (width, height, title)) return;
    if (!initGLAD  ())                     return;
    if (!initImGui ())                     return;
    if (!initScene ())                     return;
    m_ready = true;
}

Application::~Application()
{
    if (m_window)
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(m_window);
    }
    glfwTerminate();
}

// run
int Application::run()
{
    if (!m_ready) return EXIT_FAILURE;

    double prevTime = glfwGetTime();

    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        const double now = glfwGetTime();
        const float  dt  = static_cast<float>(now - prevTime);
        prevTime = now;

        update(dt);
        render();
        glfwSwapBuffers(m_window);
    }

    return EXIT_SUCCESS;
}

// update — advance simulation, handle UI commands
void Application::update(float dt)
{
    auto& cfg   = m_state.netConfig;
    auto& train = m_state.train;
    auto& prop  = m_state.prop;

    // Architecture rebuild
    if (cfg.rebuildPending)
    {
        cfg.rebuildPending = false;
        cfg.weightsLoaded  = false;
        rebuildNetwork();
    }

    // Weight file load
    if (cfg.loadPending)
    {
        cfg.loadPending = false;

        const bool pathIsEmpty = (cfg.weightPath[0] == '\0');
        const std::string pathToLoad = pathIsEmpty ? openFileDialog()
                                                   : std::string(cfg.weightPath);

        if (!pathToLoad.empty())
        {
            if (pathIsEmpty)
            {
                std::strncpy(cfg.weightPath, pathToLoad.c_str(),
                             sizeof(cfg.weightPath) - 1);
                cfg.weightPath[sizeof(cfg.weightPath) - 1] = '\0';
            }

            auto loadedArch = m_network.loadWeights(pathToLoad.c_str());
            cfg.lastLoadOk        = !loadedArch.empty();
            cfg.lastLoadAttempted = true;

            if (cfg.lastLoadOk)
            {
                cfg.weightsLoaded = true;
                cfg.layerSizes.clear();
                cfg.layerActivations.clear();
                for (const auto& desc : loadedArch)
                {
                    cfg.layerSizes.push_back(desc.neuronCount);
                    cfg.layerActivations.push_back(desc.activation);
                }
                cfg.inputValues.resize(cfg.layerSizes.front(), 0.0f);
            }
        }
    }

    // Training
    if (train.randomizePending)
    {
        m_network.randomizeWeights();
        train.randomizePending = false;
        train.currentEpoch     = 0;
        train.currentLoss      = 0.0f;
    }

    if (train.active)
    {
        static const float xorInputs[4][2] = {
            {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
        };
        static const float xorTargets[4][1] = {
            {0.0f}, {1.0f}, {1.0f}, {0.0f}
        };

        if (m_network.getLayerCount() >= 3 &&
            m_network.getLayers().front().neuronCount == 2 &&
            m_network.getLayers().back().neuronCount  == 1)
        {
            float batchLoss = 0.0f;
            for (int e = 0; e < train.epochsPerFrame; ++e)
            {
                batchLoss = 0.0f;
                for (int i = 0; i < 4; ++i)
                {
                    m_network.setInputs(xorInputs[i], 2);
                    m_network.forwardPass();
                    batchLoss += m_network.backwardPass(xorTargets[i], 1);
                }
                m_network.applyGradients(train.learningRate);
                train.currentEpoch++;
            }
            train.currentLoss = batchLoss / 4.0f;
        }
        else
        {
            train.active = false;
        }
    }
    else if (cfg.inferenceMode)
    {
        m_network.setInputs(cfg.inputValues.data(),
                            static_cast<int>(cfg.inputValues.size()));

        if (prop.active)
        {
            prop.timer += dt;
            if (prop.timer >= prop.delay)
            {
                prop.timer -= prop.delay;
                m_network.forwardPassLayer(prop.layerIndex);
                prop.layerIndex++;
                if (prop.layerIndex >= m_network.getLayerCount())
                {
                    prop.active = false;
                    m_network.forwardPass();
                }
            }
        }
        else
        {
            m_network.forwardPass();
        }
    }
    else
    {
        m_network.tick(dt * m_state.params.animSpeed);
    }
}

// render — clear → 3D scene → ImGui overlay → done
void Application::render()
{
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(m_window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);

    glClearColor(0.04f, 0.04f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Compute highlight layer for the renderer (-1 = none)
    const int highlightLayer = (m_state.prop.active) ? m_state.prop.layerIndex : -1;

    // 3D scene
    m_renderer.draw(m_network, m_camera, fbW, fbH, m_state.params, highlightLayer);

    // ImGui overlay — strictly follows the required frame lifecycle
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    m_ui.draw(m_state, m_renderer, m_network, m_camera);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// initGLFW
bool Application::initGLFW(int w, int h, const char* title)
{
    glfwSetErrorCallback(cbError);
    if (!glfwInit())
    {
        std::fprintf(stderr, "Failed to initialise GLFW.\n");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // 4× MSAA

    m_window = glfwCreateWindow(w, h, title, nullptr, nullptr);
    if (!m_window)
    {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // vsync

    glfwSetWindowUserPointer  (m_window, this);
    glfwSetMouseButtonCallback(m_window, cbMouseButton);
    glfwSetCursorPosCallback  (m_window, cbCursorPos);
    glfwSetScrollCallback     (m_window, cbScroll);

    return true;
}

// initGLAD
bool Application::initGLAD()
{
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::fprintf(stderr, "Failed to initialise GLAD.\n");
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    return true;
}

// initImGui
bool Application::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style       = ImGui::GetStyle();
    style.WindowRounding    = 6.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding      = 4.0f;

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

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init(kGlslVersion);
    return true;
}

// initScene
bool Application::initScene()
{
    // Build the initial MLP: Input(4) → Hidden1(8) → Hidden2(6) → Output(3)
    m_network.build(
    {
        { 4, NeuralNetwork::ActivationType::Linear,  "Input"    },
        { 8, NeuralNetwork::ActivationType::Sigmoid, "Hidden 1" },
        { 6, NeuralNetwork::ActivationType::Sigmoid, "Hidden 2" },
        { 3, NeuralNetwork::ActivationType::Sigmoid, "Output"   }
    },
    /*layerSpacing=*/  3.0f,
    /*neuronSpacing=*/ 1.2f);

    // Sync AppState to match the built network
    auto& cfg = m_state.netConfig;
    cfg.layerSizes = {4, 8, 6, 3};
    cfg.layerActivations = {
        NeuralNetwork::ActivationType::Linear,
        NeuralNetwork::ActivationType::Sigmoid,
        NeuralNetwork::ActivationType::Sigmoid,
        NeuralNetwork::ActivationType::Sigmoid
    };
    cfg.inputValues.resize(4, 0.0f);

    if (!m_renderer.init())
    {
        std::fprintf(stderr, "Renderer::init() failed.\n");
        return false;
    }

    return true;
}

// rebuildNetwork
// Converts m_state.netConfig into a LayerDesc list and calls build().
void Application::rebuildNetwork()
{
    const auto& cfg = m_state.netConfig;

    std::vector<NeuralNetwork::LayerDesc> layers;
    layers.reserve(cfg.layerSizes.size());

    for (size_t i = 0; i < cfg.layerSizes.size(); ++i)
    {
        std::string label;
        if      (i == 0)                         label = "Input";
        else if (i == cfg.layerSizes.size() - 1) label = "Output";
        else                                      label = "Hidden " + std::to_string(i);

        layers.push_back({ cfg.layerSizes[i], cfg.layerActivations[i], std::move(label) });
    }

    m_network.build(layers, 3.0f, 1.2f);
}

// GLFW static callback trampolines
void Application::cbError(int code, const char* msg)
{
    std::fprintf(stderr, "[GLFW Error %d] %s\n", code, msg);
}

void Application::cbMouseButton(GLFWwindow* win, int button,
                                int action, int /*mods*/)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(win));
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(win, &x, &y);
    app->m_camera.onMouseButton(button, action, x, y);
}

void Application::cbCursorPos(GLFWwindow* win, double x, double y)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(win));
    app->m_camera.onMouseMove(x, y);
}

void Application::cbScroll(GLFWwindow* win, double /*dx*/, double dy)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(win));
    app->m_camera.onScroll(dy);
}
