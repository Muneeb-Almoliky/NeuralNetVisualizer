// Entry point for the Neural Network Visualizer.

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>   
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>

// Constants
static constexpr int   kInitWidth   = 1280;
static constexpr int   kInitHeight  = 720;
static constexpr char  kWindowTitle[] = "Neural Net Visualizer";
static constexpr char  kGlslVersion[] = "#version 330 core";

// GLFW error callback
static void glfwErrorCallback(int error, const char* description)
{
    std::fprintf(stderr, "[GLFW Error %d] %s\n", error, description);
}

// main
int main()
{
    // GLFW initialisation
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        std::fprintf(stderr, "Failed to initialise GLFW.\n");
        return EXIT_FAILURE;
    }

    // Request OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x MSAA

    GLFWwindow* window = glfwCreateWindow(kInitWidth, kInitHeight, kWindowTitle, nullptr, nullptr);
    if (!window)
    {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // GLAD – load OpenGL function pointers
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::fprintf(stderr, "Failed to initialise GLAD.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    // Dear ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Dark, slightly customised theme
    ImGui::StyleColorsDark();
    {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding    = 6.0f;
        style.FrameRounding     = 4.0f;
        style.PopupRounding     = 4.0f;
        style.ScrollbarRounding = 6.0f;
        style.GrabRounding      = 4.0f;

        // Accent palette – deep violet / electric blue
        ImVec4* colors = style.Colors;
        colors[ImGuiCol_WindowBg]        = ImVec4(0.07f, 0.07f, 0.10f, 0.95f);
        colors[ImGuiCol_Header]          = ImVec4(0.26f, 0.16f, 0.55f, 0.80f);
        colors[ImGuiCol_HeaderHovered]   = ImVec4(0.36f, 0.22f, 0.75f, 0.85f);
        colors[ImGuiCol_HeaderActive]    = ImVec4(0.40f, 0.26f, 0.85f, 1.00f);
        colors[ImGuiCol_Button]          = ImVec4(0.20f, 0.12f, 0.50f, 0.85f);
        colors[ImGuiCol_ButtonHovered]   = ImVec4(0.30f, 0.18f, 0.70f, 0.90f);
        colors[ImGuiCol_ButtonActive]    = ImVec4(0.38f, 0.24f, 0.85f, 1.00f);
        colors[ImGuiCol_FrameBg]         = ImVec4(0.12f, 0.10f, 0.20f, 0.80f);
        colors[ImGuiCol_FrameBgHovered]  = ImVec4(0.20f, 0.16f, 0.35f, 0.80f);
        colors[ImGuiCol_FrameBgActive]   = ImVec4(0.25f, 0.20f, 0.45f, 1.00f);
        colors[ImGuiCol_TitleBg]         = ImVec4(0.05f, 0.05f, 0.09f, 1.00f);
        colors[ImGuiCol_TitleBgActive]   = ImVec4(0.18f, 0.10f, 0.42f, 1.00f);
        colors[ImGuiCol_SliderGrab]      = ImVec4(0.45f, 0.28f, 0.90f, 1.00f);
        colors[ImGuiCol_SliderGrabActive]= ImVec4(0.55f, 0.38f, 1.00f, 1.00f);
        colors[ImGuiCol_CheckMark]       = ImVec4(0.55f, 0.38f, 1.00f, 1.00f);
        colors[ImGuiCol_Tab]             = ImVec4(0.12f, 0.08f, 0.25f, 0.90f);
        colors[ImGuiCol_TabHovered]      = ImVec4(0.30f, 0.18f, 0.65f, 0.90f);
        colors[ImGuiCol_TabActive]       = ImVec4(0.22f, 0.13f, 0.50f, 1.00f);
        colors[ImGuiCol_SeparatorHovered]= ImVec4(0.45f, 0.28f, 0.90f, 0.80f);
        colors[ImGuiCol_SeparatorActive] = ImVec4(0.55f, 0.38f, 1.00f, 1.00f);
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(kGlslVersion);

    // OpenGL global state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Framebuffer size (handles window resize correctly)
        int fbWidth  = 0;
        int fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);

        // Clear the screen with a very dark navy
        glClearColor(0.04f, 0.04f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // "Hello Mesh" info panel
        ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(340.0f, 0.0f), ImGuiCond_Always);
        ImGui::Begin("Neural Net Visualizer", nullptr,
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove   |
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

        // Display basic renderer info
        ImGui::TextDisabled("Renderer : %s", glGetString(GL_RENDERER));
        ImGui::TextDisabled("GL ver.  : %s", glGetString(GL_VERSION));

        ImGui::End();

        // Render ImGui draw data
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
