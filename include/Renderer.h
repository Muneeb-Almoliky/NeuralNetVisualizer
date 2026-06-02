#pragma once

// 3D draw pipeline for the Neural Network Visualizer.
// Owns all OpenGL objects (shaders, VAOs, VBOs) via RAII wrappers and
// exposes two public entry points: draw() for the 3D scene and
// drawImGui() for the side-panel dashboard.

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

// Forward declarations — avoids pulling heavy headers into every TU
// that includes Renderer.h.
class Camera;
class NeuralNetwork;

// Lightweight RAII wrappers for raw OpenGL handles.
// Constructors do NOT call GL — call create() explicitly after GLAD is ready.
// This prevents accidental GL calls before the context exists when these
// structs are used as class members.
struct GlProgram
{
    GLuint id = 0;
    GlProgram()                            = default;
    void   create()                        { id = glCreateProgram(); }
    ~GlProgram()                           { if (id) glDeleteProgram(id); }
    GlProgram(const GlProgram&)            = delete;
    GlProgram& operator=(const GlProgram&) = delete;
};

struct GlVAO
{
    GLuint id = 0;
    GlVAO()                        = default;
    void   create()                { glGenVertexArrays(1, &id); }
    ~GlVAO()                       { if (id) glDeleteVertexArrays(1, &id); }
    GlVAO(const GlVAO&)            = delete;
    GlVAO& operator=(const GlVAO&) = delete;
};

struct GlBuffer
{
    GLuint id = 0;
    GlBuffer()                           = default;
    void   create()                      { glGenBuffers(1, &id); }
    ~GlBuffer()                          { if (id) glDeleteBuffers(1, &id); }
    GlBuffer(const GlBuffer&)            = delete;
    GlBuffer& operator=(const GlBuffer&) = delete;
};

// Renderer
class Renderer
{
public:
    // Runtime-tunable parameters (written to by ImGui sliders)
    struct Params
    {
        float neuronRadius = 0.28f;  // sphere scale
        float glowStrength = 2.50f;  // emissive multiplier for high-activation nodes
        float animSpeed    = 1.00f;  // dt multiplier fed to NeuralNetwork::tick()
        float synapseAlpha = 0.35f;  // transparency of synapse lines
        bool  showNeurons  = true;
        bool  showSynapses = true;
    } params;

    // Must be called once, after a valid OpenGL 3.3 context is current.
    bool init();

    // Render the full 3D scene (neurons + synapses).
    void draw   (const NeuralNetwork& net, const Camera& cam, int fbW, int fbH);

    // Render the ImGui side panel (controls + live stats).
    // Accepts Camera so it can display the live eye-position readout.
    void drawImGui(const NeuralNetwork& net, const Camera& cam);

private:
    // Shader programs
    GlProgram m_neuronShader;
    GlProgram m_lineShader;

    // Cached uniform locations (queried once in init)
    struct NeuronUni
    {
        GLint model, viewProj, normalMat;
        GLint camPos, baseColor, activation, glowStrength;
    } m_nUni{};

    struct LineUni
    {
        GLint viewProj, alpha;
    } m_lUni{};

    // Sphere geometry (static — uploaded once)
    GlVAO    m_sphereVAO;
    GlBuffer m_sphereVBO;
    GlBuffer m_sphereEBO;
    GLsizei  m_sphereIndexCount = 0;

    // Synapse line geometry (dynamic — rebuilt each frame)
    GlVAO              m_lineVAO;
    GlBuffer           m_lineVBO;
    std::vector<float> m_lineVerts; // [x,y,z,r,g,b] × 2 per synapse

    // Internal draw passes
    void drawNeurons (const NeuralNetwork& net,
                      const glm::mat4& viewProj,
                      const glm::vec3& camPos);
    void drawSynapses(const NeuralNetwork& net,
                      const glm::mat4& viewProj);

    // Static helpers
    static GLuint     compileShader(GLenum type, const char* src);
    static GLuint     linkProgram  (GLuint vert, GLuint frag);
    static void       buildSphere  (int stacks, int slices,
                                    std::vector<float>&  outVerts,
                                    std::vector<GLuint>& outIdx);
    static glm::vec3  layerColor   (int layerIdx, int layerCount);
};
