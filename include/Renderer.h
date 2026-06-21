#pragma once

// Pure 3D draw pipeline for the Neural Network Visualizer.
// Owns all OpenGL objects (shaders, VAOs, VBOs) via RAII wrappers.
// UI rendering is handled separately by the UI class.
// Application state (Params, NetworkConfig, etc.) lives in AppState.h.

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>
#include "Network.h"
#include "AppState.h"

// Forward declarations
class Camera;

// Lightweight RAII wrappers for raw OpenGL handles.
// Constructors do NOT call GL — call create() explicitly after GLAD is ready.
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

// LayerHistory
// Circular buffer recording mean layer activation per frame for history plots.
struct LayerHistory
{
    static constexpr int kHistLen = 128;
    float buf[kHistLen] = {};
    int   head          = 0;
    int   count         = 0;

    void push(float v)
    {
        buf[head] = v;
        head      = (head + 1) % kHistLen;
        if (count < kHistLen) ++count;
    }

    // Fills out[0..count-1] in chronological order (oldest first).
    void linearise(float* out) const
    {
        const int start = (count < kHistLen) ? 0 : head;
        for (int i = 0; i < count; ++i)
            out[i] = buf[(start + i) % kHistLen];
    }
};

// Renderer
// Single responsibility: draw the 3D scene.
// Receives Params and highlightLayer from Application each frame.
class Renderer
{
public:
    // Must be called once after a valid OpenGL 3.3 context is current.
    bool init();

    // Render the full 3D scene (neurons + synapses).
    // params        : rendering configuration (neuron radius, glow, alpha, visibility)
    // highlightLayer: layer index to pulse-highlight during propagation (-1 = none)
    void draw(const NeuralNetwork& net,
              const Camera&        cam,
              int fbW, int fbH,
              const Params&        params,
              int                  highlightLayer);

    // Read-only access to activation history for the UI to plot.
    [[nodiscard]] const std::vector<LayerHistory>& getLayerHistories() const
    {
        return m_history;
    }

    // Color scheme used by both Renderer (sphere tint) and UI (labels).
    static glm::vec3 layerColor(int layerIdx, int layerCount);

private:
    // Shader programs
    GlProgram m_neuronShader;
    GlProgram m_lineShader;

    // Cached uniform locations (queried once in init)
    struct NeuronUni
    {
        GLint model, viewProj, normalMat;
        GLint camPos, baseColor, activation, glowStrength;
        GLint time, layerIdx, highlightLayer;
    } m_nUni{};

    struct LineUni
    {
        GLint viewProj, alpha, viewport;
    } m_lUni{};

    // Sphere geometry (static — uploaded once)
    GlVAO    m_sphereVAO;
    GlBuffer m_sphereVBO;
    GlBuffer m_sphereEBO;
    GLsizei  m_sphereIndexCount = 0;

    // Activation history (one LayerHistory per layer)
    std::vector<LayerHistory> m_history;

    // Synapse quad geometry (dynamic — rebuilt each frame)
    GlVAO                m_lineVAO;
    GlBuffer             m_lineVBO;
    GlBuffer             m_lineEBO;
    std::vector<float>   m_lineVerts;   // 11 floats × 4 verts per synapse
    std::vector<GLuint>  m_lineIdxBuf;  // 6 indices per synapse

    // Internal draw passes
    void drawNeurons (const NeuralNetwork& net,
                      const glm::mat4& viewProj,
                      const glm::vec3& camPos,
                      const Params&    params,
                      int              highlightLayer);

    void drawSynapses(const NeuralNetwork& net,
                      const glm::mat4& viewProj,
                      const Params&    params);

    void pushHistory (const NeuralNetwork& net);

    // Static helpers
    static GLuint compileShader(GLenum type, const char* src);
    static GLuint linkProgram  (GLuint vert, GLuint frag);
    static void   buildSphere  (int stacks, int slices,
                                std::vector<float>&  outVerts,
                                std::vector<GLuint>& outIdx);
};
