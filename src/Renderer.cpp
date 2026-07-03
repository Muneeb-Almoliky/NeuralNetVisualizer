// Pure 3D render pipeline implementation.
// All shader source is embedded as raw string literals — no file I/O required.
// ImGui/UI logic lives in UI.cpp.

#include "Renderer.h"
#include "Camera.h"
#include "Network.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstdio>
#include <GLFW/glfw3.h>

// Embedded GLSL — Neuron (sphere) shaders
static const char* kNeuronVert = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform mat3 uNormalMat;

out vec3 vNormal;
out vec3 vFragPos;

void main()
{
    vec4 world  = uModel * vec4(aPos, 1.0);
    vFragPos    = world.xyz;
    vNormal     = uNormalMat * aNormal;
    gl_Position = uViewProj * world;
}
)glsl";

static const char* kNeuronFrag = R"glsl(
#version 330 core

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3  uCamPos;
uniform vec3  uBaseColor;
uniform float uActivation;
uniform float uGlowStrength;
uniform float uTime;
uniform int   uLayerIdx;
uniform int   uHighlightLayer;

out vec4 FragColor;

void main()
{
    const vec3 kLightPos = vec3(12.0, 18.0, 12.0);

    vec3 N = normalize(vNormal);
    vec3 L = normalize(kLightPos - vFragPos);
    vec3 V = normalize(uCamPos   - vFragPos);
    vec3 H = normalize(L + V);

    // Blend base layer color with activation value (Blue -> Red)
    vec3 lowColor  = vec3(0.15, 0.15, 0.90);
    vec3 highColor = vec3(0.90, 0.15, 0.15);
    vec3 activationColor = mix(lowColor, highColor, clamp(uActivation, 0.0, 1.0));
    vec3 baseColor = mix(uBaseColor, activationColor, 0.7);

    // Ambient
    vec3 ambient = baseColor * 0.22;

    // Lambertian diffuse
    float diff   = max(dot(N, L), 0.0);
    vec3  diffuse = baseColor * diff * 0.58;

    // Blinn-Phong specular
    float spec    = pow(max(dot(N, H), 0.0), 80.0);
    vec3  specular = vec3(0.90) * spec * 0.50;

    // Emissive glow — quadratic ramp so only highly-active neurons pop
    float clampedAct = min(uActivation, 2.5);
    float glow       = clampedAct * clampedAct;
    vec3  emissive   = vec3(0.25, 0.55, 1.00) * glow * uGlowStrength;

    if (uLayerIdx == uHighlightLayer) {
        emissive *= 1.5 + 0.5 * sin(uTime * 6.0);
    }

    FragColor = vec4(ambient + diffuse + specular + emissive, 1.0);
}
)glsl";

// Embedded GLSL — Synapse (screen-space quad) shaders
// Each synapse is a screen-aligned rectangle of fixed pixel width.
// Alpha (per-vertex) encodes |weight| / maxWeight — strong connections
// are opaque, weak ones fade out. Color encodes weight sign (blue/red).
static const char* kLineVert = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;       // this endpoint world position
layout(location = 1) in vec4 aColor;     // rgba: hue + weight-opacity
layout(location = 2) in vec3 aOtherPos;  // opposite endpoint world position
layout(location = 3) in float aSide;     // -1.0 or +1.0 (quad edge)

uniform mat4 uViewProj;
uniform vec2 uViewport;  // (width, height) in pixels

out vec4 vColor;

void main()
{
    vColor = aColor;

    // Fixed pixel half-width — visible but not thick enough to clutter
    const float kHalfWidth = 1.5;

    // Project both endpoints to clip space
    vec4 clip0 = uViewProj * vec4(aPos,      1.0);
    vec4 clip1 = uViewProj * vec4(aOtherPos, 1.0);

    // NDC direction, converted to screen pixels to find perpendicular
    vec2 ndc0      = clip0.xy / clip0.w;
    vec2 ndc1      = clip1.xy / clip1.w;
    vec2 dirScreen = normalize((ndc1 - ndc0) * uViewport);
    vec2 perp      = vec2(-dirScreen.y, dirScreen.x);

    // Push vertex left/right by kHalfWidth pixels (back to NDC)
    vec2 offset = perp * aSide * kHalfWidth / uViewport * 2.0;

    gl_Position = vec4(clip0.xy + offset * clip0.w, clip0.z, clip0.w);
}
)glsl";

static const char* kLineFrag = R"glsl(
#version 330 core

in  vec4 vColor;
out vec4 FragColor;

void main()
{
    FragColor = vColor;  // alpha already encoded per-vertex
}
)glsl";

// compileShader
GLuint Renderer::compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource (s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::fprintf(stderr, "[Shader compile]\n%s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

// linkProgram
GLuint Renderer::linkProgram(GLuint vert, GLuint frag)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram (prog);
    // Detach + delete stages — the program retains the linked binary
    glDetachShader(prog, vert); glDeleteShader(vert);
    glDetachShader(prog, frag); glDeleteShader(frag);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::fprintf(stderr, "[Shader link]\n%s\n", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// buildSphere — generates a UV sphere centred at origin with radius 1.
// Layout per vertex: [px, py, pz, nx, ny, nz] (position + normal)
void Renderer::buildSphere(int stacks, int slices,
                           std::vector<float>&  outVerts,
                           std::vector<GLuint>& outIdx)
{
    constexpr float kPi = 3.14159265358979f;

    for (int s = 0; s <= stacks; ++s)
    {
        // phi sweeps -PI/2 → +PI/2 (south pole to north pole)
        float phi  = kPi * (static_cast<float>(s) / static_cast<float>(stacks) - 0.5f);
        float cosP = std::cos(phi);
        float sinP = std::sin(phi);

        for (int sl = 0; sl <= slices; ++sl)
        {
            float theta = 2.0f * kPi * static_cast<float>(sl) / static_cast<float>(slices);
            float x = cosP * std::cos(theta);
            float y = sinP;
            float z = cosP * std::sin(theta);

            // Position + normal (unit sphere → normal == position)
            outVerts.insert(outVerts.end(), { x, y, z, x, y, z });
        }
    }

    for (int s = 0; s < stacks; ++s)
    {
        for (int sl = 0; sl < slices; ++sl)
        {
            GLuint p0 = static_cast<GLuint>( s      * (slices + 1) + sl    );
            GLuint p1 = static_cast<GLuint>( s      * (slices + 1) + sl + 1);
            GLuint p2 = static_cast<GLuint>((s + 1) * (slices + 1) + sl    );
            GLuint p3 = static_cast<GLuint>((s + 1) * (slices + 1) + sl + 1);
            outIdx.insert(outIdx.end(), { p0, p2, p1, p1, p2, p3 });
        }
    }
}

// layerColor — input=green, output=orange, hidden=violet→blue gradient
glm::vec3 Renderer::layerColor(int li, int total)
{
    if (total <= 1)      return glm::vec3(0.55f, 0.28f, 0.90f);
    if (li == 0)         return glm::vec3(0.20f, 0.80f, 0.38f); // green  – input
    if (li == total - 1) return glm::vec3(0.90f, 0.38f, 0.12f); // orange – output
    float t = static_cast<float>(li) / static_cast<float>(total - 1);
    return glm::mix(glm::vec3(0.55f, 0.20f, 0.90f),   // violet (near input)
                    glm::vec3(0.12f, 0.48f, 0.95f), t); // blue   (near output)
}

// init
bool Renderer::init()
{
    // Compile & link neuron shader
    {
        m_neuronShader.create();
        GLuint v = compileShader(GL_VERTEX_SHADER,   kNeuronVert);
        GLuint f = compileShader(GL_FRAGMENT_SHADER, kNeuronFrag);
        if (!v || !f) return false;
        m_neuronShader.id = linkProgram(v, f);
        if (!m_neuronShader.id) return false;
    }

    // Cache neuron uniform locations (avoids string look-up every draw call)
    m_nUni.model        = glGetUniformLocation(m_neuronShader.id, "uModel");
    m_nUni.viewProj     = glGetUniformLocation(m_neuronShader.id, "uViewProj");
    m_nUni.normalMat    = glGetUniformLocation(m_neuronShader.id, "uNormalMat");
    m_nUni.camPos       = glGetUniformLocation(m_neuronShader.id, "uCamPos");
    m_nUni.baseColor    = glGetUniformLocation(m_neuronShader.id, "uBaseColor");
    m_nUni.activation   = glGetUniformLocation(m_neuronShader.id, "uActivation");
    m_nUni.glowStrength = glGetUniformLocation(m_neuronShader.id, "uGlowStrength");
    m_nUni.time         = glGetUniformLocation(m_neuronShader.id, "uTime");
    m_nUni.layerIdx     = glGetUniformLocation(m_neuronShader.id, "uLayerIdx");
    m_nUni.highlightLayer = glGetUniformLocation(m_neuronShader.id, "uHighlightLayer");

    // Compile & link line shader
    {
        m_lineShader.create();
        GLuint v = compileShader(GL_VERTEX_SHADER,   kLineVert);
        GLuint f = compileShader(GL_FRAGMENT_SHADER, kLineFrag);
        if (!v || !f) return false;
        m_lineShader.id = linkProgram(v, f);
        if (!m_lineShader.id) return false;
    }

    m_lUni.viewProj  = glGetUniformLocation(m_lineShader.id, "uViewProj");
    m_lUni.viewport  = glGetUniformLocation(m_lineShader.id, "uViewport");

    // Build sphere geometry and upload once
    {
        m_sphereVAO.create();
        m_sphereVBO.create();
        m_sphereEBO.create();

        std::vector<float>  verts;
        std::vector<GLuint> idx;
        buildSphere(18, 28, verts, idx);
        m_sphereIndexCount = static_cast<GLsizei>(idx.size());

        glBindVertexArray(m_sphereVAO.id);

        glBindBuffer(GL_ARRAY_BUFFER, m_sphereVBO.id);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                     verts.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_sphereEBO.id);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(idx.size() * sizeof(GLuint)),
                     idx.data(), GL_STATIC_DRAW);

        constexpr GLsizei stride = 6 * sizeof(float);
        glEnableVertexAttribArray(0); // aPos
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1); // aNormal
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(3 * sizeof(float)));

        // State restore
        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Prepare synapse quad VAO (data streamed each frame)
    // Per-vertex layout: [pos(3), rgba(4), otherPos(3), side(1)] = 11 floats
    // 4 vertices + 6 indices per synapse (2 triangles forming a rectangle).
    {
        m_lineVAO.create();
        m_lineVBO.create();
        m_lineEBO.create();

        glBindVertexArray(m_lineVAO.id);
        glBindBuffer(GL_ARRAY_BUFFER, m_lineVBO.id);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_lineEBO.id);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

        constexpr GLsizei stride = 11 * sizeof(float);
        glEnableVertexAttribArray(0); // aPos
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1); // aColor (rgba)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glEnableVertexAttribArray(2); // aOtherPos
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(7 * sizeof(float)));
        glEnableVertexAttribArray(3); // aSide
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(10 * sizeof(float)));

        // State restore
        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    return true;
}

// draw — public entry point, computes VP matrix then dispatches passes
void Renderer::draw(const NeuralNetwork& net,
                    const Camera&        cam,
                    int fbW, int fbH,
                    const Params&        params,
                    int                  highlightLayer)
{
    const float     aspect   = (fbH > 0)
                               ? static_cast<float>(fbW) / static_cast<float>(fbH)
                               : 1.0f;
    const glm::mat4 view     = cam.getViewMatrix();
    const glm::mat4 proj     = glm::perspective(glm::radians(45.0f),
                                                aspect, 0.1f, 200.0f);
    const glm::mat4 viewProj = proj * view;
    const glm::vec3 camPos   = cam.getPosition();

    // Synapses drawn first so opaque neurons composite on top
    if (params.showSynapses) drawSynapses(net, viewProj, params);
    if (params.showNeurons)  drawNeurons (net, viewProj, camPos, params, highlightLayer);

    pushHistory(net);
}

// drawNeurons
void Renderer::drawNeurons(const NeuralNetwork& net,
                           const glm::mat4&     viewProj,
                           const glm::vec3&     camPos,
                           const Params&        params,
                           int                  highlightLayer)
{
    glUseProgram(m_neuronShader.id);

    // Per-frame constants (same for every neuron)
    glUniformMatrix4fv(m_nUni.viewProj,       1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform3fv      (m_nUni.camPos,         1, glm::value_ptr(camPos));
    glUniform1f       (m_nUni.glowStrength,   params.glowStrength);
    glUniform1f       (m_nUni.time,           static_cast<float>(glfwGetTime()));
    glUniform1i       (m_nUni.highlightLayer, highlightLayer);

    glBindVertexArray(m_sphereVAO.id);

    const int layerCount = net.getLayerCount();

    for (const Neuron& n : net.getNeurons())
    {
        // Build model matrix: translate to world position, scale by radius
        glm::mat4 model = glm::translate(glm::mat4(1.0f), n.position);
        model           = glm::scale(model, glm::vec3(params.neuronRadius));
        const glm::mat3 normalMat =
            glm::mat3(glm::transpose(glm::inverse(model)));

        const glm::vec3 base = layerColor(n.layerIndex, layerCount);

        glUniformMatrix4fv(m_nUni.model,      1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix3fv(m_nUni.normalMat,  1, GL_FALSE, glm::value_ptr(normalMat));
        glUniform3fv      (m_nUni.baseColor,  1, glm::value_ptr(base));
        glUniform1f       (m_nUni.activation, n.activation);
        glUniform1i       (m_nUni.layerIdx,   n.layerIndex);

        glDrawElements(GL_TRIANGLES, m_sphereIndexCount, GL_UNSIGNED_INT, nullptr);
    }

    // State restore
    glBindVertexArray(0);
    glUseProgram(0);
}

// drawSynapses
// Renders each synapse as a screen-space quad (2 triangles) of fixed pixel
// width. Alpha is per-vertex and encodes |weight| / maxWeight, so the
// strongest connection is always opaque and weak ones fade out. This gives
// clearly visible lines that scale correctly for both random and imported weights.
void Renderer::drawSynapses(const NeuralNetwork& net,
                            const glm::mat4&     viewProj,
                            const Params&        params)
{
    const auto& neurons  = net.getNeurons();
    const auto& synapses = net.getSynapses();
    if (synapses.empty()) return;

    // Normalise weights so the strongest synapse is always fully opaque
    float maxW = 0.0f;
    for (const Synapse& s : synapses)
        maxW = std::max(maxW, std::abs(s.weight));
    if (maxW < 1e-6f) maxW = 1.0f; // guard against zero-weight networks

    // Rebuild quad vertex + index buffers
    // Per-vertex layout: [pos(3), rgba(4), otherPos(3), side(1)] = 11 floats
    // Per-synapse: 4 vertices, 6 indices (2 triangles)
    m_lineVerts.clear();
    m_lineIdxBuf.clear();
    m_lineVerts  .reserve(synapses.size() * 4 * 11);
    m_lineIdxBuf .reserve(synapses.size() * 6);

    for (size_t si = 0; si < synapses.size(); ++si)
    {
        const Synapse& s   = synapses[si];
        const Neuron&  src = neurons[s.src];
        const Neuron&  dst = neurons[s.dst];

        // Hue: blue = positive weight, red = negative weight
        const float     t      = glm::clamp((s.weight + 1.0f) * 0.5f, 0.0f, 1.0f);
        const glm::vec3 posCol = glm::vec3(0.18f, 0.50f, 0.95f);
        const glm::vec3 negCol = glm::vec3(0.95f, 0.18f, 0.28f);
        const glm::vec3 rgb    = glm::mix(negCol, posCol, t);

        // Alpha: |weight| / maxW so scale is always relative to the network.
        // Multiplied by activation brightness and the global synapseAlpha slider.
        constexpr float kMinAlpha = 0.05f;
        const float normW  = std::abs(s.weight) / maxW;
        const float alpha  = glm::mix(kMinAlpha, 1.0f, normW)
                             * (0.35f + src.activation * 0.65f)
                             * params.synapseAlpha;

        const auto& sp = src.position;
        const auto& dp = dst.position;

        // 4 vertices per quad: v0(src,-1), v1(src,+1), v2(dst,+1), v3(dst,-1)
        // For dst vertices aOtherPos and aSide are swapped so the perpendicular
        // direction stays consistent across both endpoint pairs.
        auto emit = [&](glm::vec3 pos, glm::vec3 other, float side)
        {
            m_lineVerts.insert(m_lineVerts.end(), {
                pos.x,   pos.y,   pos.z,
                rgb.r,   rgb.g,   rgb.b,   alpha,
                other.x, other.y, other.z,
                side
            });
        };

        const GLuint base = static_cast<GLuint>(si * 4);
        emit(sp, dp, -1.0f); // v0
        emit(sp, dp, +1.0f); // v1
        emit(dp, sp, +1.0f); // v2
        emit(dp, sp, -1.0f); // v3

        m_lineIdxBuf.insert(m_lineIdxBuf.end(),
            { base+0, base+1, base+2,
              base+0, base+2, base+3 });
    }

    // Upload vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_lineVBO.id);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_lineVerts.size() * sizeof(float)),
                 m_lineVerts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Upload indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_lineEBO.id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_lineIdxBuf.size() * sizeof(GLuint)),
                 m_lineIdxBuf.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Query viewport for the perpendicular-offset calculation
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);

    // Draw with alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); // transparent quads don't write depth

    glUseProgram(m_lineShader.id);
    glUniformMatrix4fv(m_lUni.viewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform2f       (m_lUni.viewport,
                       static_cast<float>(vp[2]),
                       static_cast<float>(vp[3]));

    glBindVertexArray(m_lineVAO.id);
    glDrawElements(GL_TRIANGLES,
                   static_cast<GLsizei>(m_lineIdxBuf.size()),
                   GL_UNSIGNED_INT, nullptr);

    // State restore
    glBindVertexArray(0);
    glUseProgram(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

// pushHistory
// Called every frame from draw(). Lazily resizes m_history when the
// network is rebuilt, then samples the mean activation of each layer.
void Renderer::pushHistory(const NeuralNetwork& net)
{
    const int total = net.getLayerCount();
    if (total == 0) return;

    // Reset buffers whenever the architecture changes
    if (static_cast<int>(m_history.size()) != total)
        m_history.assign(total, LayerHistory{});

    const auto& neurons = net.getNeurons();
    const auto& layers  = net.getLayers();

    for (int li = 0; li < total; ++li)
    {
        const uint32_t base  = net.layerStart(li);
        const int      count = layers[li].neuronCount;
        float avg = 0.0f;
        for (int ni = 0; ni < count; ++ni)
            avg += neurons[base + ni].activation;
        m_history[li].push(avg / static_cast<float>(count));
    }
}

