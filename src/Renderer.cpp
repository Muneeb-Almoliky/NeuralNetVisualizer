// Full 3D render pipeline implementation.
// All shader source is embedded as raw string literals — no file I/O required.

#include "Renderer.h"
#include "Camera.h"
#include "Network.h"

#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstdio>

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

out vec4 FragColor;

void main()
{
    const vec3 kLightPos = vec3(12.0, 18.0, 12.0);

    vec3 N = normalize(vNormal);
    vec3 L = normalize(kLightPos - vFragPos);
    vec3 V = normalize(uCamPos   - vFragPos);
    vec3 H = normalize(L + V);

    // Ambient
    vec3 ambient = uBaseColor * 0.22;

    // Lambertian diffuse
    float diff   = max(dot(N, L), 0.0);
    vec3  diffuse = uBaseColor * diff * 0.58;

    // Blinn-Phong specular
    float spec    = pow(max(dot(N, H), 0.0), 80.0);
    vec3  specular = vec3(0.90) * spec * 0.50;

    // Emissive glow — quadratic ramp so only highly-active neurons pop
    float glow    = uActivation * uActivation;
    vec3  emissive = vec3(0.25, 0.55, 1.00) * glow * uGlowStrength;

    FragColor = vec4(ambient + diffuse + specular + emissive, 1.0);
}
)glsl";

// Embedded GLSL — Synapse (line) shaders
static const char* kLineVert = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uViewProj;

out vec3 vColor;

void main()
{
    vColor      = aColor;
    gl_Position = uViewProj * vec4(aPos, 1.0);
}
)glsl";

static const char* kLineFrag = R"glsl(
#version 330 core

in  vec3  vColor;
uniform float uAlpha;
out vec4  FragColor;

void main()
{
    FragColor = vec4(vColor, uAlpha);
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

    // Compile & link line shader
    {
        m_lineShader.create();
        GLuint v = compileShader(GL_VERTEX_SHADER,   kLineVert);
        GLuint f = compileShader(GL_FRAGMENT_SHADER, kLineFrag);
        if (!v || !f) return false;
        m_lineShader.id = linkProgram(v, f);
        if (!m_lineShader.id) return false;
    }

    m_lUni.viewProj = glGetUniformLocation(m_lineShader.id, "uViewProj");
    m_lUni.alpha    = glGetUniformLocation(m_lineShader.id, "uAlpha");

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

    // Prepare line VAO (data streamed each frame)
    {
        m_lineVAO.create();
        m_lineVBO.create();

        glBindVertexArray(m_lineVAO.id);
        glBindBuffer(GL_ARRAY_BUFFER, m_lineVBO.id);
        // Empty allocation — size grows dynamically in drawSynapses()
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

        constexpr GLsizei stride = 6 * sizeof(float);
        glEnableVertexAttribArray(0); // aPos
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1); // aColor
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                              reinterpret_cast<void*>(3 * sizeof(float)));

        // State restore
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    return true;
}

// draw — public entry point, computes VP matrix then dispatches passes
void Renderer::draw(const NeuralNetwork& net,
                    const Camera&        cam,
                    int fbW, int fbH)
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
    if (params.showSynapses) drawSynapses(net, viewProj);
    if (params.showNeurons)  drawNeurons (net, viewProj, camPos);
}

// drawNeurons
void Renderer::drawNeurons(const NeuralNetwork& net,
                           const glm::mat4&     viewProj,
                           const glm::vec3&     camPos)
{
    glUseProgram(m_neuronShader.id);

    // Per-frame constants (same for every neuron)
    glUniformMatrix4fv(m_nUni.viewProj,     1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform3fv      (m_nUni.camPos,       1, glm::value_ptr(camPos));
    glUniform1f       (m_nUni.glowStrength, params.glowStrength);

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

        glDrawElements(GL_TRIANGLES, m_sphereIndexCount, GL_UNSIGNED_INT, nullptr);
    }

    // State restore
    glBindVertexArray(0);
    glUseProgram(0);
}

// drawSynapses
void Renderer::drawSynapses(const NeuralNetwork& net,
                            const glm::mat4&     viewProj)
{
    const auto& neurons  = net.getNeurons();
    const auto& synapses = net.getSynapses();
    if (synapses.empty()) return;

    // Rebuild line vertex buffer
    m_lineVerts.clear();
    m_lineVerts.reserve(synapses.size() * 12); // 2 verts × [x,y,z,r,g,b]

    for (const Synapse& s : synapses)
    {
        const Neuron& src = neurons[s.src];
        const Neuron& dst = neurons[s.dst];

        // Weight → hue: positive weight = blue, negative = red
        const float      t      = (s.weight + 1.0f) * 0.5f; // [-1,1] → [0,1]
        const glm::vec3  posCol = glm::vec3(0.18f, 0.50f, 0.95f);
        const glm::vec3  negCol = glm::vec3(0.95f, 0.18f, 0.28f);
        // Brightness scaled by source activation so live signals stand out
        const glm::vec3  col    =
            glm::mix(negCol, posCol, t) * (0.35f + src.activation * 0.65f);

        // Source vertex
        m_lineVerts.insert(m_lineVerts.end(),
            { src.position.x, src.position.y, src.position.z,
              col.r, col.g, col.b });
        // Destination vertex
        m_lineVerts.insert(m_lineVerts.end(),
            { dst.position.x, dst.position.y, dst.position.z,
              col.r, col.g, col.b });
    }

    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, m_lineVBO.id);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_lineVerts.size() * sizeof(float)),
                 m_lineVerts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Draw with alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); // transparent lines don't write depth

    glUseProgram(m_lineShader.id);
    glUniformMatrix4fv(m_lUni.viewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform1f       (m_lUni.alpha,    params.synapseAlpha);

    glBindVertexArray(m_lineVAO.id);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(synapses.size() * 2));

    // State restore
    glBindVertexArray(0);
    glUseProgram(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

// drawImGui
void Renderer::drawImGui(const NeuralNetwork& net, const Camera& cam)
{
    ImGui::SetNextWindowPos (ImVec2(20.0f, 20.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(315.0f, 0.0f), ImGuiCond_Always);
    ImGui::Begin("Neural Net Visualizer", nullptr,
                 ImGuiWindowFlags_NoResize   |
                 ImGuiWindowFlags_NoMove     |
                 ImGuiWindowFlags_NoCollapse);

    ImGui::TextColored(ImVec4(0.55f, 0.38f, 1.0f, 1.0f),
                       "3D Neural Network Visualizer");
    ImGui::Separator();
    ImGui::Spacing();

    // Render Settings
    if (ImGui::CollapsingHeader("Render Settings",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Neuron Radius",  &params.neuronRadius, 0.10f, 0.70f, "%.2f");
        ImGui::SliderFloat("Glow Strength",  &params.glowStrength, 0.00f, 6.00f, "%.2f");
        ImGui::SliderFloat("Synapse Alpha",  &params.synapseAlpha, 0.00f, 1.00f, "%.2f");
        ImGui::SliderFloat("Anim Speed",     &params.animSpeed,    0.00f, 4.00f, "%.2fx");
        ImGui::Spacing();
        ImGui::Checkbox("Neurons",  &params.showNeurons);
        ImGui::SameLine();
        ImGui::Checkbox("Synapses", &params.showSynapses);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Network Stats
    if (ImGui::CollapsingHeader("Network Stats",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TextDisabled("%d layers  |  %d neurons  |  %d synapses",
                            net.getLayerCount(),
                            net.getNeuronCount(),
                            net.getSynapseCount());
        ImGui::Spacing();

        const auto& layers  = net.getLayers();
        const auto& neurons = net.getNeurons();
        const int   total   = net.getLayerCount();

        for (int li = 0; li < total; ++li)
        {
            const auto&    desc  = layers[li];
            const uint32_t base  = net.layerStart(li);
            const int      count = desc.neuronCount;

            float avg = 0.0f;
            for (int ni = 0; ni < count; ++ni)
                avg += neurons[base + ni].activation;
            avg /= static_cast<float>(count);

            ImGui::TextDisabled("  %-10s  n=%-2d  avg=%.2f",
                                desc.label.c_str(), count, avg);
            ImGui::SameLine();

            const glm::vec3 lc = layerColor(li, total);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                  ImVec4(lc.r, lc.g, lc.b, 1.0f));
            ImGui::ProgressBar(avg, ImVec2(-1.0f, 6.0f), "");
            ImGui::PopStyleColor();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Camera
    if (ImGui::CollapsingHeader("Camera"))
    {
        const glm::vec3 eye = cam.getPosition();
        ImGui::TextDisabled("Eye  (%.2f, %.2f, %.2f)", eye.x, eye.y, eye.z);
        ImGui::Spacing();
        ImGui::TextDisabled("LMB drag -> orbit   |   Scroll -> zoom");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Renderer : %s", glGetString(GL_RENDERER));
    ImGui::TextDisabled("GL ver.  : %s", glGetString(GL_VERSION));

    ImGui::End();
}
