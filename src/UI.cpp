// ImGui side-panel dashboard implementation.
// All application state is read/written via AppState; rendering details
// are accessed through the Renderer's read-only accessors.

#include "UI.h"
#include "Renderer.h"
#include "Camera.h"

#include <imgui.h>
#include <glad/glad.h>
#include <cstdio>

#define ICON_OK  "[OK]"
#define ICON_ERR "[!]"

// UI::draw
void UI::draw(AppState& state, const Renderer& renderer,
              NeuralNetwork& net, const Camera& cam)
{
    auto& params    = state.params;
    auto& netConfig = state.netConfig;
    auto& train     = state.train;
    auto& prop      = state.prop;

    ImGui::SetNextWindowPos (ImVec2(20.0f, 20.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(340.0f, 680.0f), ImGuiCond_Always);
    ImGui::Begin("Neural Net Visualizer", nullptr,
                 ImGuiWindowFlags_NoResize   |
                 ImGuiWindowFlags_NoMove     |
                 ImGuiWindowFlags_NoCollapse);

    ImGui::TextColored(ImVec4(0.55f, 0.38f, 1.0f, 1.0f),
                       "3D Neural Network Visualizer");
    ImGui::Separator();
    ImGui::Spacing();

    // Render Settings
    if (ImGui::CollapsingHeader("Render Settings"))
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

    // Architecture
    if (ImGui::CollapsingHeader("Architecture", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (netConfig.weightsLoaded)
        {
            ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.2f, 1.0f), "Locked by imported weights.");
            if (ImGui::Button("Unlock & Reset Architecture"))
            {
                netConfig.weightsLoaded     = false;
                netConfig.rebuildPending    = true;
                netConfig.lastLoadAttempted = false;
                netConfig.weightPath[0]     = '\0';
            }
            ImGui::Spacing();
        }

        ImGui::BeginDisabled(netConfig.weightsLoaded);

        ImGui::TextDisabled("Layers (excl. input and output):");
        ImGui::Spacing();

        if (ImGui::Button("Add Layer"))
        {
            netConfig.layerSizes.push_back(1);
            netConfig.layerActivations.push_back(NeuralNetwork::ActivationType::Sigmoid);
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove Layer") && netConfig.layerSizes.size() > 2)
        {
            netConfig.layerSizes.pop_back();
            netConfig.layerActivations.pop_back();
        }
        ImGui::Spacing();

        // Clamp sizes that might be out of range
        for (size_t i = 0; i < netConfig.layerSizes.size(); ++i)
        {
            if (netConfig.layerSizes[i] < 1)
                netConfig.layerSizes[i] = 1;
        }

        // Per-layer neuron count sliders and activation dropdowns
        for (int i = 0; i < static_cast<int>(netConfig.layerSizes.size()); ++i)
        {
            const bool isInput  = (i == 0);
            const bool isOutput = (i == static_cast<int>(netConfig.layerSizes.size()) - 1);

            char label[32];
            if (isInput)       std::snprintf(label, sizeof(label), "Input neurons");
            else if (isOutput) std::snprintf(label, sizeof(label), "Output neurons");
            else               std::snprintf(label, sizeof(label), "Hidden %d", i);

            ImVec4 col = isInput  ? ImVec4(0.20f, 0.80f, 0.38f, 1.0f)
                       : isOutput ? ImVec4(0.90f, 0.38f, 0.12f, 1.0f)
                       :            ImVec4(0.55f, 0.28f, 0.90f, 1.0f);

            ImGui::PushID(i);
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            ImGui::SetNextItemWidth(130.0f);
            ImGui::SliderInt("##size", &netConfig.layerSizes[i], 1, 64);
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::BeginDisabled(isInput);
            ImGui::SetNextItemWidth(70.0f);
            const char* actNames[] = { "Sigmoid", "ReLU", "Tanh", "Linear" };
            int currentAct = static_cast<int>(netConfig.layerActivations[i]);
            if (ImGui::Combo("##act", &currentAct, actNames, 4))
                netConfig.layerActivations[i] = static_cast<NeuralNetwork::ActivationType>(currentAct);
            ImGui::EndDisabled();

            ImGui::SameLine();
            ImGui::Text("%s", label);
            ImGui::PopID();
        }

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.20f, 0.45f, 0.20f, 0.90f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.65f, 0.28f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.35f, 0.80f, 0.35f, 1.00f));
        if (ImGui::Button("  Rebuild Network  ", ImVec2(-1.0f, 0.0f)))
        {
            netConfig.rebuildPending = true;
            netConfig.inferenceMode  = false;
            std::fill(netConfig.inputValues.begin(), netConfig.inputValues.end(), 0.0f);
        }
        ImGui::PopStyleColor(3);

        ImGui::EndDisabled();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Inference / Training
    if (ImGui::CollapsingHeader("Inference", ImGuiTreeNodeFlags_DefaultOpen))
    {
        int currentMode = train.active ? 2 : (netConfig.inferenceMode ? 1 : 0);
        ImGui::TextDisabled("Operating Mode:");
        if (ImGui::RadioButton("Animation##Mode", currentMode == 0)) {
            train.active = false;
            netConfig.inferenceMode = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Inference##Mode", currentMode == 1)) {
            train.active = false;
            netConfig.inferenceMode = true;
        }
        ImGui::SameLine();

        const bool canTrain = (net.getLayerCount() >= 3 &&
                               net.getLayers().front().neuronCount == 2 &&
                               net.getLayers().back().neuronCount  == 1);

        ImGui::BeginDisabled(!canTrain);
        if (ImGui::RadioButton("Training##Mode", currentMode == 2)) {
            train.active           = true;
            netConfig.inferenceMode = false;
            train.randomizePending  = true;
        }
        ImGui::EndDisabled();

        if (!canTrain && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("XOR Training requires exactly 2 inputs, 1 output,\n"
                              "and at least 1 hidden layer.\n"
                              "Adjust the architecture in the Network panel first.");
        }

        ImGui::Spacing();

        if (train.active)
        {
            ImGui::TextColored(ImVec4(0.90f, 0.38f, 0.12f, 1.0f), "Training Mode (XOR Dataset)");
            ImGui::Spacing();

            ImGui::SliderFloat("Learn Rate", &train.learningRate, 0.01f, 1.0f, "%.3f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderInt("Epochs/Frame", &train.epochsPerFrame, 1, 100);

            if (ImGui::Button("Randomize Weights", ImVec2(-1.0f, 0.0f)))
                train.randomizePending = true;

            ImGui::Spacing();
            ImGui::Text("Epoch: %d", train.currentEpoch);
            ImGui::Text("Loss:  %.6f", train.currentLoss);

            static float lossHist[100] = {0};
            static int   lossIdx       = 0;
            lossHist[lossIdx] = train.currentLoss;
            lossIdx = (lossIdx + 1) % 100;

            ImGui::PlotLines("##Loss", lossHist, 100, lossIdx, "MSE Loss",
                             0.0f, 0.3f, ImVec2(-1.0f, 60.0f));
            ImGui::Spacing();
            ImGui::Separator();
        }
        else if (netConfig.inferenceMode)
        {
            const int layerCount = net.getLayerCount();

            // Propagation Controls
            if (layerCount <= 2)
            {
                ImGui::BeginDisabled();
                ImGui::Button("Propagate Step-by-Step", ImVec2(-1.0f, 0.0f));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Animation requires at least one hidden layer.");
                ImGui::EndDisabled();
            }
            else if (!prop.active)
            {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.15f, 0.45f, 0.85f, 0.90f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.55f, 0.95f, 0.95f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.35f, 0.65f, 1.00f, 1.00f));
                if (ImGui::Button("Propagate Step-by-Step", ImVec2(-1.0f, 0.0f)))
                {
                    prop.active     = true;
                    prop.layerIndex = 1;
                    prop.timer      = 0.0f;
                    net.clearActivations(1);
                }
                ImGui::PopStyleColor(3);
            }
            else
            {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.85f, 0.25f, 0.25f, 0.90f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.35f, 0.35f, 0.95f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(1.00f, 0.45f, 0.45f, 1.00f));
                if (ImGui::Button("Stop Animation", ImVec2(-1.0f, 0.0f)))
                {
                    prop.active = false;
                    net.forwardPass();
                }
                ImGui::PopStyleColor(3);
            }

            ImGui::SliderFloat("Prop. Speed", &prop.delay, 0.05f, 1.5f, "%.2fs per layer");
            ImGui::Spacing();

            // Input sliders
            ImGui::BeginDisabled(prop.active);
            const int inputCount = netConfig.layerSizes[0];
            if (static_cast<int>(netConfig.inputValues.size()) != inputCount)
                netConfig.inputValues.resize(inputCount, 0.0f);

            ImGui::TextDisabled("Set input activations:");
            for (int i = 0; i < inputCount; ++i)
            {
                char label[16];
                std::snprintf(label, sizeof(label), "x%d", i);
                ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(0.20f, 0.80f, 0.38f, 1.0f));
                ImGui::SliderFloat(label, &netConfig.inputValues[i], 0.0f, 1.0f, "%.3f");
                ImGui::PopStyleColor();
            }
            ImGui::EndDisabled();
            ImGui::Spacing();

            // Output readout
            const auto& neurons = net.getNeurons();
            if (layerCount > 0)
            {
                const int      outLayer = layerCount - 1;
                const uint32_t outBase  = net.layerStart(outLayer);
                const int      outCount = net.getLayers()[outLayer].neuronCount;

                ImGui::TextDisabled("Output activations:");
                for (int i = 0; i < outCount; ++i)
                {
                    const float v = neurons[outBase + i].activation;
                    char label[16], overlay[16];
                    std::snprintf(label,   sizeof(label),   "y%d", i);
                    std::snprintf(overlay, sizeof(overlay), "%.3f", v);

                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                          ImVec4(0.90f, 0.38f, 0.12f, 1.0f));
                    ImGui::ProgressBar(v, ImVec2(160.0f, 0.0f), overlay);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", label);
                }
            }
        }
        else
        {
            ImGui::TextDisabled("Sin-wave oscillators drive the\n"
                                "input layer. Switch to Inference\n"
                                "to set inputs manually.");
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Network Stats
    if (ImGui::CollapsingHeader("Network Stats", ImGuiTreeNodeFlags_DefaultOpen))
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

            float avg = 0.0f, mx = 0.0f;
            for (int ni = 0; ni < count; ++ni)
            {
                const float a = neurons[base + ni].activation;
                avg += a;
                if (a > mx) mx = a;
            }
            avg /= static_cast<float>(count);

            const glm::vec3 lc = Renderer::layerColor(li, total);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(lc.r, lc.g, lc.b, 1.0f));
            ImGui::Text("%-10s", desc.label.c_str());
            ImGui::PopStyleColor();

            ImGui::SameLine();
            ImGui::TextDisabled("n=%-2d avg=%.2f max=%.2f", count, avg, mx);

            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(lc.r, lc.g, lc.b, 1.0f));
            ImGui::ProgressBar(avg, ImVec2(-1.0f, 4.0f), "");
            ImGui::PopStyleColor();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Activation History
    if (ImGui::CollapsingHeader("Activation History"))
    {
        const auto& history = renderer.getLayerHistories();
        const int   total   = net.getLayerCount();

        if (total == 0 || static_cast<int>(history.size()) != total)
        {
            ImGui::TextDisabled("No data yet.");
        }
        else
        {
            // Find global max activation for plotting (handles ReLU > 1.0)
            float globalMax = 0.0f;
            float linBuf[LayerHistory::kHistLen];
            for (int li = 0; li < total; ++li)
            {
                history[li].linearise(linBuf);
                for (int i = 0; i < history[li].count; ++i)
                    if (linBuf[i] > globalMax) globalMax = linBuf[i];
            }
            float scaleMax = std::max(1.0f, globalMax * 1.2f);

            for (int li = 0; li < total; ++li)
            {
                const LayerHistory& h = history[li];
                if (h.count == 0) continue;

                h.linearise(linBuf);

                const glm::vec3 lc = Renderer::layerColor(li, total);
                ImGui::PushStyleColor(ImGuiCol_PlotLines,
                                      ImVec4(lc.r, lc.g, lc.b, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_PlotLinesHovered,
                                      ImVec4(lc.r * 1.3f, lc.g * 1.3f, lc.b * 1.3f, 1.0f));

                char label[32];
                std::snprintf(label, sizeof(label), "%s##h%d",
                              net.getLayers()[li].label.c_str(), li);
                ImGui::PlotLines(label, linBuf, h.count, 0,
                                 nullptr, 0.0f, scaleMax, ImVec2(-1.0f, 38.0f));

                ImGui::PopStyleColor(2);
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Load Weights
    if (ImGui::CollapsingHeader("Load Weights"))
    {
        ImGui::TextDisabled("Import trained weights from a CSV file.");
        ImGui::Spacing();

        ImGui::SetNextItemWidth(-80.0f);
        ImGui::InputText("##wpath", netConfig.weightPath, sizeof(netConfig.weightPath));
        ImGui::SameLine();
        if (ImGui::Button("Browse"))
            netConfig.loadPending = true;

        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.15f, 0.35f, 0.65f, 0.90f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.50f, 0.90f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.28f, 0.60f, 1.00f, 1.00f));
        if (ImGui::Button("  Load from path  ", ImVec2(-1.0f, 0.0f)))
        {
            netConfig.loadPending       = true;
            netConfig.lastLoadAttempted = false;
        }
        ImGui::PopStyleColor(3);

        if (netConfig.lastLoadAttempted)
        {
            if (netConfig.lastLoadOk)
                ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.4f, 1.0f),
                                   ICON_OK " Weights loaded successfully.");
            else
                ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f),
                                   ICON_ERR " Load failed. Check path and layer sizes.");
        }

        ImGui::Spacing();
        if (ImGui::TreeNode("Python export snippet"))
        {
            const char* snippet =
                "import torch, csv\n"
                "m = torch.load('model.pth', map_location='cpu')\n"
                "rows = []\n"
                "state = m.state_dict()\n"
                "for k, v in state.items():\n"
                "    if 'weight' in k and len(v.shape) == 2:\n"
                "        out_f, in_f = v.shape\n"
                "        b_key = k.replace('weight', 'bias')\n"
                "        b = state[b_key].detach().cpu().numpy().tolist()\n"
                "        rows.append([in_f, out_f] + b + v.detach().cpu().numpy().flatten().tolist())\n"
                "with open('weights.csv','w',newline='') as f:\n"
                "    w = csv.writer(f)\n"
                "    for r in rows: w.writerow(r)";

            static double lastCopyTime = -100.0;
            if (ImGui::Button("Copy to Clipboard"))
            {
                ImGui::SetClipboardText(snippet);
                lastCopyTime = ImGui::GetTime();
            }

            if (ImGui::GetTime() - lastCopyTime < 2.0)
            {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.4f, 1.0f), ICON_OK " Copied!");
            }

            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.1f, 0.1f, 0.3f));
            ImGui::BeginChild("SnippetScroll", ImVec2(0, 175), true,
                              ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextUnformatted(snippet);
            ImGui::PopStyleColor();
            ImGui::EndChild();
            ImGui::PopStyleColor();

            ImGui::TreePop();
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
