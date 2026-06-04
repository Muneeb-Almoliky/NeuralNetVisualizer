// NeuralNetwork implementation.
// Lays out an MLP in 3D space along the Z-axis and runs a live
// forward-pass animation each frame.

#include "Network.h"

#include <algorithm> // std::clamp
#include <cmath>     // std::exp, std::sin
#include <fstream>
#include <sstream>
#include <random>

// Helpers
float NeuralNetwork::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static float applyActivation(float x, NeuralNetwork::ActivationType type)
{
    switch (type)
    {
        case NeuralNetwork::ActivationType::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
        case NeuralNetwork::ActivationType::ReLU:    return std::max(0.0f, x);
        case NeuralNetwork::ActivationType::Tanh:    return std::tanh(x);
        case NeuralNetwork::ActivationType::Linear:  return x;
    }
    return x;
}

// build()
void NeuralNetwork::build(const std::vector<LayerDesc>& layers,
                          float layerSpacing,
                          float neuronSpacing)
{
    m_layers   = layers;
    m_neurons .clear();
    m_synapses.clear();
    m_layerStart.clear();
    m_time = 0.0f;

    const int numLayers = static_cast<int>(layers.size());
    if (numLayers == 0) return;

    // Centre the entire network on the world origin along Z
    const float totalDepth = static_cast<float>(numLayers - 1) * layerSpacing;
    const float startZ     = -totalDepth * 0.5f;

    // Seeded Mersenne-Twister for reproducible layouts and weights
    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> weightDist(-1.0f,  1.0f);
    std::uniform_real_distribution<float> actDist   ( 0.0f,  1.0f);

    // Neuron positions
    for (int li = 0; li < numLayers; ++li)
    {
        m_layerStart.push_back(static_cast<uint32_t>(m_neurons.size()));

        const int   count  = layers[li].neuronCount;
        const float z      = startZ + static_cast<float>(li) * layerSpacing;
        const float height = static_cast<float>(count - 1) * neuronSpacing;

        for (int ni = 0; ni < count; ++ni)
        {
            Neuron n;
            n.layerIndex = li;
            n.activation = actDist(rng);
            n.bias       = 0.0f; // Default bias to zero on build
            // X=0 (all neurons centred); Y spread vertically; Z = layer depth
            n.position = glm::vec3(
                0.0f,
                static_cast<float>(ni) * neuronSpacing - height * 0.5f,
                z
            );
            m_neurons.push_back(n);
        }
    }

    // Synapses — fully-connected between every pair of adjacent layers
    for (int li = 0; li + 1 < numLayers; ++li)
    {
        const uint32_t srcBase  = m_layerStart[li];
        const uint32_t dstBase  = m_layerStart[li + 1];
        const int      srcCount = layers[li    ].neuronCount;
        const int      dstCount = layers[li + 1].neuronCount;

        for (int si = 0; si < srcCount; ++si)
        {
            for (int di = 0; di < dstCount; ++di)
            {
                Synapse s;
                s.src    = srcBase + static_cast<uint32_t>(si);
                s.dst    = dstBase + static_cast<uint32_t>(di);
                s.weight = weightDist(rng);
                m_synapses.push_back(s);
            }
        }
    }

    // Pre-size the scratch buffer used during tick()
    m_dstBuf.resize(m_neurons.size(), 0.0f);
}

// tick()
void NeuralNetwork::tick(float dt)
{
    m_time += dt;

    const int numLayers = getLayerCount();
    if (numLayers == 0) return;

    // Drive input layer with phase-shifted sinusoidal oscillators
    {
        const uint32_t base  = m_layerStart[0];
        const int      count = m_layers[0].neuronCount;

        for (int ni = 0; ni < count; ++ni)
        {
            const float phase = static_cast<float>(ni) * 0.75f;
            m_neurons[base + ni].activation =
                0.5f + 0.5f * std::sin(m_time * 1.3f + phase);
        }
    }

    // Forward pass through hidden and output layers
    // Accumulate weighted sums into m_dstBuf, then apply sigmoid in-place.
    for (int li = 1; li < numLayers; ++li)
    {
        const uint32_t dstBase  = m_layerStart[li];
        const int      dstCount = m_layers[li].neuronCount;

        // Zero the destination accumulators
        for (int di = 0; di < dstCount; ++di)
            m_dstBuf[dstBase + static_cast<uint32_t>(di)] = 0.0f;

        // Accumulate weighted inputs
        // Only synapses whose destination falls in this layer are processed.
        for (const Synapse& s : m_synapses)
        {
            const uint32_t dstEnd = dstBase + static_cast<uint32_t>(dstCount);
            if (s.dst >= dstBase && s.dst < dstEnd)
            {
                m_dstBuf[s.dst] +=
                    m_neurons[s.src].activation * s.weight;
            }
        }

        // Apply activation with bias and write back
        const auto actType = m_layers[li].activation;
        for (int di = 0; di < dstCount; ++di)
        {
            const uint32_t idx = dstBase + static_cast<uint32_t>(di);
            m_neurons[idx].activation = applyActivation(m_dstBuf[idx] + m_neurons[idx].bias, actType);
        }
    }
}

// setInputs
void NeuralNetwork::setInputs(const float* values, int count)
{
    if (getLayerCount() == 0) return;
    const uint32_t base = m_layerStart[0];
    const int      n    = m_layers[0].neuronCount;
    for (int i = 0; i < n; ++i)
        m_neurons[base + i].activation =
            (i < count) ? std::clamp(values[i], 0.0f, 1.0f) : 0.0f;
}

// forwardPass
// Deterministic weighted-sum + sigmoid propagation from layer 0 onward.
// Inputs must already be set via setInputs() or by directly writing
// m_neurons[layerStart(0)..].activation before this call.
void NeuralNetwork::forwardPass()
{
    const int numLayers = getLayerCount();
    for (int li = 1; li < numLayers; ++li)
    {
        const uint32_t dstBase  = m_layerStart[li];
        const int      dstCount = m_layers[li].neuronCount;

        // Zero accumulators
        for (int di = 0; di < dstCount; ++di)
            m_dstBuf[dstBase + static_cast<uint32_t>(di)] = 0.0f;

        // Accumulate weighted inputs from the previous layer
        for (const Synapse& s : m_synapses)
        {
            const uint32_t dstEnd = dstBase + static_cast<uint32_t>(dstCount);
            if (s.dst >= dstBase && s.dst < dstEnd)
                m_dstBuf[s.dst] += m_neurons[s.src].activation * s.weight;
        }

        // Apply activation function with bias
        const auto actType = m_layers[li].activation;
        for (int di = 0; di < dstCount; ++di)
        {
            const uint32_t idx = dstBase + static_cast<uint32_t>(di);
            m_neurons[idx].activation = applyActivation(m_dstBuf[idx] + m_neurons[idx].bias, actType);
        }
    }
}

// loadWeights
// Reads a CSV file (one line per layer transition).
// The first two columns of each line must be `srcCount` and `dstCount`.
// The remaining columns are the weight matrix for that transition flattened in
// PyTorch / row-major [dst_count][src_count] order.
// Rebuilds the network to match the discovered architecture.
std::vector<NeuralNetwork::LayerDesc> NeuralNetwork::loadWeights(const char* path)
{
    std::ifstream file(path);
    if (!file.is_open()) return {};

    std::vector<LayerDesc>          newLayers;
    std::vector<std::vector<float>> parsedWeights;
    std::vector<std::vector<float>> parsedBiases;
    std::string                     line;
    int                             expectedNextSrc = -1;
    int                             layerIndex      = 0;

    while (std::getline(file, line))
    {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string        token;
        std::vector<float> row;

        while (std::getline(ss, token, ','))
        {
            try         { row.push_back(std::stof(token)); }
            catch (...) { return {}; }
        }

        // We need at least srcCount, dstCount, and 1 weight
        if (row.size() < 3) return {};

        const int srcCount = static_cast<int>(row[0]);
        const int dstCount = static_cast<int>(row[1]);

        int remaining = static_cast<int>(row.size()) - 2;
        int expectedWeights = srcCount * dstCount;

        std::vector<float> biases;
        std::vector<float> weights;

        if (remaining != dstCount + expectedWeights) {
            return {}; // Invalid format length (must have biases)
        }

        biases.assign(row.begin() + 2, row.begin() + 2 + dstCount);
        weights.assign(row.begin() + 2 + dstCount, row.end());

        // Ensure layer sequence matches (e.g., transition 1 dst == transition 2 src)
        if (expectedNextSrc != -1 && srcCount != expectedNextSrc) return {};

        // For the first line, add the input layer (Input layer activation is ignored)
        if (newLayers.empty())
            newLayers.push_back({srcCount, ActivationType::Linear, "Input"});

        // Add the destination layer (Default to Sigmoid for backward compatibility)
        char label[32];
        std::snprintf(label, sizeof(label), "Layer %d", layerIndex + 1);
        newLayers.push_back({dstCount, ActivationType::Sigmoid, label});

        // Store the parsed biases and weights
        parsedBiases.emplace_back(std::move(biases));
        parsedWeights.emplace_back(std::move(weights));

        expectedNextSrc = dstCount;
        layerIndex++;
    }

    if (newLayers.empty()) return {};
    newLayers.back().label = "Output";

    // Apply the new architecture
    build(newLayers);

    // Apply the weights and biases
    int synapseBase = 0;
    for (size_t li = 0; li < parsedWeights.size(); ++li)
    {
        const int srcCount   = newLayers[li].neuronCount;
        const int dstCount   = newLayers[li + 1].neuronCount;
        const auto& w        = parsedWeights[li];
        const auto& b        = parsedBiases[li];
        const uint32_t dstBase = m_layerStart[li + 1];

        // Apply biases to destination layer
        for (int di = 0; di < dstCount; ++di)
            m_neurons[dstBase + di].bias = b[di];

        // Remap: CSV is [dst][src] (PyTorch row-major);
        // synapses are stored src-major: base + si*dstCount + di
        for (int si = 0; si < srcCount; ++si)
            for (int di = 0; di < dstCount; ++di)
                m_synapses[synapseBase + si * dstCount + di].weight =
                    w[di * srcCount + si];

        synapseBase += srcCount * dstCount;
    }

    return newLayers;
}
