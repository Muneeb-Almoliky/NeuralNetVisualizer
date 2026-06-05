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

    // Pre-size training buffers
    m_deltas.assign(m_neurons.size(), 0.0f);
    m_biasGradients.assign(m_neurons.size(), 0.0f);
    m_weightGradients.assign(m_synapses.size(), 0.0f);
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
void NeuralNetwork::forwardPassLayer(int li)
{
    const int numLayers = getLayerCount();
    if (li < 1 || li >= numLayers) return;

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

void NeuralNetwork::forwardPass()
{
    const int numLayers = getLayerCount();
    for (int li = 1; li < numLayers; ++li)
    {
        forwardPassLayer(li);
    }
}

void NeuralNetwork::clearActivations(int startLayer)
{
    const int numLayers = getLayerCount();
    if (startLayer < 0 || startLayer >= numLayers) return;

    const uint32_t base = m_layerStart[startLayer];
    const uint32_t count = static_cast<uint32_t>(m_neurons.size()) - base;

    for (uint32_t i = 0; i < count; ++i)
    {
        m_neurons[base + i].activation = 0.0f;
        m_dstBuf[base + i] = 0.0f;
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

// Training Math
float NeuralNetwork::applyActivationDerivative(float y, ActivationType type)
{
    switch (type)
    {
    case ActivationType::Sigmoid: return y * (1.0f - y);
    case ActivationType::ReLU:    return (y > 0.0f) ? 1.0f : 0.0f;
    case ActivationType::Tanh:    return 1.0f - y * y;
    case ActivationType::Linear:  return 1.0f;
    default:                      return 1.0f;
    }
}

void NeuralNetwork::randomizeWeights()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> biasDist(-0.1f, 0.1f);
    
    for (auto& s : m_synapses)
        s.weight = dist(rng);
        
    for (int li = 1; li < getLayerCount(); ++li)
    {
        uint32_t base = layerStart(li);
        for (int i = 0; i < m_layers[li].neuronCount; ++i)
            m_neurons[base + i].bias = biasDist(rng);
    }
}

float NeuralNetwork::backwardPass(const float* targets, int targetCount)
{
    int numLayers = getLayerCount();
    if (numLayers < 2) return 0.0f;

    std::fill(m_deltas.begin(), m_deltas.end(), 0.0f);

    float loss = 0.0f;

    // 1. Output Layer Deltas (MSE derivative)
    int outLayer = numLayers - 1;
    uint32_t outBase = m_layerStart[outLayer];
    int outCount = m_layers[outLayer].neuronCount;
    auto outActType = m_layers[outLayer].activation;

    for (int i = 0; i < outCount; ++i)
    {
        uint32_t idx = outBase + i;
        float y = m_neurons[idx].activation;
        float target = (i < targetCount) ? targets[i] : 0.0f;
        float err = y - target;
        loss += err * err;
        m_deltas[idx] = err * applyActivationDerivative(y, outActType);
    }
    loss /= static_cast<float>(outCount);

    // 2. Backpropagate deltas through synapses
    for (int si = static_cast<int>(m_synapses.size()) - 1; si >= 0; --si)
    {
        const Synapse& s = m_synapses[si];
        m_deltas[s.src] += s.weight * m_deltas[s.dst];
    }

    // 3. Apply derivatives to hidden deltas
    for (int li = outLayer - 1; li >= 1; --li)
    {
        uint32_t base = m_layerStart[li];
        int count = m_layers[li].neuronCount;
        auto actType = m_layers[li].activation;
        for (int i = 0; i < count; ++i)
        {
            uint32_t idx = base + i;
            float y = m_neurons[idx].activation;
            m_deltas[idx] *= applyActivationDerivative(y, actType);
        }
    }

    // 4. Accumulate gradients
    for (size_t si = 0; si < m_synapses.size(); ++si)
    {
        const Synapse& s = m_synapses[si];
        m_weightGradients[si] += m_neurons[s.src].activation * m_deltas[s.dst];
    }
    
    for (int li = 1; li < numLayers; ++li)
    {
        uint32_t base = m_layerStart[li];
        int count = m_layers[li].neuronCount;
        for (int i = 0; i < count; ++i)
        {
            uint32_t idx = base + i;
            m_biasGradients[idx] += m_deltas[idx];
        }
    }

    return loss;
}

void NeuralNetwork::applyGradients(float learningRate)
{
    for (size_t si = 0; si < m_synapses.size(); ++si)
    {
        m_synapses[si].weight -= learningRate * m_weightGradients[si];
        m_weightGradients[si] = 0.0f;
    }
    for (size_t ni = 0; ni < m_neurons.size(); ++ni)
    {
        m_neurons[ni].bias -= learningRate * m_biasGradients[ni];
        m_biasGradients[ni] = 0.0f;
    }
}
