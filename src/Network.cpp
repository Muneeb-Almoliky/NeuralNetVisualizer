// NeuralNetwork implementation.
// Lays out an MLP in 3D space along the Z-axis and runs a live
// forward-pass animation each frame.

#include "Network.h"

#include <algorithm> // std::clamp
#include <cmath>     // std::exp, std::sin
#include <random>

// Helpers
float NeuralNetwork::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
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

        // Apply sigmoid and write back
        for (int di = 0; di < dstCount; ++di)
        {
            const uint32_t idx = dstBase + static_cast<uint32_t>(di);
            m_neurons[idx].activation = sigmoid(m_dstBuf[idx]);
        }
    }
}
