#pragma once

// Data model for a fully-connected Multi-Layer Perceptron.
// Stores neuron positions, real-time activations, synapse weights,
// and drives a live forward-pass animation via tick().

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Neuron
struct Neuron
{
    glm::vec3 position;   // world-space centre of this neuron's sphere
    float     activation; // current activation in [0, 1]
    int       layerIndex; // which layer this neuron belongs to
};

// Synapse
struct Synapse
{
    uint32_t src;    // index into NeuralNetwork::getNeurons()
    uint32_t dst;    // index into NeuralNetwork::getNeurons()
    float    weight; // signed connection strength in [-1, 1]
};

// NeuralNetwork
class NeuralNetwork
{
public:
    // Per-layer descriptor passed to build().
    struct LayerDesc
    {
        int         neuronCount;
        std::string label; // e.g. "Input", "Hidden 1", "Output"
    };

    // Construct and lay out a fully-connected MLP.
    //   layers        : ordered list of layer descriptors (input → output)
    //   layerSpacing  : Z-distance between consecutive layers
    //   neuronSpacing : Y-distance between consecutive neurons within a layer
    void build(const std::vector<LayerDesc>& layers,
               float layerSpacing  = 3.0f,
               float neuronSpacing = 1.2f);

    // Advance the live animation simulation by dt seconds.
    // Input neurons are driven by phase-shifted sinusoidal oscillators;
    // subsequent layers receive sigmoid of the weighted sum of the previous layer.
    void tick(float dt);

    // Set input-layer activations directly from an external array.
    // Values are clamped to [0,1]; entries beyond neuronCount are ignored;
    // missing entries default to 0.
    void setInputs(const float* values, int count);

    // Run one deterministic forward pass from the current input activations.
    // Call after setInputs() to propagate through all hidden and output layers.
    void forwardPass();

    // Accessors
    [[nodiscard]] const std::vector<Neuron>&    getNeurons()  const { return m_neurons;  }
    [[nodiscard]] const std::vector<Synapse>&   getSynapses() const { return m_synapses; }
    [[nodiscard]] const std::vector<LayerDesc>& getLayers()   const { return m_layers;   }

    [[nodiscard]] int      getLayerCount()       const { return static_cast<int>(m_layers.size()); }
    [[nodiscard]] int      getNeuronCount()       const { return static_cast<int>(m_neurons.size()); }
    [[nodiscard]] int      getSynapseCount()      const { return static_cast<int>(m_synapses.size()); }

    // Index of the first neuron that belongs to the given layer.
    [[nodiscard]] uint32_t layerStart(int layer) const { return m_layerStart[layer]; }

private:
    std::vector<Neuron>    m_neurons;
    std::vector<Synapse>   m_synapses;
    std::vector<LayerDesc> m_layers;
    std::vector<uint32_t>  m_layerStart; // first neuron index per layer
    std::vector<float>     m_dstBuf;     // scratch buffer for forward pass

    float m_time = 0.0f; // accumulated simulation time (seconds)

    // Logistic activation function.
    [[nodiscard]] static float sigmoid(float x);
};
