#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Data structures
// A single neuron: its world-space position, which layer it belongs to,
// and its current activation.
struct Neuron
{
    glm::vec3 position   = glm::vec3(0.0f);
    float     activation = 0.0f;
    float     bias       = 0.0f;
    int       layerIndex = 0;
};

// A weighted directed edge between two neurons.
struct Synapse
{
    uint32_t src    = 0;
    uint32_t dst    = 0;
    float    weight = 0.0f;
};

// NeuralNetwork
// Holds the fully-connected MLP layout in 3D space and runs both an animated
// forward pass (tick) and a deterministic inference forward pass (forwardPass).
class NeuralNetwork
{
public:
    enum class ActivationType
    {
        Sigmoid,
        ReLU,
        Tanh,
        Linear
    };

    // Description of one layer, supplied to build().
    struct LayerDesc
    {
        int            neuronCount = 1;
        ActivationType activation  = ActivationType::Sigmoid;
        std::string    label;
    };

    // Lifecycle

    // (Re-)construct the network from a list of layer descriptors.
    // Positions neurons in 3D, creates all fully-connected synapses with seeded
    // random weights, and resets the simulation time.
    void build(const std::vector<LayerDesc>& layers,
               float layerSpacing  = 3.0f,
               float neuronSpacing = 1.2f);

    // Simulation

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

    // Run forward pass for a single specific layer (layerIndex > 0).
    void forwardPassLayer(int layerIndex);

    // Set activations and internal buffers of layers >= startLayer to 0.0.
    void clearActivations(int startLayer = 1);

    // Weight I/O

    // Load synaptic weights and architecture from a CSV file.
    // Expected format — one line per layer transition, comma-separated:
    //   Line 0: srcCount, dstCount, weights...
    //   Line 1: srcCount, dstCount, weights...
    // Rebuilds the network to match the discovered architecture.
    // Returns the discovered layers on success, or empty vector on failure.
    std::vector<LayerDesc> loadWeights(const char* path);

    // Accessors
    [[nodiscard]] const std::vector<Neuron>&    getNeurons()  const { return m_neurons;  }
    [[nodiscard]] const std::vector<Synapse>&   getSynapses() const { return m_synapses; }
    [[nodiscard]] const std::vector<LayerDesc>& getLayers()   const { return m_layers;   }

    [[nodiscard]] int getLayerCount()   const { return static_cast<int>(m_layers.size());   }
    [[nodiscard]] int getNeuronCount()  const { return static_cast<int>(m_neurons.size());  }
    [[nodiscard]] int getSynapseCount() const { return static_cast<int>(m_synapses.size()); }

    // Returns the index of the first neuron in layer li.
    [[nodiscard]] uint32_t layerStart(int li) const { return m_layerStart[li]; }

private:
    std::vector<LayerDesc> m_layers;
    std::vector<Neuron>    m_neurons;
    std::vector<Synapse>   m_synapses;
    std::vector<uint32_t>  m_layerStart;  // m_layerStart[li] = first neuron idx
    std::vector<float>     m_dstBuf;      // scratch accumulator for forward pass
    float                  m_time = 0.0f;

    static float sigmoid(float x);
};
