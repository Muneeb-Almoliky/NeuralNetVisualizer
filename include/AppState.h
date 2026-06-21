#pragma once

#include <vector>
#include <string>
#include "Network.h" // For NeuralNetwork::ActivationType

// Application State
// Centralized state structs previously owned by the Renderer.
// These are now owned by Application and passed to UI and Renderer.

// Runtime-tunable parameters (written to by ImGui sliders)
struct Params
{
    float neuronRadius = 0.28f;  // sphere scale
    float glowStrength = 2.50f;  // emissive multiplier for high-activation nodes
    float animSpeed    = 1.00f;  // dt multiplier fed to NeuralNetwork::tick()
    float synapseAlpha = 0.35f;  // transparency of synapse lines
    bool  showNeurons  = true;
    bool  showSynapses = true;
};

// Network configuration (written by ImGui, read by Application)
struct NetworkConfig
{
    std::vector<int> layerSizes = {4, 8, 6, 3};
    std::vector<NeuralNetwork::ActivationType> layerActivations = {
        NeuralNetwork::ActivationType::Linear,
        NeuralNetwork::ActivationType::Sigmoid,
        NeuralNetwork::ActivationType::Sigmoid,
        NeuralNetwork::ActivationType::Sigmoid
    };

    bool rebuildPending         = false; // Application resets this after handling

    bool  inferenceMode         = false;
    std::vector<float> inputValues = {0.0f, 0.0f, 0.0f, 0.0f};

    // Weight file loader
    char weightPath[512]  = {};   // text-input buffer
    bool loadPending      = false; // Application calls loadWeights() then clears
    bool lastLoadOk       = false;
    bool lastLoadAttempted= false;
    bool weightsLoaded    = false; // locks architecture sliders
};

// Animation state (written by UI, executed by Application)
struct PropagationState
{
    bool  active      = false;
    int   layerIndex  = 1;
    float timer       = 0.0f;
    float delay       = 0.35f;
};

// Training state (written by UI, executed by Application)
struct TrainingState
{
    bool  active          = false;
    float learningRate    = 0.1f;
    int   epochsPerFrame  = 10;
    int   currentEpoch    = 0;
    float currentLoss     = 0.0f;
    bool  randomizePending= false;
};

// Master AppState containing all UI-editable states
struct AppState
{
    Params           params;
    NetworkConfig    netConfig;
    PropagationState prop;
    TrainingState    train;
};
