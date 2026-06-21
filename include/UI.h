#pragma once

// Renders the ImGui side-panel dashboard.
// Reads/writes application state via AppState; reads Renderer only
// for its layer-history data and static layerColor helper.

#include "AppState.h"
#include "Network.h"

class Renderer;
class Camera;

class UI
{
public:
    void draw(AppState& state, const Renderer& renderer,
              NeuralNetwork& net, const Camera& cam);
};
