// Entry point — constructs the Application and starts the main loop.
// All engine logic lives in Application, Renderer, Camera, and Network.

#include "Application.h"

int main()
{
    Application app(1280, 720, "Neural Net Visualizer");
    return app.run();
}
