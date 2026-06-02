#pragma once

// Arcball / orbital camera.
// The camera orbits a fixed target point using spherical coordinates
// (yaw, pitch, radius).  Input is fed via three callbacks that mirror
// the GLFW cursor / scroll / mouse-button signatures so they can be
// forwarded directly from the GLFW event handlers in main.cpp.

#include <glm/glm.hpp>

class Camera
{
public:
    // Construction
    // target  : world-space point the camera orbits around
    // yawDeg  : initial horizontal angle (degrees)
    // pitchDeg: initial vertical   angle (degrees, clamped to avoid gimbal)
    // radius  : initial distance from target
    explicit Camera(glm::vec3 target   = glm::vec3(0.0f),
                    float     yawDeg   = 45.0f,
                    float     pitchDeg = 25.0f,
                    float     radius   = 8.0f);

    // GLFW-forwarded input

    // Call from glfwSetMouseButtonCallback.
    // button / action mirror GLFW_MOUSE_BUTTON_* / GLFW_PRESS / GLFW_RELEASE.
    void onMouseButton(int button, int action, double cursorX, double cursorY);

    // Call from glfwSetCursorPosCallback.
    void onMouseMove(double cursorX, double cursorY);

    // Call from glfwSetScrollCallback.
    void onScroll(double offsetY);

    // Camera matrices
    [[nodiscard]] glm::mat4 getViewMatrix()  const;

    // Convenience: world-space eye position (useful for lighting calculations).
    [[nodiscard]] glm::vec3 getPosition()    const;
    [[nodiscard]] glm::vec3 getTarget()      const { return m_target; }

    // Tuning
    float sensitivity = 0.25f; // degrees per pixel dragged
    float zoomSpeed   = 0.35f; // radius units per scroll notch
    float minRadius   = 1.5f;
    float maxRadius   = 50.0f;

private:
    glm::vec3 m_target;

    float m_yaw;    // degrees, rotates around world-Y
    float m_pitch;  // degrees, elevation above horizon

    float m_radius; // distance from target

    // Drag state
    bool   m_dragging   = false;
    double m_lastX      = 0.0;
    double m_lastY      = 0.0;

    // Recomputes world-space eye position from spherical coordinates.
    [[nodiscard]] glm::vec3 computeEyePosition() const;
};
