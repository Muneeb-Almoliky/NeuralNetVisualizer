// Arcball / orbital camera implementation.

#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <GLFW/glfw3.h>

#include <algorithm> // std::clamp

// Construction
Camera::Camera(glm::vec3 target,
               float     yawDeg,
               float     pitchDeg,
               float     radius)
    : m_target(target)
    , m_yaw   (yawDeg)
    , m_pitch (pitchDeg)
    , m_radius(radius)
{}

// Input callbacks
void Camera::onMouseButton(int button, int action, double cursorX, double cursorY)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            m_dragging = true;
            m_lastX    = cursorX;
            m_lastY    = cursorY;
        }
        else if (action == GLFW_RELEASE)
        {
            m_dragging = false;
        }
    }
}

void Camera::onMouseMove(double cursorX, double cursorY)
{
    if (!m_dragging) return;

    const double dx = cursorX - m_lastX;
    const double dy = cursorY - m_lastY;

    m_lastX = cursorX;
    m_lastY = cursorY;

    // Horizontal drag  → yaw (orbit left/right)
    m_yaw   += static_cast<float>(dx) * sensitivity;

    // Vertical drag    → pitch (orbit up/down), clamped to avoid flipping
    m_pitch -= static_cast<float>(dy) * sensitivity; // subtract: screen-Y is inverted
    m_pitch  = std::clamp(m_pitch, -89.0f, 89.0f);
}

void Camera::onScroll(double offsetY)
{
    m_radius -= static_cast<float>(offsetY) * zoomSpeed;
    m_radius  = std::clamp(m_radius, minRadius, maxRadius);
}

// Camera matrices
glm::vec3 Camera::computeEyePosition() const
{
    // Convert spherical (yaw, pitch, radius) → Cartesian offset from target.
    const float yawRad   = glm::radians(m_yaw);
    const float pitchRad = glm::radians(m_pitch);

    const float cosP = glm::cos(pitchRad);

    return m_target + glm::vec3(
        m_radius * cosP * glm::sin(yawRad),   // X
        m_radius * glm::sin(pitchRad),         // Y  (elevation)
        m_radius * cosP * glm::cos(yawRad)    // Z
    );
}

glm::mat4 Camera::getViewMatrix() const
{
    const glm::vec3 eye = computeEyePosition();
    return glm::lookAt(eye, m_target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::vec3 Camera::getPosition() const
{
    return computeEyePosition();
}
