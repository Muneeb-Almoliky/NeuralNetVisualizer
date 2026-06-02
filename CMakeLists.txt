cmake_minimum_required(VERSION 3.21)
project(NeuralNetVisualizer VERSION 0.1.0 LANGUAGES CXX)

# ── Language standard ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Dependencies (resolved via vcpkg manifest) ────────────────────────────────
find_package(OpenGL   REQUIRED)
find_package(glad     REQUIRED CONFIG)
find_package(glfw3    REQUIRED CONFIG)
find_package(glm      REQUIRED CONFIG)
find_package(imgui    REQUIRED CONFIG)

# ── Source files ──────────────────────────────────────────────────────────────
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME} ${SOURCES})

# ── Include directories ───────────────────────────────────────────────────────
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    glad::glad
    OpenGL::GL
    glfw
    glm::glm
    imgui::imgui
)

# ── MSVC-specific flags ───────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX-)
    # Suppress min/max macro pollution from Windows headers
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
