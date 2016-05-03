project(biped-evolve)

cmake_minimum_required(VERSION 2.8.4)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Find GLUT
find_package(GLUT REQUIRED)

# Local includes
set(INCLUDE_DIR include)
include_directories(${INCLUDE_DIR})
file(GLOB INCLUDES ${INCLUDE_DIR}/*.h)

# Glob all source files
set(SRC_DIR ${biped-evolve_SOURCE_DIR}/src)
file(GLOB SOURCES ${SRC_DIR}/*.cpp)

# Create executable
add_executable(biped-evolve ${SOURCES} ${INCLUDES})
target_link_libraries(biped-evolve ode drawstuff ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})