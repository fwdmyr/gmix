# Works with 3.14 and tested through 3.26
cmake_minimum_required(VERSION 3.14...3.26)

# Project name and a few useful settings. Other commands can pick up the results
project(
  GMSAM
  VERSION 0.1
  DESCRIPTION "WIP"
  LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Testing only available if this is the main app
  # Note this needs to be done in the main CMakeLists
  # since it calls enable_testing, which must be in the
  # main CMakeLists.
  option(GMSAM_ENABLE_TESTS "Build tests" ON)
endif()

find_package(Boost REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(gmsam INTERFACE)
target_sources(gmsam INTERFACE include/Common.hpp, include/GaussianComponent.hpp, include/GaussianMixture.hpp)
target_include_directories(gmsam INTERFACE include)
target_compile_features(gmsam INTERFACE cxx_std_17)

# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND GMSAM_ENABLE_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
