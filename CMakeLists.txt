cmake_minimum_required(VERSION 3.16)
project(mmwave_gait_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(USE_LIBTORCH "Enable LibTorch backend for PointNet++" OFF)

add_executable(mmwave_app
    src/main.cpp
    src/config.cpp
    src/dataset.cpp
    src/preprocess.cpp
    src/features.cpp
    src/classifier.cpp
    src/softmax_classifier.cpp
    src/fgst_model.cpp
    src/metrics.cpp
    src/model_io.cpp
    src/pipeline.cpp
)

target_include_directories(mmwave_app PRIVATE include)

if (USE_LIBTORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(mmwave_app PRIVATE MMWAVE_USE_LIBTORCH=1)
    target_link_libraries(mmwave_app PRIVATE "${TORCH_LIBRARIES}")
endif()

if (MSVC)
    target_compile_options(mmwave_app PRIVATE /W4 /EHsc)
else()
    target_compile_options(mmwave_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

