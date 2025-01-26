cmake_minimum_required(VERSION 3.30)
project(untitled)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled main.cpp
        src/Matrix.cpp
        src/Matrix.h
        src/Node.cpp
        src/Node.h
        src/Layer.h
        src/Layer.cpp
        src/Activation.h
        src/Activation.cpp
        src/LossFunction.h
        src/LossFunction.cpp
        src/Gradient_Descent.h
        src/Gradient_Descent.cpp
        src/Dense.h
        src/Dense.cpp
)
