project(Bubbles LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

option(CONVEXHULL_QHULL "Use QHull for convex hull" ON)

set(BUBBLES "bubbles")
set(BBTOOL "bbtool")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/core"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/shapes"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/third"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/third/graphy"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/third/gDel3D/GDelFlipping/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/tests"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/apps"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/reconstruction"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/boundaries")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUDA_OUTPUT ${CMAKE_BINARY_DIR}/checkcuda)
execute_process(COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/checkcuda.cu -o ${CUDA_OUTPUT})
execute_process(COMMAND ${CUDA_OUTPUT}
                        RESULT_VARIABLE CUDA_RETURN_CODE
                        OUTPUT_VARIABLE CHK_CUDA_OUTPUT)


if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build")
    set(BB_FLAGS "-std=c++14 -g -G -DDEBUG --use_fast_math --extended-lambda")
else()
    message(STATUS "Release build")
    set(BB_FLAGS "-std=c++14 -Xptxas -O3 -DRELEASE --use_fast_math -maxrregcount 64 --extended-lambda")
endif()

set(cxx_warning_flags "-fdiagnostics-format=json")

if(NOT ${CUDA_RETURN_CODE} EQUAL 0)
    message(SEND_ERROR ${CHK_CUDA_OUTPUT})
else()
    set(ARCH "${CHK_CUDA_OUTPUT}")
    message(STATUS "CUDA Architecture: ${ARCH}")
    set(BB_FLAGS "${BB_FLAGS} --gpu-architecture=${ARCH}")
endif()

if(CONVEXHULL_QHULL)
    add_definitions(-DUSE_QHULL)
endif()

set(BB_COMPILE "-x cu -dc ${BB_FLAGS}")
set(BB_LINK ${BB_FLAGS})

if(CONVEXHULL_QHULL)
    set(BB_LINK_LIBS "dl;qhull_r")
else()
    set(BB_LINK_LIBS "dl")
endif()

file(GLOB BUBBLES_CORE ${CMAKE_CURRENT_SOURCE_DIR}/src/boundaries/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/equations/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/generator/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/shapes/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/reconstruction/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/third/*.cpp)


file(GLOB BUBBLES_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/bubbles.cpp)
file(GLOB BBTOOL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/bbtool.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/apps/*.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/third/gDel3D)

set(CMAKE_CXX_COMPILER nvcc)

add_library(corelib OBJECT ${BUBBLES_CORE})
set_target_properties(corelib PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")

add_executable(${BUBBLES} ${BUBBLES_SRCS} $<TARGET_OBJECTS:corelib>)
target_link_libraries(${BUBBLES} ${BB_LINK_LIBS};gflip3d)
set_target_properties(${BUBBLES} PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")

add_executable(${BBTOOL} ${BBTOOL_SRCS} $<TARGET_OBJECTS:corelib>)
target_link_libraries(${BBTOOL} ${BB_LINK_LIBS};gflip3d)
set_target_properties(${BBTOOL} PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")

