project(Bubbles LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(BUBBLES "bubbles")
set(BBTOOL "bbtool")
set(CMAKE_CXX_COMPILER nvcc)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/core"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/third"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/third/graphy"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/cuda"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/tests"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/apps"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/boundaries")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUDA_OUTPUT ${CMAKE_BINARY_DIR}/checkcuda)
execute_process(COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/checkcuda.cu -o ${CUDA_OUTPUT})
execute_process(COMMAND ${CUDA_OUTPUT}
				RESULT_VARIABLE CUDA_RETURN_CODE
				OUTPUT_VARIABLE CHK_CUDA_OUTPUT)


if(CMAKE_BUILD_TYPE MATCHES Debug)
	message(STATUS "Debug build")
	set(BB_FLAGS "-g -G --use_fast_math")
else()
	message(STATUS "Release build")
	set(BB_FLAGS "-Xptxas -O3 --use_fast_math -maxrregcount 128")
endif()

if(NOT ${CUDA_RETURN_CODE} EQUAL 0)
	message(SEND_ERROR ${CHK_CUDA_OUTPUT})
else()
	set(ARCH "${CHK_CUDA_OUTPUT}")
	message(STATUS "CUDA Architecture: ${ARCH}")
	set(BB_FLAGS "${BB_FLAGS} --gpu-architecture=${ARCH}")
endif()

set(BB_COMPILE "-x cu -dc ${BB_FLAGS}")
set(BB_LINK ${BB_FLAGS})
add_subdirectory(src)