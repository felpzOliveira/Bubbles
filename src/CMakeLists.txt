file(GLOB BUBBLES_CORE boundaries/*.cpp 
					   core/*.cpp 
					   cuda/*.cpp 
					   equations/*.cpp 
					   generator/*.cpp 
					   shapes/*.cpp 
					   solvers/*.cpp 
					   tests/*.cpp 
					   third/*.cpp 
					   apps/*.cpp)

file(GLOB BUBBLES_SRCS bubbles.cpp)
file(GLOB BBTOOL_SRCS bbtool.cpp)
list(APPEND BUBBLES_SRCS ${BUBBLES_CORE})
list(APPEND BBTOOL_SRCS ${BUBBLES_CORE})

add_executable(${BUBBLES} ${BUBBLES_SRCS})
target_link_libraries(${BUBBLES} "dl")
set_target_properties(${BUBBLES} PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")

add_executable(${BBTOOL} ${BBTOOL_SRCS})
target_link_libraries(${BBTOOL} "dl")
set_target_properties(${BBTOOL} PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")
