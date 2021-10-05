file(GLOB BUBBLES_CORE boundaries/*.cpp 
					   core/*.cpp 
					   cuda/*.cpp 
					   equations/*.cpp 
					   generator/*.cpp 
					   shapes/*.cpp 
					   solvers/*.cpp 
                       reconstruction/*.cpp
					   tests/*.cpp 
					   third/*.cpp)

file(GLOB BUBBLES_SRCS bubbles.cpp)
file(GLOB BBTOOL_SRCS bbtool.cpp apps/*.cpp)

add_library(corelib OBJECT ${BUBBLES_CORE})
set_target_properties(corelib PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")

add_executable(${BUBBLES} ${BUBBLES_SRCS} $<TARGET_OBJECTS:corelib>)
target_link_libraries(${BUBBLES} ${BB_LINK_LIBS})
set_target_properties(${BUBBLES} PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")

add_executable(${BBTOOL} ${BBTOOL_SRCS} $<TARGET_OBJECTS:corelib>)
target_link_libraries(${BBTOOL} ${BB_LINK_LIBS})
set_target_properties(${BBTOOL} PROPERTIES COMPILE_FLAGS "${BB_COMPILE}" LINK_FLAGS "${BB_FLAGS}")
