include_directories(${BMIA_TOOL_SOURCE_DIR})
include_directories(${BMIA_TOOL_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})
include_directories(${BMIA_LIBS_BINARY_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin/plugins)
set(BMIA_SHADER_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)

add_subdirectory( stable )
add_subdirectory( testing )
