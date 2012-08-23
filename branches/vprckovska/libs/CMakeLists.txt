project(BMIA_LIBS)

#include_directories(${BMIA_TOOL_SOURCE_DIR})
#include_directories(${BMIA_TOOL_BINARY_DIR})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin/plugins)
#set(BMIA_SHADER_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)

add_subdirectory( Helpers )
add_subdirectory( HWShading )
add_subdirectory( HARDI )
add_subdirectory( TensorMath )
add_subdirectory( NIfTI )

# avoid CMake complaining that bmia_libs doesn't have any source files
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.c "")
add_library(bmia_libs "${CMAKE_CURRENT_BINARY_DIR}/dummy.c")

target_link_libraries(bmia_libs bmia_data bmia_plugin vtkIO vtkGraphics vtkRendering vtkVolumeRendering vtkWidgets vtkHelpers vtkHWShading)
