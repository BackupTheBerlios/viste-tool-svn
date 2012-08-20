OPTION(PLUGINS_BUILD_GPUGLYPHS "Build GPU Glyphs Plugin" ON)

if (PLUGINS_BUILD_GPUGLYPHS)

set( BMIA_GPUGLYPHS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/GPUGlyphs)

subdirs(GPUGlyphs)

set(SRCS
    GpuGlyphsPlugin.cxx
    GpuGlyphsPlugin.h
	vtkThresholdFilter.cxx
	vtkThresholdFilter.h
)

qt4_wrap_ui(UiHeaders GpuGlyphs.ui)
qt4_wrap_cpp( MOC_SRCS GpuGlyphsPlugin.h )
set_source_files_properties(GpuGlyphsPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})

add_library( GpuGlyphsPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( GpuGlyphsPlugin bmia_plugin bmia_data vtkGPUGlyphs vtkIO vtkGraphics vtkRendering vtkHWShading )

endif(PLUGINS_BUILD_GPUGLYPHS)