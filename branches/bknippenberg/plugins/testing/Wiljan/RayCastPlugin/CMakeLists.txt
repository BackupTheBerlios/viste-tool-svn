cmake_minimum_required(VERSION 2.6)
set( BMIA_RAYCASTPLUGIN_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    RayCastPlugin.h
    RayCastPlugin.cxx
    RayCastVolumeMapper.h
	RayCastVolumeMapper.cxx
    vtkShaderBase.h
    vtkShaderBase.cxx
    vtkShaderRayDirections.h
    vtkShaderRayDirections.cxx
    vtkShaderDVR.h
    vtkShaderDVR.cxx
    vtkShaderMIP.h
    vtkShaderMIP.cxx
    vtkShaderIsosurface.h
    vtkShaderIsosurface.cxx
    vtkShaderToon.h
    vtkShaderToon.cxx
    vtkClippingPlane.h
    vtkClippingPlane.cxx
	vtkImageMask2.cxx
	vtkImageMask2.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

qt4_wrap_ui(UiHeaders RayCastPlugin.ui)
qt4_wrap_cpp( MOC_SRCS RayCastPlugin.h RayCastVolumeMapper.h)
set_source_files_properties(RayCastPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library( RayCastPlugin SHARED ${SRCS} ${MOC_SRCS} vtkShaderBase.h vtkShaderRayDirections.h vtkShaderDVR.h vtkShaderMIP.h vtkShaderIsosurface.h vtkShaderToon.h vtkClippingPlane.h)
target_link_libraries( RayCastPlugin bmia_plugin vtkIO vtkGraphics vtkVolumeRendering )
