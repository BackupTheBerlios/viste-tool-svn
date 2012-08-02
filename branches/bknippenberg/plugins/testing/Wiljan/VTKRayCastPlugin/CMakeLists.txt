set( BMIA_VTKRAYCAST_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    VTKRayCastPlugin.cxx
)

qt4_wrap_cpp( MOC_SRCS VTKRayCastPlugin.h )

add_library( bmia_VTKRayCastPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( bmia_VTKRayCastPlugin bmia_plugin bmia_data vtkIO vtkGraphics vtkVolumeRendering )
