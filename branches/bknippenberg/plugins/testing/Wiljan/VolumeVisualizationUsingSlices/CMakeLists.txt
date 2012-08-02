set( BMIA_VOLUMEVISUALIZATIONUSINGSLICES_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    VolumeVisualizationUsingSlicesPlugin.cxx
)

qt4_wrap_cpp( MOC_SRCS VolumeVisualizationUsingSlicesPlugin.h )

add_library( bmia_VolumeVisualizationUsingSlicesPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( bmia_VolumeVisualizationUsingSlicesPlugin bmia_plugin bmia_data vtkIO vtkGraphics vtkVolumeRendering )
