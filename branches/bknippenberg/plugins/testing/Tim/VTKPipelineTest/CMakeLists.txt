find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    VTKPipelineTestPlugin.h
    VTKPipelineTestPlugin.cxx
)

qt4_wrap_cpp( MOC_SRCS VTKPipelineTestPlugin.h )

add_library( VTKPipelineTest SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( VTKPipelineTest bmia_plugin bmia_data )
