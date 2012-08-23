OPTION(PLUGINS_BUILD_VOLUMEREADER "Build Volume Reader Plugin" ON)

if(PLUGINS_BUILD_VOLUMEREADER)

set( BMIA_VOLUMEREADER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    VolumeReaderPlugin.h
    VolumeReaderPlugin.cxx
)

qt4_wrap_cpp( MOC_SRCS VolumeReaderPlugin.h )

add_library( VolumeReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( VolumeReaderPlugin bmia_plugin bmia_data vtkIO )

endif(PLUGINS_BUILD_VOLUMEREADER)