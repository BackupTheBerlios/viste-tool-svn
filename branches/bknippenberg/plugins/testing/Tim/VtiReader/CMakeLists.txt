set( BMIA_VTIREADER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
   VtiReaderPlugin.h
   VtiReaderPlugin.cxx
)

qt4_wrap_cpp( MOC_SRCS VtiReaderPlugin.h )

add_library( VtiReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( VtiReaderPlugin bmia_libs )
