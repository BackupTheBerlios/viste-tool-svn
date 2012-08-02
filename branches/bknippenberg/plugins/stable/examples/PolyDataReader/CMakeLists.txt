set( BMIA_POLYDATAREADER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    PolyDataReaderPlugin.cxx
    PolyDataReaderPlugin.h
)

qt4_wrap_cpp( MOC_SRCS PolyDataReaderPlugin.h )

add_library( PolyDataReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( PolyDataReaderPlugin bmia_libs )
