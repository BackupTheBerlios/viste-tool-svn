set( BMIA_DTIREADER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    vtkDTIReader2.cxx
	vtkDTIReader2.h
    DTIReaderPlugin.cxx
    DTIReaderPlugin.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_cpp( MOC_SRCS DTIReaderPlugin.h )

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( DTIReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( DTIReaderPlugin bmia_libs )
