OPTION(PLUGINS_BUILD_HARDIREADER "Build HARDI Reader Plugin" ON)

if(PLUGINS_BUILD_HARDIREADER)

set( BMIA_HARDIREADER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	HARDIReaderPlugin.cxx
	HARDIReaderPlugin.h
	vtkHARDIReader.cxx
	vtkHARDIReader.h
	vtkSHARMReader.cxx
	vtkSHARMReader.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_cpp( MOC_SRCS HARDIReaderPlugin.h )

add_library( HARDIReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( HARDIReaderPlugin bmia_libs )

endif(PLUGINS_BUILD_HARDIREADER)