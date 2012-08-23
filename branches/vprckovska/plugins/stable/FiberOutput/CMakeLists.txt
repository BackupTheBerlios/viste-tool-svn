OPTION(PLUGINS_BUILD_FIBEROUTPUT "Build Fiber Output Plugin" ON)

if (PLUGINS_BUILD_FIBEROUTPUT)

set( BMIA_FIBEROUTPUT_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	FiberOutputPlugin.cxx
	FiberOutputPlugin.h	
	FiberOutput.cxx
	FiberOutput.h	
 	FiberOutputTXT.cxx
	FiberOutputTXT.h	
	FiberOutputXML.cxx
    FiberOutputXML.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders FiberOutput.ui)
qt4_wrap_cpp(MOC_SRCS FiberOutputPlugin.h)
set_source_files_properties(FiberOutputPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( FiberOutputPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( FiberOutputPlugin bmia_libs vtkTensorMath )

endif(PLUGINS_BUILD_FIBEROUTPUT)