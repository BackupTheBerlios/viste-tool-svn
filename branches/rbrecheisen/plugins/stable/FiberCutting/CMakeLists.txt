OPTION(PLUGINS_BUILD_FIBERCUTTING "Build Fiber Cutting Plugin" ON)

if (PLUGINS_BUILD_FIBERCUTTING)

set( BMIA_FIBERCUTTING_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	FiberCuttingPlugin.cxx
	FiberCuttingPlugin.h
	
	vtkInteractorStyleTrackballCellPicker.h
	vtkInteractorStyleTrackballCellPicker.cxx
	
	vtkInteractorStyleTrackballCellEndPicker.h
	vtkInteractorStyleTrackballCellEndPicker.cxx
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders FiberCutting.ui)
qt4_wrap_cpp( MOC_SRCS FiberCuttingPlugin.h )
set_source_files_properties(FiberCuttingPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library( FiberCuttingPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( FiberCuttingPlugin bmia_libs bmia_core )

endif(PLUGINS_BUILD_FIBERCUTTING)