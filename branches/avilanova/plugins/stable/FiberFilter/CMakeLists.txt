OPTION(PLUGINS_BUILD_FIBERFILTER "Build Fiber Filter Plugin" ON)

if(PLUGINS_BUILD_FIBERFILTER)

set( BMIA_FIBERFILTER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
   FiberFilterWidget.h
   FiberFilterWidget.cxx
   FiberFilterPlugin.h
   FiberFilterPlugin.cxx
   vtk2DROIFiberFilter.h
   vtk2DROIFiberFilter.cxx
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

qt4_wrap_ui(UiHeaders fiberfilter.ui)
qt4_wrap_cpp( MOC_SRCS FiberFilterPlugin.h FiberFilterWidget.h )
set_source_files_properties(FiberFilterWidget.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library( FiberFilterPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( FiberFilterPlugin bmia_libs )

endif(PLUGINS_BUILD_FIBERFILTER)