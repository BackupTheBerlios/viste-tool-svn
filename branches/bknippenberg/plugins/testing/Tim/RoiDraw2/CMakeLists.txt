set( BMIA_ROIDRAW_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    RoiDrawPlugin.cxx
    RoiDrawPlugin.h
	ROIGroupDialog.cxx
	ROIGroupDialog.h
	vtkImageTracerWidget2.cxx
	vtkImageTracerWidget2.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

qt4_wrap_ui(UiHeaders ROIEdit.ui)
qt4_wrap_cpp(MOC_SRCS RoiDrawPlugin.h)
set_source_files_properties(RoiDrawPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library( RoiDrawPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( RoiDrawPlugin bmia_libs )
