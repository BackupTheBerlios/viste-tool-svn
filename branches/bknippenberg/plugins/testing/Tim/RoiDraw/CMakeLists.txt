set( BMIA_ROIDRAW_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    RoiDrawDialog.cxx
    RoiDrawPlugin.cxx
    RoiDrawDialog.h
    RoiDrawPlugin.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

qt4_wrap_ui(UiHeaders rois.ui roidraw.ui)
qt4_wrap_cpp( MOC_SRCS RoiDrawPlugin.h RoiDrawDialog.h )
set_source_files_properties(RoiDrawDialog.cxx RoiDrawPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library( RoiDrawPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( RoiDrawPlugin bmia_libs )
