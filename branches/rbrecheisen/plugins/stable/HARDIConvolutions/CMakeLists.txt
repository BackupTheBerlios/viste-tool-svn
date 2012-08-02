OPTION(PLUGINS_BUILD_HARDICONVOLUTIONS "Build HARDI Convolutions Plugin" ON)

if (PLUGINS_BUILD_HARDICONVOLUTIONS)

set( BMIA_HARDICONVOLUTIONS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	HARDIConvolutionsPlugin.cxx
	HARDIConvolutionsPlugin.h
	KernelGenerator.cxx
	KernelGenerator.h
	KernelNIfTIWriter.cxx
	KernelNIfTIWriter.h
	KernelNIfTIReader.cxx
	KernelNIfTIReader.h
	vtkHARDIConvolutionFilter.cxx
	vtkHARDIConvolutionFilter.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders HARDIConvolutions.ui)
qt4_wrap_cpp(MOC_SRCS HARDIConvolutionsPlugin.h)
set_source_files_properties(HARDIConvolutionsPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( HARDIConvolutionsPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( HARDIConvolutionsPlugin bmia_core bmia_plugin bmia_data vtkNIfTI )

endif(PLUGINS_BUILD_HARDICONVOLUTIONS)