set( BMIA_SEEDING_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	vtk2DRoiToSeedFilter.cxx
	vtk2DRoiToSeedFilter.h
	SeedingPlugin.cxx
	SeedingPlugin.h
	vtkPolyDataToSeedPoints.cxx
	vtkPolyDataToSeedPoints.h
	vtkScalarVolumeToSeedPoints.cxx
	vtkScalarVolumeToSeedPoints.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders Seeding.ui)
qt4_wrap_cpp( MOC_SRCS SeedingPlugin.h )
set_source_files_properties(SeedingPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})

add_library( SeedingPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( SeedingPlugin bmia_libs )
