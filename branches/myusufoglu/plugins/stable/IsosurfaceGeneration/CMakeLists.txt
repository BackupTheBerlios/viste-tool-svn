OPTION(PLUGINS_BUILD_ISOSURFACEVISUALIZATION "Build Isosurface Visualization Plugin" ON)

if(PLUGINS_BUILD_ISOSURFACEVISUALIZATION)

	set( BMIA_ISOSURFACEVISUALIZATION_PLUGIN )

	FIND_PACKAGE(VTK REQUIRED)
	INCLUDE(${VTK_USE_FILE})
	FIND_PACKAGE(Qt4 REQUIRED)
	INCLUDE(${QT_USE_FILE})

	IF(NOT VTK_USE_CHARTS)
		MESSAGE(FATAL_ERROR "Isosurface Visualization Plugin requires VTK_USE_CHARTS.")
	ENDIF(NOT VTK_USE_CHARTS)

	SET (SRCS
		IsosurfaceGeneration.cxx
		IsosurfaceGeneration.h
		vtkInteractorStyleTrackballPositionPicker.cxx
		vtkInteractorStyleTrackballPositionPicker.h
		vtkCurvaturesShapeIndex.cxx
		vtkCurvaturesShapeIndex.h
	)

	include_directories(${CMAKE_CURRENT_BINARY_DIR})
	include_directories(${BMIA_LIBS_SOURCE_DIR})

	QT4_WRAP_UI(UiHeaders IsosurfaceGeneration.ui)
	SET_SOURCE_FILES_PROPERTIES(IsosurfaceGeneration.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

	QT4_WRAP_CPP(MOC_SRCS IsosurfaceGeneration.h)
	ADD_LIBRARY(IsosurfaceGeneration SHARED ${SRCS} ${MOC_SRCS})

	TARGET_LINK_LIBRARIES(IsosurfaceGeneration bmia_core bmia_plugin bmia_data vtkHybrid vtkCharts)

endif(PLUGINS_BUILD_ISOSURFACEVISUALIZATION)
