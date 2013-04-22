set( BMIA_HARDIFIBERTRACKING_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    	HARDIFiberTrackingPlugin.h
	HARDIFiberTrackingPlugin.cxx
	HARDIFiberTrackingPlugin_Deterministic.cxx
	HARDIdeterministicTracker.cxx
	HARDIdeterministicTracker.h
	vtkHARDIFiberTrackingFilter.h
	vtkHARDIFiberTrackingFilter.cxx
	HARDI/sphereTesselator.cxx
	HARDI/sphereTesselator.h
	HARDI/SphereTriangulator.cxx
	HARDI/SphereTriangulator.h
	HARDI/tesselation.cxx
	HARDI/tesselation.h	
	HARDI/triangle.cxx
	HARDI/triangle.h
	HARDI/vertex.cxx
	HARDI/vertex.h
	HARDI/HARDIMath.cxx
	HARDI/HARDIMath.h
	HARDI/HARDITransformationManager.cxx
	HARDI//HARDITransformationManager.h
	vtkGeometryGlyphBuilder.cxx
	vtkGeometryGlyphBuilder.h
	vtkGeometryGlyphFromSHBuilder.cxx
	vtkGeometryGlyphFromSHBuilder.h

)

#set(DEFAULT_SRCS ${MAIN_SRCS}
#	FiberTrackingPlugin_Config_Default.cxx
#	FiberTrackingPlugin_Geodesic.cxx
#	geodesicFiberNeighborhood.cxx
#	geodesicFiberNeighborhood.h
#	geodesicPreProcessor.cxx
#	geodesicPreProcessor.h
#	geodesicFiberTracker.cxx
#	geodesicFiberTracker.h
#	vtkFiberTrackingGeodesicFilter.cxx
#	vtkFiberTrackingGeodesicFilter.h
#)


qt4_wrap_cpp(MOC_SRCS HARDIFiberTrackingPlugin.h)
add_library(HARDIFiberTrackingPlugin SHARED ${SRCS} ${MOC_SRCS})
target_link_libraries(HARDIFiberTrackingPlugin bmia_core bmia_plugin bmia_data)


include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

link_directories(${BMIA_LIBS_BINARY_DIR})

qt4_wrap_ui(UiHeaders HARDIFiberTracking.ui)
set_source_files_properties(HARDIFiberTrackingPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

