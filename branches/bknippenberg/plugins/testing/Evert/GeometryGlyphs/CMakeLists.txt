set( BMIA_GEOMETRYGLYPHS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	GeometryGlyphsPlugin.cxx
	GeometryGlyphsPlugin.h
	vtkGeometryGlyphBuilder.cxx
	vtkGeometryGlyphBuilder.h
	vtkGeometryGlyphFromSHBuilder.cxx
	vtkGeometryGlyphFromSHBuilder.h
)


include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})


qt4_wrap_ui(UiHeaders GeometryGlyphs.ui)
qt4_wrap_cpp(MOC_SRCS GeometryGlyphsPlugin.h)
set_source_files_properties(GeometryGlyphsPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})

add_library( GeometryGlyphsPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( GeometryGlyphsPlugin bmia_libs vtkHARDI )
