#PROJECT( BMIA_HELPERS_TEST )

#INCLUDE(${DTITool2_SOURCE_DIR}/VTK/UseVTK.cmake)
include_directories(${BMIA_LIBS_SOURCE_DIR}/Helpers)

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/simplevolume.vtk
		${CMAKE_CURRENT_BINARY_DIR}/simplevolume.vtk
		COPYONLY
		)

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/tensors.vtk
		${CMAKE_CURRENT_BINARY_DIR}/tensors.vtk
		COPYONLY
		)

ADD_EXECUTABLE(interpolation interpolation.cxx)
TARGET_LINK_LIBRARIES(interpolation vtkHelpers vtkIO)

ADD_EXECUTABLE(sliceactor sliceactor.cxx)
TARGET_LINK_LIBRARIES(sliceactor vtkHelpers vtkRendering)
