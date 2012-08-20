PROJECT( BMIA_HELPERS )

find_package(VTK)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
endif(VTK_FOUND)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

SET( Helpers_classes
vtkImageDataInterpolator
vtkBetterDataReader
vtkDoubleToUCharFilter
#vtkLookupTableCollection
vtkImageSliceActor
vtkImageOrthogonalSlicesActor
vtkThresholdMask
vtkPointClassification
vtkDTIComponentReader
TransformationMatrixIO
vtkStreamlineToSimplifiedStreamline
)

FOREACH(class ${Helpers_classes})
  SET( Helpers_SRCS ${Helpers_SRCS} ${class}.h ${class}.cxx )
ENDFOREACH(class)

ADD_LIBRARY( vtkHelpers ${Helpers_SRCS} )
TARGET_LINK_LIBRARIES( vtkHelpers vtkCommon vtkIO vtkFiltering vtkImaging vtkGraphics vtkRendering ${QT_LIBRARIES})
