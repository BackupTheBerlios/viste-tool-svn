PROJECT( BMIA_HARDI )

find_package(VTK)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
endif(VTK_FOUND)

SET( HARDI_classes
HARDIMath
HARDIMeasures
HARDITransformationManager
SphereTriangulator
sphereTesselator
tesselation
triangle
vertex
)

FOREACH(class ${HARDI_classes})
  SET( HARDI_SRCS ${HARDI_SRCS} ${class}.h ${class}.cxx )
ENDFOREACH(class)

ADD_LIBRARY( vtkHARDI ${HARDI_SRCS} )
TARGET_LINK_LIBRARIES( vtkHARDI vtkCommon vtkIO vtkFiltering vtkImaging vtkGraphics vtkRendering )
