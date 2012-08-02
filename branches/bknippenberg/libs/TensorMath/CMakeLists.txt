PROJECT( BMIA_TENSORMATH )

find_package(VTK)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
endif(VTK_FOUND)

SET( TensorMath_classes
	ScalarMeasures
	DistanceMeasures
	Invariants
	vtkTensorStatistics
	AnisotropyMeasures
	vtkTensorMath
)

FOREACH(class ${TensorMath_classes})
  SET( TensorMath_SRCS ${TensorMath_SRCS} ${class}.h ${class}.cxx )
ENDFOREACH(class)

ADD_LIBRARY( vtkTensorMath ${TensorMath_SRCS} )
TARGET_LINK_LIBRARIES( vtkTensorMath vtkCommon vtkIO vtkFiltering vtkImaging )
