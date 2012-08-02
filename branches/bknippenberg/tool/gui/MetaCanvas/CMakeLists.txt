SET( MetaCanvas_classes
vtkAbstractSubCanvas
vtkAbstractSubCanvasCollection
vtkSubCanvas
vtkGenericMetaCanvas
vtkMetaCanvas
vtkMetaCanvasInteractorStyle
vtkMetaCanvasInteractorStyleForward
vtkMetaCanvasInteractorStyleWM
vtkMetaCanvasInteractorStyleSwitch
vtkInteractorStyleSwitchFixed
vtkConeSubCanvas
vtkCylinderSubCanvas
vtkMedicalCanvas
)

include_directories(${BMIA_LIBS_SOURCE_DIR})

FOREACH(class ${MetaCanvas_classes})
  SET( MetaCanvas_SRCS ${MetaCanvas_SRCS} ${class}.h ${class}.cxx )
ENDFOREACH(class)

link_directories(${BMIA_LIBS_BINARY_DIR})
ADD_LIBRARY(vtkMetaCanvas ${MetaCanvas_SRCS})
target_link_libraries( vtkMetaCanvas vtkRendering )

IF(BMIA_BUILD_TESTS)
 SUBDIRS(test)
ENDIF(BMIA_BUILD_TESTS)
