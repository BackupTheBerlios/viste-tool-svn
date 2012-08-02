PROJECT( BMIA_GPUGLYPHS )

SET( GPUGlyphs_classes
#vtkVectorVolumeMapper
#vtkGlyphMapper
#vtkDTIGlyphMapper
vtkGlyphMapperVA
vtkDTIGlyphMapperVA
vtkSHGlyphMapper
vtkCHGlyphMapper
)

#IF(BMIA_USE_HARDI)
#SET( GPUGlyphs_classes ${GPUGlyphs_classes} vtkSHGlyphMapper vtkCHGlyphMapper )
#ENDIF(BMIA_USE_HARDI)

FOREACH(class ${GPUGlyphs_classes})
  SET( GPUGlyphs_SRCS ${GPUGlyphs_SRCS} ${class}.h ${class}.cxx )
ENDFOREACH(class)

link_directories(${BMIA_LIBS_BINARY_DIR})
ADD_LIBRARY( vtkGPUGlyphs ${GPUGlyphs_SRCS} )
TARGET_LINK_LIBRARIES( vtkGPUGlyphs vtkHWShading vtkVolumeRendering )

subdirs(Shaders)

IF(BMIA_BUILD_TESTS)
	#  SUBDIRS(test)
ENDIF(BMIA_BUILD_TESTS)
