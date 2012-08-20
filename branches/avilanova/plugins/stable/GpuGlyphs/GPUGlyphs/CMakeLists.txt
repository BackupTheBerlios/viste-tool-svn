PROJECT( BMIA_GPUGLYPHS )

# Evert van Aart: This folder contains a number of files that were probably used in
# the old DTITool, but are no longer used now. I'm leaving them here for future reference.
# Also, this folder is essentially a library, so it should be moved to the library folder.

SET( GPUGlyphs_classes
vtkGlyphMapperVA
vtkDTIGlyphMapperVA
vtkSHGlyphMapper
vtkCHGlyphMapper
)

FOREACH(class ${GPUGlyphs_classes})
  SET( GPUGlyphs_SRCS ${GPUGlyphs_SRCS} ${class}.h ${class}.cxx )
ENDFOREACH(class)

link_directories(${BMIA_LIBS_BINARY_DIR})
ADD_LIBRARY( vtkGPUGlyphs ${GPUGlyphs_SRCS} )
TARGET_LINK_LIBRARIES( vtkGPUGlyphs vtkHWShading vtkVolumeRendering )

subdirs(Shaders)
