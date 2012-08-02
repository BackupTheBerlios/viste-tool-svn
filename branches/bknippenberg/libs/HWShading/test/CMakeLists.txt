PROJECT(BMIA_HWSHADING_TEST)

#INCLUDE(${BMIA_DTITOOL2_SOURCE_DIR}/UseVTK.cmake)

SET(COPYFILES
testshader.frag
testshader.vert
testshader.prog
average.shad
DirectionalLight.vert
standard.vert
standard.prog
ppshader.vert
ppshader.frag
ppshader.prog
alshader.vert
alshader.prog
alshaderpp.prog
alshaderpp.vert
alshaderpp.frag
texture.prog
texture.vert
texture.frag
)

FOREACH(copyfile ${COPYFILES})
CONFIGURE_FILE(	${BMIA_HWSHADING_TEST_SOURCE_DIR}/${copyfile}
		${BMIA_HWSHADING_TEST_BINARY_DIR}/${copyfile}
		COPYONLY
	      )
ENDFOREACH(copyfile)

ADD_EXECUTABLE(NoShader NoShader.cxx)
TARGET_LINK_LIBRARIES(NoShader vtkRendering)

ADD_EXECUTABLE(TestShader TestShader.cxx)
TARGET_LINK_LIBRARIES(TestShader vtkHWShading)

ADD_EXECUTABLE(texturemap texturemap.cxx)
# vtkTensorData for loading a tensor dataset for testing
TARGET_LINK_LIBRARIES(texturemap vtkHWShading) #vtkRendering vtkTensorData)

ADD_EXECUTABLE(shadows shadows.cxx)
TARGET_LINK_LIBRARIES(shadows vtkHWShading)

ADD_EXECUTABLE(fibermapper fibermapper.cxx)
TARGET_LINK_LIBRARIES(fibermapper vtkHWShading vtkIO)

ADD_EXECUTABLE(fibershadows fibershadows.cxx)
TARGET_LINK_LIBRARIES(fibershadows vtkHWShading vtkIO)

#ADD_EXECUTABLE(ext LoadOpenGLExtension.cxx)
#TARGET_LINK_LIBRARIES(ext vtkRendering)
