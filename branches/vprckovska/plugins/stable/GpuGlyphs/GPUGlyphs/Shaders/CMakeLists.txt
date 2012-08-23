INCLUDE(shaders.cmake)
FOREACH(shaderfile ${ShaderFiles})
    CONFIGURE_FILE(	${CMAKE_CURRENT_SOURCE_DIR}/${shaderfile}
			${BMIA_SHADER_OUTPUT_DIRECTORY}/${shaderfile}
			COPYONLY
		  )
ENDFOREACH(shaderfile)
