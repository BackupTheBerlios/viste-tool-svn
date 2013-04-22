/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * GpuShader.cxx
 *
 * 2009-02-18	Ron Otten
 * - First version
 */

#ifndef opengl_GpuShader_h
#define opengl_GpuShader_h

#include <string>

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

/** A high level representation of an OpenGL shader object. Capable
 *  of representing vertex, fragment and geometry shaders.
 *  @remarks
 *  	Currently requires that GLEW (GL Extension Wrangler) has been
 *  	succesfully initialized prior to usage of this class.
 */
class GpuShader
{
public:
	/** Enumerates the possible types of OpenGL shader object the
	 *  GpuShader class can represent.
	 */
	enum GpuShaderType
	{
		GST_VERTEX   = GL_VERTEX_SHADER,
		GST_FRAGMENT = GL_FRAGMENT_SHADER,
		GST_GEOMETRY = GL_GEOMETRY_SHADER_EXT
	};

	/** Gets the name under which the GPU shader was created.
	 *  @return
	 *  	The name of the GPU shader.
	 */
	std::string getName() const
	{
		return mName;
	}

	/** Gets the source code of the GPU shader.
	 *  @return
	 *  	The source code.
	 */
	std::string getSourceCode() const
	{
		return mShaderCode;
	}

	/** Assigns a piece of source code to the GPU shader. */
	void setSourceCode(const std::string& code)
	{
		mShaderCode = code;
	}

	/** Compiles the GPU shader
	 *	@return
	 *		Whether compilation was successful.
	 */
	bool compile();

	/** Gets the log of the GPU shader's most recent compilation.
	 *  @return
	 *  	The GPU shader's last compilation log.
	 */
	inline std::string getLastCompilationLog() const
	{
		return mCompilationLog;
	}

protected:
	friend class GpuProgram;

	/** Creates a new GPU shader.
	 *  @param name
	 *		The name by which the GPU shader will be identified.
	 *	@param filePath
	 *		The path to the file containing the shader code.
	 *	@param type
	 *		The type of shader to create.
	 */
	GpuShader(const std::string& name, const std::string& filePath, GpuShaderType type);

	/** Destroys the GPU shader.
	 */
	virtual ~GpuShader();

	/** Gets the OpenGL handle associated with the GPU shader.
	 *  @return
	 *  	The shader's OpenGL handle.
	 */
	GLuint getShaderHandle()
	{
		return mShaderHandle;
	}

private:
	GLuint mShaderHandle;
	std::string mName;
	std::string mFilePath;
	std::string mShaderCode;
	std::string mCompilationLog;

	/** Loads the text content of a file
	 *  @param filePath
	 *  	The path to the file to load.
	 *  @return
	 *  	The text content of the file.
	 */
	std::string loadFile(const std::string& filePath);

	/* Copy constructor and assignment operator are private to ensure that
	 * this class can't be copied by value.
	 */
	GpuShader(const GpuShader& rhs);
	GpuShader& operator =(GpuShader& rhs);
};

}

#endif /* GPUSHADER_H_ */
