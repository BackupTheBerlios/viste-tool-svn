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
 * GpuProgram.cpp
 *
 * 2009-02-18	Ron Otten
 * - First version
 *
 * 2009-04-08	Ron Otten
 * - Added setUniform overload for uniform vec2
 */

#include "GpuProgram.h"

#include <sstream>

namespace opengl
{

GpuProgram::GpuProgram() :
	mProgramHandle(0), mBuildLog("")
{
	mProgramHandle = glCreateProgram();
}

GpuProgram::~GpuProgram()
{
	// First unbind the program, ...
	this->unbind();

	// ... then clean up any attached shaders ...
	GpuShaderMap::const_iterator it = mShaders.begin();
	for (; it != mShaders.end(); ++it)
	{
		GpuShader* shader = it->second;

		glDetachShader(mProgramHandle, shader->getShaderHandle());
		delete shader;
	}
	mShaders.clear();

	// ... and finally delete the underlying OpenGL Program object.
	if (glIsProgram(mProgramHandle) == GL_TRUE)
	{
		glDeleteProgram(mProgramHandle);
	}
}

void GpuProgram::bind()
{
	if (glIsProgram(mProgramHandle) == GL_TRUE)
	{
		glUseProgram(mProgramHandle);
	}
}

void GpuProgram::unbind()
{
	GLint currentProgramHandle = 0;

	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);
	if (static_cast<GLuint> (currentProgramHandle) == mProgramHandle)
	{
		glUseProgram(0);
	}
}

GpuShader* GpuProgram::createShader(const std::string& name, const std::string& filePath, GpuShader::GpuShaderType type)
{
	GpuShaderMap::const_iterator it = mShaders.find(name);
	if (it != mShaders.end())
	{
		// Shader with this name already exists.
		// TODO: Throw exception?

		return it->second;
	}

	GpuShader* shader = new GpuShader(name, filePath, type);
	mShaders.insert(std::make_pair(name, shader));

	glAttachShader(mProgramHandle, shader->getShaderHandle());

	return shader;
}

GpuShader* GpuProgram::getShader(const std::string& name)
{
	GpuShaderMap::const_iterator it = mShaders.find(name);
	if (it == mShaders.end())
	{
		// Shader with this name does not exist.
		// TODO: Throw exception?

		return NULL;
	}

	return it->second;
}

void GpuProgram::destroyShader(const std::string& name)
{
	GpuShaderMap::iterator it = mShaders.find(name);
	if (it == mShaders.end())
	{
		// Shader with this name does not exist.
		// TODO: Throw exception?

		return;
	}

	GpuShader* shader = it->second;
	glDetachShader(mProgramHandle, shader->getShaderHandle());
	delete shader;

	mShaders.erase(it);
}

void GpuProgram::destroyShader(GpuShader* shader)
{
	this->destroyShader(shader->getName());
}

bool GpuProgram::link(InputGeometry inputType, OutputGeometry outputType, unsigned int nrOutputVertices)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return false;

	glProgramParameteriEXT(mProgramHandle, GL_GEOMETRY_INPUT_TYPE_EXT,
	static_cast<GLint> (inputType));
	glProgramParameteriEXT(mProgramHandle, GL_GEOMETRY_OUTPUT_TYPE_EXT,
	static_cast<GLint> (outputType));
	glProgramParameteriEXT(mProgramHandle, GL_GEOMETRY_VERTICES_OUT_EXT,
	static_cast<GLint> (nrOutputVertices));

	if (!this->setActiveVaryings())
		return false;

	glLinkProgram(mProgramHandle);

	GLint status;
	glGetProgramiv(mProgramHandle, GL_LINK_STATUS, &status);

	GLint logLength;
	glGetProgramiv(mProgramHandle, GL_INFO_LOG_LENGTH, &logLength);

	char* log = new char[logLength];

	glGetProgramInfoLog(mProgramHandle, logLength, NULL, log);
	mBuildLog = std::string(log);

	delete[] log;

	if (status == GL_FALSE)
		return false;

	if (!this->setFeedbackVaryings())
	{
		mBuildLog += "\nOne or more 'varyings' not found\n.";
		return false;
	}

	return true;
}

bool GpuProgram::build(InputGeometry inputType, OutputGeometry outputType, unsigned int nrOutputVertices)
{
	bool hasErrors = false;

	std::stringstream buildLog(std::stringstream::in | std::stringstream::out);

	buildLog << std::endl << "Compiling component shaders ... " << std::endl;

	GpuShaderMap::const_iterator it = mShaders.begin();
	for (; it != mShaders.end(); ++it)
	{
		GpuShader* shader = it->second;

		buildLog << "Compiling " << shader->getName() << " ... " << std::endl;
		hasErrors = !shader->compile();
		buildLog << shader->getLastCompilationLog() << std::endl;

		if (hasErrors)
		{
			break;
		}
	}

	if (hasErrors)
	{
		mBuildLog = buildLog.str();
		return false;
	}

	buildLog << "Linking program ... " << std::endl;

	hasErrors = !this->link(inputType, outputType, nrOutputVertices);
	buildLog << mBuildLog << std::endl;

	mBuildLog = buildLog.str();

	return !hasErrors;
}

void GpuProgram::setUniform(const std::string& name, bool value)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform1i(glGetUniformLocation(mProgramHandle, name.c_str()), value);
		glUseProgram(currentProgramHandle);
	}
	else
	{
		glUniform1i(glGetUniformLocation(mProgramHandle, name.c_str()), value);
	}
}

void GpuProgram::setUniform(const std::string& name, bool v1, bool v2, bool v3)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform3i(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3);
		glUseProgram(currentProgramHandle);
	}
	else
	{
		glUniform3i(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3);
	}
}

void GpuProgram::setUniform(const std::string& name, int value)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform1i(glGetUniformLocation(mProgramHandle, name.c_str()), value);
		glUseProgram(currentProgramHandle);
	}
	else
	{
		glUniform1i(glGetUniformLocation(mProgramHandle, name.c_str()), value);
	}
}

void GpuProgram::setUniform(const std::string& name, int v1, int v2, int v3)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform3i(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3);
		glUseProgram(currentProgramHandle);
	}
	else
	{
		glUniform3i(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3);
	}
}

void GpuProgram::setUniform(const std::string& name, float value)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform1f(glGetUniformLocation(mProgramHandle, name.c_str()), value);
		glUseProgram(currentProgramHandle);
	}
	else
	{
		glUniform1f(glGetUniformLocation(mProgramHandle, name.c_str()), value);
	}
}

void GpuProgram::setUniform(const std::string& name, float v1, float v2)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform2f(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2);
		glUseProgram(currentProgramHandle);
	}
	else
	{
		glUniform2f(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2);
	}
}

void GpuProgram::setUniform(const std::string& name, float v1, float v2, float v3)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform3f(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3);
		glUseProgram(currentProgramHandle);
	}
	else
	{		
		glUniform3f(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3);
	}
}

void GpuProgram::setUniform(const std::string& name, float v1, float v2, float v3, float v4)
{
	if (glIsProgram(mProgramHandle) != GL_TRUE)
		return;

	GLint currentProgramHandle = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgramHandle);

	if (static_cast<GLuint> (currentProgramHandle) != mProgramHandle)
	{
		glUseProgram(mProgramHandle);
		glUniform4f(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3, v4);
		glUseProgram(currentProgramHandle);
	}
	else
	{		
		glUniform4f(glGetUniformLocation(mProgramHandle, name.c_str()), v1, v2, v3, v4);
	}
}

void GpuProgram::addVarying(const std::string& varying)
{
	VaryingsList::iterator it = mVaryings.begin();
	for (; it != mVaryings.end(); ++it)
	{
		if (*it == varying)
			return;
	}

	mVaryings.push_back(varying);
}

void GpuProgram::removeVarying(const std::string& varying)
{
	VaryingsList::iterator it = mVaryings.begin();
	for (; it != mVaryings.end(); ++it)
	{
		if (*it == varying)
		{
			mVaryings.erase(it);
			return;
		}
	}
}

bool GpuProgram::setActiveVaryings()
{
	VaryingsEnumerator varyings(mVaryings);
	while (varyings.moveNext())
	{
		glActiveVaryingNV(mProgramHandle, varyings.getCurrent().c_str());
	}

	return true;
}

bool GpuProgram::setFeedbackVaryings()
{
	GLint* attribs = new GLint[mVaryings.size()];

	VaryingsEnumerator varyings(mVaryings);
	for (int i = 0; varyings.moveNext(); ++i)
	{
		GLint location = glGetVaryingLocationNV(mProgramHandle, varyings.getCurrent().c_str());

		if (location == -1)
		{
			delete[] attribs;
			return false;
		}

		attribs[i] = location;
	}

	glTransformFeedbackVaryingsNV(mProgramHandle, mVaryings.size(), attribs, GL_INTERLEAVED_ATTRIBS_NV);

	delete[] attribs;
	return true;
}

}

