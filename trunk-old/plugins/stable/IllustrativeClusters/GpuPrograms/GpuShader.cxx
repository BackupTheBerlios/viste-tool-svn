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
 * GpuShader.cpp
 *
 * 2009-02-18	Ron Otten
 * - First version
 */

#include "GpuShader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vtkDirectory.h>

namespace opengl
{

GpuShader::GpuShader(const std::string& name, const std::string& filePath,
		GpuShaderType type) :
	mShaderHandle(0), mName(name), mCompilationLog("")
{
	mFilePath = filePath;
	mShaderHandle = glCreateShader(static_cast<GLenum> (type));
	mShaderCode = this->loadFile(mFilePath);
}

GpuShader::~GpuShader()
{
	if (glIsShader(mShaderHandle) == GL_TRUE)
	{
		glDeleteShader(mShaderHandle);
	}
}

bool GpuShader::compile()
{
	if (glIsShader(mShaderHandle) != GL_TRUE)
		return false;

	if (mShaderCode.size() == 0)
	{
		mCompilationLog = "File " + mFilePath + " was not found or has zero length.";
		return false;
	}

	const GLchar* shaderCodePtr = mShaderCode.c_str();
	glShaderSource(mShaderHandle, 1, &shaderCodePtr, NULL);

	glCompileShader(mShaderHandle);

	GLint status;
	glGetShaderiv(mShaderHandle, GL_COMPILE_STATUS, &status);

	GLint logLength;
	glGetShaderiv(mShaderHandle, GL_INFO_LOG_LENGTH, &logLength);

	char* log = new char[logLength];

	glGetShaderInfoLog(mShaderHandle, logLength, NULL, log);
	mCompilationLog = std::string(log);

	delete[] log;

	return (status == GL_TRUE);
}

std::string GpuShader::loadFile(const std::string& filePath)
{
	// Use C++ STL streams to pull the shader file into a string stream, which in turn
	// puts out a string. Note that stringstream needs out type for the str() conversion
	// method.
	std::stringstream contentsStream(std::stringstream::in
			| std::stringstream::out);
	std::ifstream fileStream(filePath.c_str());

	if (fileStream.is_open())
	{
		contentsStream << fileStream.rdbuf();
	}

	return contentsStream.str();
}

}
