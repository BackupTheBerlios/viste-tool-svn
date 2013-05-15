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
