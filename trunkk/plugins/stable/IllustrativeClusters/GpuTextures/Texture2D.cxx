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
 * Texture3D.cpp
 *
 *  Created on: Feb 24, 2009
 *      Author: ron
 */

#include "Texture2D.h"
#include <algorithm>

namespace opengl
{

Texture2D::Texture2D() :
	mTextureHandle(0)
{
}

Texture2D::Texture2D(TextureDeclaration declaration) :
	mTextureHandle(0)
{
	this->declare(declaration);
}

Texture2D::~Texture2D()
{
	if (glIsTexture(mTextureHandle) == GL_TRUE)
	{
		this->unbind();
		glDeleteTextures(1, &mTextureHandle);
	}
}

void Texture2D::declare(TextureDeclaration declaration)
{
	mDeclaration = declaration;

	GLint maxTexCoords;
	glGetIntegerv(GL_MAX_TEXTURE_COORDS, &maxTexCoords);

	GLint maxCombinedUnits;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxCombinedUnits);

	GLenum maxTextureUnit = GL_TEXTURE0 + std::max<GLint>(maxTexCoords,
			maxCombinedUnits) - 1;

	GLenum textureUnit = GL_TEXTURE0 + mDeclaration.getTextureUnit();

	if (textureUnit > maxTextureUnit)
	{
		// Texture unit is not available on this hardware.
		// TODO: Throw exception?
		return;
	}

	GLint originalTextureUnit;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &originalTextureUnit);
	glActiveTexture(textureUnit);

	glGenTextures(1, &mTextureHandle);

	GLint originalTextureHandle;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &originalTextureHandle);
	glBindTexture(GL_TEXTURE_2D, mTextureHandle);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLfloat>(mDeclaration.getFiltering()));
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLfloat>(mDeclaration.getFiltering()));
	glTexParameterf(GL_TEXTURE_2D,     GL_TEXTURE_WRAP_S,  static_cast<GLfloat>(mDeclaration.getWrapping()));
	glTexParameterf(GL_TEXTURE_2D,     GL_TEXTURE_WRAP_T,  static_cast<GLfloat>(mDeclaration.getWrapping()));
	glTexParameterf(GL_TEXTURE_2D,     GL_TEXTURE_WRAP_R,  static_cast<GLfloat>(mDeclaration.getWrapping()));
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glBindTexture(GL_TEXTURE_2D, originalTextureHandle);
	glActiveTexture(originalTextureUnit);
}

bool Texture2D::fill(void* data, unsigned int xDim, unsigned int yDim)
{
	unsigned int dimensions[2];
	dimensions[0] = xDim;
	dimensions[1] = yDim;	

	GLint originalTextureHandle;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &originalTextureHandle);
	glBindTexture(GL_TEXTURE_2D, mTextureHandle);

	// Check the dimensions and pixel formats using GL_PROXY_TEXTURE_2D
	glTexImage2D(
		GL_PROXY_TEXTURE_2D, 0,
		static_cast<GLenum> (mDeclaration.getInternalFormat()),
		dimensions[0], dimensions[1], 0,
		static_cast<GLenum> (mDeclaration.getPixelFormat()),
		static_cast<GLenum> (mDeclaration.getDataType()), data
	);

	// Check if GL_TEXTURE_WIDTH took the requested dimensions or is set to 0
	int texWidth;
	glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texWidth);

	if (texWidth == 0)
		return false;

	glTexImage2D(
		GL_TEXTURE_2D, 0,
		static_cast<GLenum> (mDeclaration.getInternalFormat()),
		dimensions[0], dimensions[1], 0,
		static_cast<GLenum> (mDeclaration.getPixelFormat()),
		static_cast<GLenum> (mDeclaration.getDataType()), data
	);

	glBindTexture(GL_TEXTURE_2D, originalTextureHandle);

	return true;
}

void Texture2D::bind()
{
	if (glIsTexture(mTextureHandle) != GL_TRUE)
		return;

	GLint maxTexCoords;
	glGetIntegerv(GL_MAX_TEXTURE_COORDS, &maxTexCoords);

	GLint maxCombinedUnits;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxCombinedUnits);

	GLenum maxTextureUnit = GL_TEXTURE0 + std::max<GLint>(maxTexCoords, maxCombinedUnits) - 1;
	GLenum textureUnit = GL_TEXTURE0 + mDeclaration.getTextureUnit();
	if (textureUnit > maxTextureUnit)
	{
		// Texture unit is not available on this hardware.
		// TODO: Throw exception?
		return;
	}

	GLint originalTextureUnit;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &originalTextureUnit);
	glActiveTexture(textureUnit);

	glBindTexture(GL_TEXTURE_2D, mTextureHandle);

	glActiveTexture(originalTextureUnit);
	
}

void Texture2D::unbind()
{
	if (glIsTexture(mTextureHandle) != GL_TRUE)
		return;

	GLint maxTexCoords;
	glGetIntegerv(GL_MAX_TEXTURE_COORDS, &maxTexCoords);

	GLint maxCombinedUnits;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxCombinedUnits);

	GLenum maxTextureUnit = GL_TEXTURE0 + std::max<GLint>(maxTexCoords, maxCombinedUnits) - 1;
	GLenum textureUnit = GL_TEXTURE0 + mDeclaration.getTextureUnit();
	if (textureUnit > maxTextureUnit)
	{
		// Texture unit is not available on this hardware.
		// TODO: Throw exception?
		return;
	}

	GLint originalTextureUnit;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &originalTextureUnit);
	glActiveTexture(textureUnit);

	GLint currentHandle;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &currentHandle);

	if (static_cast<GLuint> (currentHandle) == mTextureHandle)
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glActiveTexture(originalTextureUnit);
}

}
