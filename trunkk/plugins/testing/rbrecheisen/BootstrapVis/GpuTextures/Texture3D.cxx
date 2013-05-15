/*
 * Texture3D.cpp
 *
 *  Created on: Feb 24, 2009
 *      Author: ron
 */

#include "Texture3D.h"
#include <algorithm>

namespace opengl
{

Texture3D::Texture3D() :
	mTextureHandle(0)
{
}

Texture3D::Texture3D(TextureDeclaration declaration) :
	mTextureHandle(0)
{
	this->declare(declaration);
}

Texture3D::~Texture3D()
{
	if (glIsTexture(mTextureHandle) == GL_TRUE)
	{
		this->unbind();
		glDeleteTextures(1, &mTextureHandle);
	}
}

void Texture3D::declare(TextureDeclaration declaration)
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
	glGetIntegerv(GL_TEXTURE_BINDING_3D, &originalTextureHandle);
	glBindTexture(GL_TEXTURE_3D, mTextureHandle);

	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER,
			static_cast<GLfloat>(mDeclaration.getFiltering()));
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER,
			static_cast<GLfloat>(mDeclaration.getFiltering()));
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S,
			static_cast<GLfloat>(mDeclaration.getWrapping()));
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,
			static_cast<GLfloat>(mDeclaration.getWrapping()));
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R,
			static_cast<GLfloat>(mDeclaration.getWrapping()));
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glBindTexture(GL_TEXTURE_3D, originalTextureHandle);
	glActiveTexture(originalTextureUnit);
}

bool Texture3D::fill(void* data, unsigned int xDim, unsigned int yDim,
		unsigned int zDim)
{
	unsigned int dimensions[3];
	dimensions[0] = xDim;
	dimensions[1] = yDim;
	dimensions[2] = zDim;

	GLint originalTextureHandle;
	glGetIntegerv(GL_TEXTURE_BINDING_3D, &originalTextureHandle);
	glBindTexture(GL_TEXTURE_3D, mTextureHandle);

	// Check the dimensions and pixel formats using GL_PROXY_TEXTURE_3D
	glTexImage3D(GL_PROXY_TEXTURE_3D, 0,
			static_cast<GLenum> (mDeclaration.getInternalFormat()),
			dimensions[0], dimensions[1], dimensions[2], 0,
			static_cast<GLenum> (mDeclaration.getPixelFormat()),
			static_cast<GLenum> (mDeclaration.getDataType()), data);

	// Check if GL_TEXTURE_WIDTH took the requested dimensions or is set to 0
	int texWidth;
	glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0,
			GL_TEXTURE_WIDTH, &texWidth);

	if (texWidth == 0)
		return false;

	glTexImage3D(GL_TEXTURE_3D, 0,
			static_cast<GLenum> (mDeclaration.getInternalFormat()),
			dimensions[0], dimensions[1], dimensions[2], 0,
			static_cast<GLenum> (mDeclaration.getPixelFormat()),
			static_cast<GLenum> (mDeclaration.getDataType()), data);

	glBindTexture(GL_TEXTURE_3D, originalTextureHandle);

	return true;
}

void Texture3D::bind()
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

	glBindTexture(GL_TEXTURE_3D, mTextureHandle);

	glActiveTexture(originalTextureUnit);
	
}

void Texture3D::unbind()
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
	glGetIntegerv(GL_TEXTURE_BINDING_3D, &currentHandle);

	if (static_cast<GLuint> (currentHandle) == mTextureHandle)
	{
		glBindTexture(GL_TEXTURE_3D, 0);
	}

	glActiveTexture(originalTextureUnit);
}

}
