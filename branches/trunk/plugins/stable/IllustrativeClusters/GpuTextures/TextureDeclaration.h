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
 * TextureDeclaration.h
 *
 * 2009-02-24	Ron Otten
 * - First version
 */

#ifndef opengl_TextureDeclaration_h
#define opengl_TextureDeclaration_h

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

/** A declaration of the structure for an OpenGL texture object.
 * 	@remarks
 * 		Consumed by instances of the Texture2D and Texture3D classes.
 */
class TextureDeclaration
{
public:
	/** Enumerates possible pixel formats. */
	enum PixelFormat
	{
		PXF_RGBA = GL_RGBA,
		PXF_DEPTH = GL_DEPTH_COMPONENT
	};

	/** Enumerates possible internal OpenGL storage formats */
	enum InternalFormat
	{
		ITF_RGBA = GL_RGBA,
		ITF_RGBA16F_UNCLAMPED = GL_RGBA16F_ARB,
		ITF_RGBA32F_UNCLAMPED = GL_RGBA32F_ARB,
		ITF_DEPTH = GL_DEPTH_COMPONENT
	};

	/** Enumerates possible data types texture data is stored with or retrieved in */
	enum DataType
	{
		ELM_FLOAT = GL_FLOAT,
		ELM_UNSIGNED_BYTE = GL_UNSIGNED_BYTE
	};

	/** Enumerates the possible ways to wrap textures */
	enum Wrapping
	{
		WRP_CLAMP = GL_CLAMP,
		WRP_CLAMP_EDGE = GL_CLAMP_TO_EDGE
	};

	/** Enumerates possible types of texture filtering */
	enum Filtering
	{
		FIL_NEAREST = GL_NEAREST,
		FIL_LINEAR = GL_LINEAR,
		FIL_CUBIC = GL_CUBIC_EXT
	};

	/** Creates a new TextureDeclaration */
	TextureDeclaration();
	
	/** Creates a new TextureDeclaration and configures it
	 *	@param textureUnit
	 *		The OpenGL texture unit textures built with this declaration will be assigned to.
	 *	@param pixelFormat
	 *		The pixel format textures built with this declaration will use.
	 *	@param internalFormat
	 *		The internal OpenGL storage format textures built with this declaration will use.
	 *	@param dataType
	 *		The datatype by which textures built with this declaration are exposed to the programmer.
	 *  @param wrapping
	 *		The type of texture wrapping textures built with this declaration will use.
	 *  @param filtering
	 *		The type of filtering textures built with this declaration will use.
	 */
	TextureDeclaration(
		unsigned int textureUnit, PixelFormat pixelFormat, InternalFormat internalFormat,
		DataType dataType, Wrapping wrapping, Filtering filtering
	);

	/** Destroys the TextureDeclaration */
	virtual ~TextureDeclaration();

	inline PixelFormat getPixelFormat() const { return mPixelFormat; }
	inline void setPixelFormat(PixelFormat pixelFormat) { mPixelFormat = pixelFormat; }

	inline InternalFormat getInternalFormat() const { return mInternalFormat; }
	inline void setInternalFormat(InternalFormat internalFormat) { mInternalFormat = internalFormat; }

	inline DataType getDataType() const { return mDataType; }
	inline void setDataType(DataType dataType) { mDataType = dataType; }

	inline Wrapping getWrapping() const { return mWrapping; }
	inline void setWrapping(Wrapping wrapping) { mWrapping = wrapping; }

	inline Filtering getFiltering() const { return mFiltering; }
	inline void setFiltering(Filtering filtering) { mFiltering = filtering; }

	inline unsigned int getTextureUnit() const { return mTextureUnit; }
	inline void setTextureUnit(unsigned int textureUnit) { mTextureUnit = textureUnit; }

private:
	unsigned int mTextureUnit;
	PixelFormat mPixelFormat;
	InternalFormat mInternalFormat;
	DataType mDataType;
	Wrapping mWrapping;
	Filtering mFiltering;
};

}

#endif /* TEXTUREDECLARATION_H_ */
