/*
 * TextureDeclaration.cpp
 *
 *  Created on: Feb 24, 2009
 *      Author: ron
 */

#include "TextureDeclaration.h"

namespace opengl
{

TextureDeclaration::TextureDeclaration() :
	mTextureUnit(0), mPixelFormat(PXF_RGBA), mInternalFormat(ITF_RGBA),
			mDataType(ELM_FLOAT), mWrapping(WRP_CLAMP_EDGE), mFiltering(
					FIL_LINEAR)
{
}

TextureDeclaration::TextureDeclaration(unsigned int textureUnit,
		PixelFormat pixelFormat, InternalFormat internalFormat,
		DataType dataType, Wrapping wrapping, Filtering filtering) :
	mTextureUnit(textureUnit), mPixelFormat(pixelFormat), mInternalFormat(
			internalFormat), mDataType(dataType), mWrapping(wrapping),
			mFiltering(filtering)
{
}

TextureDeclaration::~TextureDeclaration()
{
}

}
