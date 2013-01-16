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
