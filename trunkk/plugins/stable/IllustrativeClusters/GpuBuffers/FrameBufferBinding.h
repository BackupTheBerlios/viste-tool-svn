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
 * FrameBufferBinding.h
 *
 * 2009-04-08	Ron Otten
 * - First version
 */

#ifndef opengl_FrameBufferBinding_h
#define opengl_FrameBufferBinding_h

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

class FrameBufferBinding
{
public:
	enum BindingType
	{
		BND_COLOR_ATTACHMENT = GL_COLOR_ATTACHMENT0_EXT,
		BND_DEPTH_ATTACHMENT = GL_DEPTH_ATTACHMENT_EXT
	};

	enum DataType
	{
		ELM_FLOAT = GL_FLOAT,
		ELM_UNSIGNED_BYTE = GL_UNSIGNED_BYTE
	};

	FrameBufferBinding(BindingType bindingType, DataType elementType, unsigned int textureUnit);

	virtual ~FrameBufferBinding();

	inline BindingType getBindingType() const
	{
		return mBindingType;
	}
	inline DataType getDataType() const
	{
		return mDataType;
	}
	inline unsigned int getTextureUnit() const
	{
		return mTextureUnit;
	}

private:
	BindingType mBindingType;
	DataType mDataType;
	unsigned int mTextureUnit;
};

}

#endif /* FRAMEBUFFERBINDING_H_ */
