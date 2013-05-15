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
 * FrameBuffer.cpp
 *
 * 2009-04-08	Ron Otten
 * - First version
 */

#include "FrameBuffer.h"

namespace opengl
{

FrameBuffer::FrameBuffer() :
	mBufferHandle(0)
{
}

FrameBuffer::FrameBuffer(FrameBufferDeclaration declaration) :
	mBufferHandle(0)
{
	this->declare(declaration);
}

FrameBuffer::~FrameBuffer()
{
	if (glIsFramebufferEXT(mBufferHandle) == GL_TRUE)
	{
		GLint currentBufferHandle = 0;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &currentBufferHandle);

		if (static_cast<GLuint>(currentBufferHandle) != mBufferHandle)
		{
			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mBufferHandle);
		}		

		BindingMap::iterator it = mBindings.begin();
		for(; it != mBindings.end(); ++it)
		{	
			FrameBufferBinding::BindingType type = it->first;
			Texture2D* texture = it->second;			
			
			texture->unbind();
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, static_cast<GLenum>(type), GL_TEXTURE_2D, 0, 0);

			delete texture;
		}

		if (static_cast<GLuint>(currentBufferHandle) == mBufferHandle)
		{
			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		}
		else
		{
			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, currentBufferHandle);
		}

		glDeleteFramebuffersEXT(1, &mBufferHandle);
	}
}

bool FrameBuffer::declare(FrameBufferDeclaration declaration)
{
	if (glIsFramebufferEXT(mBufferHandle) != GL_FALSE)
	{
		// already declared buffers return false
		return false;
	}
	
	glGenFramebuffersEXT(1, &mBufferHandle);
	mDeclaration = declaration;

	GLint currentBufferHandle = 0;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &currentBufferHandle);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mBufferHandle);

	bool bindColor = false;
	bool success = true;
	FrameBufferDeclaration::FrameBufferBindingEnumerator declaredBindings = mDeclaration.getBindingsEnumerator();
	while(declaredBindings.moveNext())
	{
		FrameBufferBinding binding = declaredBindings.getCurrentValue();		

		if (binding.getBindingType() == FrameBufferBinding::BND_COLOR_ATTACHMENT)
		{
			bindColor = true;
		}

		TextureDeclaration::PixelFormat pixelFormat;
		TextureDeclaration::InternalFormat internalFormat;

		switch(binding.getBindingType())
		{
		case FrameBufferBinding::BND_COLOR_ATTACHMENT:
			pixelFormat = TextureDeclaration::PXF_RGBA;
			internalFormat = TextureDeclaration::ITF_RGBA;
			break;
		case FrameBufferBinding::BND_DEPTH_ATTACHMENT:
			pixelFormat = TextureDeclaration::PXF_DEPTH;
			internalFormat = TextureDeclaration::ITF_DEPTH;
			break;
		default:
			pixelFormat = TextureDeclaration::PXF_RGBA;
			internalFormat = TextureDeclaration::ITF_RGBA;
			break;
		}

		TextureDeclaration textureDeclaration(
			binding.getTextureUnit(),
			pixelFormat, internalFormat,
			static_cast<TextureDeclaration::DataType>(binding.getDataType()),
			TextureDeclaration::WRP_CLAMP_EDGE, TextureDeclaration::FIL_NEAREST
		);

		Texture2D* texture = new Texture2D(textureDeclaration);
		success &= texture->fill(NULL, declaration.getWidth(), declaration.getHeight());
				
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, static_cast<GLenum>(binding.getBindingType()), GL_TEXTURE_2D, texture->getHandle(), 0);
		
		mBindings.insert(std::make_pair(binding.getBindingType(), texture));
	}

	if (!bindColor)		
	{
		glReadBuffer(GL_NONE);
		glDrawBuffer(GL_NONE);
	}

	success &= (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) == GL_FRAMEBUFFER_COMPLETE_EXT);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, currentBufferHandle);

	return success;
}

void FrameBuffer::bind()
{
	if (glIsFramebufferEXT(mBufferHandle) != GL_TRUE)
	{
		return;
	}

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mBufferHandle);	
}

void FrameBuffer::unbind()
{
	if (glIsFramebufferEXT(mBufferHandle) != GL_TRUE)
	{
		return;
	}

	GLint currentBufferHandle = 0;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &currentBufferHandle);

	if (static_cast<GLuint>(currentBufferHandle) == mBufferHandle)
	{
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	}
}

Texture2D* FrameBuffer::getBoundTexture(FrameBufferBinding::BindingType bindingType)
{
	BindingMap::const_iterator it = mBindings.find(bindingType);
	if (it == mBindings.end())
	{
		// TODO: Throw exception instead?
		return NULL;
	}

	return it->second;
}

}
