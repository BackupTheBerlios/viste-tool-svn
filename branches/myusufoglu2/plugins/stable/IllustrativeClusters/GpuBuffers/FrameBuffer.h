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
 * FrameBuffer.h
 *
 * 2009-04-08	Ron Otten
 * - First version
 */

#ifndef opengl_FrameBuffer_h
#define opengl_FrameBuffer_h

#include <map>
#include <GL/glew.h>
#include <GL/gl.h>

#include "FrameBufferDeclaration.h"
#include "../GpuTextures/Texture2D.h"

namespace opengl
{

/** A high level representation of an OpenGL frame buffer object.
 *  @remarks
 *  	Currently requires that GLEW (GL Extension Wrangler) has been
 *  	succesfully initialized prior to usage of this class.
 */
class FrameBuffer
{
public:
	/** Constructor
	 *  @remarks
	 *  	Requires a call to declare() with a valid FrameBufferDeclaration
	 *      before use.
	 */
	FrameBuffer();

	/** Constructor
	 *  @param declaration
	 *  	The declaration of the structure of the buffer's content.
	 */
	FrameBuffer(FrameBufferDeclaration declaration);

	/** Destructor */
	virtual ~FrameBuffer();

	/** Creates the frame buffer and declares its structure.
	 *  @remarks
	 *  	Should only be called once and only if the parameterless constructor
	 *      was used to create the FrameBuffer instance.
	 *  @param declaration
	 *  	The declaration of the structure of the buffer's content.
	 *	@return
	 *		Whether declaration was successful.
	 */
	bool declare(FrameBufferDeclaration declaration);

	/** Gets the texture bound to the specific binding type
	 *  @param bindingType
	 *		The type of the binding for which to retrieve the texture.
	 *	@return
	 *		The retrieved texture.
	 */
	Texture2D* getBoundTexture(FrameBufferBinding::BindingType bindingType);

	/** Binds the FBO for use. */
	void bind();

	/** Unbinds the FBO from use. */
	void unbind();

	/** Gets the width in pixels of the frame buffer.
	 *	@return
	 *		The width in pixels.
	 */
	inline unsigned int getWidth() const
	{
		return mDeclaration.getWidth();
	}

	/** Gets the height in pixels of the frame buffer.
	 *	@return
	 *		The height in pixels.
	 */
	inline unsigned int getHeight() const
	{
		return mDeclaration.getHeight();
	}

protected:
	typedef std::map<FrameBufferBinding::BindingType, Texture2D*> BindingMap;

private:
	GLuint mBufferHandle;
	BindingMap mBindings;
	
	FrameBufferDeclaration mDeclaration;
};

}

#endif /* FRAMEBUFFER_H_ */
