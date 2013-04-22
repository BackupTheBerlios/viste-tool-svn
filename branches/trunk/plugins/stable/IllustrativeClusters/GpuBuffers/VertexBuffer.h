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
 * VertexBuffer.h
 *
 * 2009-02-19	Ron Otten
 * - First version
 */

#ifndef opengl_VertexBuffer_h
#define opengl_VertexBuffer_h

#include <vector>
#include <GL/glew.h>
#include <GL/gl.h>

#include "VertexBufferDeclaration.h"

namespace opengl
{

/** A high level representation of an OpenGL vertex buffer object.
 *  @remarks
 *  	Currently requires that GLEW (GL Extension Wrangler) has been
 *  	succesfully initialized prior to usage of this class.
 */
class VertexBuffer
{
public:
	/** Creates a new vertex buffer.
	 *  @remarks
	 *  	Requires a call to declare() with a valid VertexBufferDeclaration
	 *      before use.
	 */
	VertexBuffer();

	/** Creates a new vertex buffer and declares the structure of its content.
	 *  @param declaration
	 *  	The declaration of the structure of the buffer's content.
	 */
	VertexBuffer(VertexBufferDeclaration declaration);

	/** Destroys the vertex buffer. */
	virtual ~VertexBuffer();

	/** Creates the vertex buffer and declares its structure.
	 *  @remarks
	 *  	Should only be called once and only if the parameterless constructor
	 *      was used to create the VertexBuffer instance.
	 *  @param declaration
	 *  	The declaration of the structure of the buffer's content.
	 */
	void declare(VertexBufferDeclaration declaration);

	/** Fills the VBO with data.
	 *	@param data
	 *		Pointer to the raw data to fit into the buffer.
	 *	@param nrElements
	 *		Number of vertex elements to fit into the buffer.
	 */
	void fill(void* data, unsigned int nrElements);

	/** Fills a range of the VBO with data.
	 *	@param data
	 *		Pointer to the raw data to fit into the buffer.
	 *	@param nrElements
	 *		Number of vertex elements to fit into the buffer.
	 *	@param elementOffset
	 *		Offset nr of elements from the start of the buffer.
	 */
	void fillRange(void* data, unsigned int nrElements, unsigned int elementOffset = 0);

	/** Binds the VBO for normal use.
	 *  @param elementStride
	 *  	How many consecutive elements should be set as the stride.
	 *  @param offset
	 *  	Offset nr of elements from the start of the buffer.
	 */
	void bind(unsigned int elementStride = 1, unsigned int elementOffset = 0);

	/** Unbinds the VBO from normal use. */
	void unbind();

	/** Binds the VBO for use as a transform feedback buffer. */
	void feedbackBind();

	/** Unbinds the VBO from use as a transform feedback buffer. */
	void feedbackUnbind();

private:
	GLuint mBufferHandle;
	GLintptr mBytesPerElement;
	VertexBufferDeclaration mDeclaration;

};

}

#endif /* VERTEXBUFFER_H_ */
