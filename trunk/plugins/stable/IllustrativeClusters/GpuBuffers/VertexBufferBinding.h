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
 * VertexBufferBinding.h
 *
 * 2009-02-19	Ron Otten
 * - First version
 */

#ifndef opengl_VertexBufferBinding_h
#define opengl_VertexBufferBinding_h

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

/** A binding target (e.g., position, color, etc.) for an OpenGL vertex buffer object.
 * 	@remarks
 *		Consumed by instances of the VertexBufferDeclaration class.
 */
class VertexBufferBinding
{
public:
	/** Enumerates the possible binding targets for OpenGL vertex buffer objects. */
	enum BindingType
	{
		BND_VERTEX = GL_POSITION,
		BND_COLOR = GL_PRIMARY_COLOR
	};

	/** Enumerates the possible data storage type for the binding target's elements. */
	enum ElementType
	{
		ELM_FLOAT = GL_FLOAT
	};

	/** Creates a new VertexBufferBinding
	 *	@param bindingType
	 *		The type of target to bind to.
	 *	@param elementType
	 *		The data storage type for the binding's elements.
	 *	@param nrElements
	 *		How many elements / components a single vertex will take
	 *		in the binding.
	 */
	VertexBufferBinding(BindingType bindingType, ElementType elementType,
			unsigned int nrElements);

	/** Destroys the VertexBufferBinding */
	virtual ~VertexBufferBinding();

	/** Gets the target the binding is bound to.
	 *	@return
	 *		The bound target.
	 */
	inline BindingType getBindingType() const
	{
		return mBindingType;
	}

	/** Gets the data storage type for the binding's elements.
	 *	@return
	 *		The data storage type.
	 */
	inline ElementType getElementType() const
	{
		return mElementType;
	}

	/** Gets the number of elements in the binding one vertex takes.
	 *	@return
	 *		The number of elements for one vertex.
	 */
	inline unsigned int getNrElements() const
	{
		return mNrElements;
	}

	/** Gets the amount of bytes this binding would take to store
	 *  one buffer element (vertex).
	 * 	@return
	 * 		The number of bytes.
	 */
	unsigned int getSizeInBytes() const;

private:
	BindingType mBindingType;
	ElementType mElementType;
	unsigned int mNrElements;
};

}

#endif /* VERTEXBUFFERBINDING_H_ */
