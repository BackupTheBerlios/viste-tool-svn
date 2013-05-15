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
 * VertexBufferDeclaration.h
 *
 * 2009-02-19	Ron Otten
 * - First version
 */

#ifndef opengl_VertexBufferDeclaration_h
#define opengl_VertexBufferDeclaration_h

#include <map>
#include <vector>
#include "VertexBufferBinding.h"
#include "../Enumerators/MapEnumerator.h"

namespace opengl
{

/** A declaration of the structure for an OpenGL vertex buffer object.
 * 	@remarks
 * 		Consumed by instances of the VertexBuffer class.
 */
class VertexBufferDeclaration
{
public:
	/** Represents a mapping from a type of buffer binding to the actual VertexBufferBinding instance. */
	typedef std::map<VertexBufferBinding::BindingType, VertexBufferBinding> VertexBufferBindingMap;

	/** Allows enumeration over a VertexBufferBindingMap without public exposure of the map's iterators. */
	typedef MapEnumerator<VertexBufferBindingMap> VertexBufferBindingEnumerator;

	/** Creates a new VertexBufferDeclaration. */
	VertexBufferDeclaration();

	/** Destroys the VertexBufferDeclaration. */
	virtual ~VertexBufferDeclaration();

	/** Create a new buffer binding for the VertexBufferDeclaration.
	 * 	@param bindingType
	 * 		The type of buffer binding.
	 * 	@param elementType
	 * 		The type of data element the binding consists of.
	 * 	@param nrElements
	 * 		The number of data elements one buffer entry consists of.
	 * 	@return
	 * 		The newly created binding.
	 */
	VertexBufferBinding createBinding(VertexBufferBinding::BindingType bindingType, VertexBufferBinding::ElementType elementType, unsigned int nrElements);

	/** Removes a specific buffer binding from the VertexBufferDeclaration and destroys it.
	 * 	@param bindingType
	 * 		The buffer binding type to remove and destroy.
	 */
	void destroyBinding(VertexBufferBinding::BindingType bindingType);

	/** Removes a specific buffer binding from the VertexBufferDeclaration and destroys it.
	 * 	@param binding
	 * 		The buffer binding to remove and destroy.
	 */
	void destroyBinding(VertexBufferBinding binding);

	/** Retrieves a previously created buffer binding from the VertexBufferDeclaration.
	 * 	@param bindingType
	 * 		The type of the binding to retrieve.
	 * 	@return
	 * 		The requested binding.
	 */
	VertexBufferBinding getBinding(VertexBufferBinding::BindingType bindingType);

	/** Get an enumerator over the collection of buffer bindings
	 * 	the VertexBufferDeclaration contains.
	 *  @return
	 *  	The enumerator over the collection.
	 */
	inline VertexBufferBindingEnumerator getBindingsEnumerator()
	{
		return VertexBufferBindingEnumerator(mBindings);
	}

	/** Gets the number of buffer bindings the
	 * 	VertexBufferDeclaration contains.
	 * 	@return
	 * 		The number of bindings present.
	 */
	inline unsigned int getNrBindings()
	{
		return mBindings.size();
	}

	/** Gets the amount of bytes one buffer element in a buffer
	 * 	structured according to the VertexBufferDeclaration
	 * 	would take.
	 * 	@return
	 * 		The number of bytes.
	 */
	unsigned int getSizeInBytes() const;

protected:

private:
	VertexBufferBindingMap mBindings;
};

}

#endif /* VERTEXBUFFERDECLARATION_H_ */
