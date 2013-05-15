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
 * FrameBufferDeclaration.h
 *
 * 2009-04-08	Ron Otten
 * - First version
 */

#ifndef opengl_FrameBufferDeclaration_h
#define opengl_FrameBufferDeclaration_h

#include <map>
#include <vector>
#include "FrameBufferBinding.h"
#include "../Enumerators/MapEnumerator.h"

namespace opengl
{

/** A declaration of the structure for an OpenGL frame buffer object.
 * 	@remarks
 * 		Consumed by instances of the FrameBuffer class.
 */
class FrameBufferDeclaration
{
public:
	/** Represents a mapping from a type of buffer binding to the actual FrameBufferBinding instance. */
	typedef std::map<FrameBufferBinding::BindingType, FrameBufferBinding> FrameBufferBindingMap;

	/** Allows enumeration over a FrameBufferBindingMap without public exposure of the map's iterators. */
	typedef MapEnumerator<FrameBufferBindingMap> FrameBufferBindingEnumerator;

	/** Creates a new FrameBufferDeclaration */
	FrameBufferDeclaration();	

	/** Creates a new FrameBufferDeclaration and sets its dimensions.
	 *  @param width
	 *		The width in pixels the frame buffer will take.
	 *	@param height
	 *		The height in pixels the frame buffer will take.
	 */
	FrameBufferDeclaration(const unsigned int width, const unsigned int height);

	/** Destroys the FrameBufferDeclaration. */
	virtual ~FrameBufferDeclaration();

	/** Create a new buffer binding for the FrameBufferDeclaration.
	 * 	@param bindingType
	 * 		The type of buffer binding.
	 * 	@param dataType
	 * 		The type of data element the binding consists of.
	 *	@param textureUnit
	 *		The texure unit that the texture within the binding will be assigned to.
	 * 	@return
	 * 		The newly created binding.
	 */
	FrameBufferBinding createBinding(
		FrameBufferBinding::BindingType bindingType,
		FrameBufferBinding::DataType dataType,
		unsigned int textureUnit
	);

	/** Removes a specific binding from the FrameBufferDeclaration and destroys it.
	 * 	@param bindingType
	 * 		The buffer binding type to remove and destroy.
	 */
	void destroyBinding(FrameBufferBinding::BindingType bindingType);

	/** Removes a specific buffer binding from the FrameBufferDeclaration and destroys it.
	 * 	@param binding
	 * 		The buffer binding to remove and destroy.
	 */
	void destroyBinding(FrameBufferBinding binding);

	/** Retrieves a previously created buffer binding from the FrameBufferDeclaration.
	 * 	@param bindingType
	 * 		The type of the binding to retrieve.
	 * 	@return
	 * 		The requested binding.
	 */
	FrameBufferBinding getBinding(FrameBufferBinding::BindingType bindingType);

	/** Checks if a particular buffer binding exists in the FrameBufferDeclaration
	 *	@param bindingType
	 *		The type of the binding to check for.
	 *	@return
	 *		Whether the binding exists.
	 */
	bool containsBinding(FrameBufferBinding::BindingType bindingType);
	
	/** Get an enumerator over the collection of buffer bindings
	 * 	the FrameBufferDeclaration contains.
	 *  @return
	 *  	The enumerator over the collection.
	 */
	inline FrameBufferBindingEnumerator getBindingsEnumerator()
	{
		return FrameBufferBindingEnumerator(mBindings);
	}

	/** Gets the width in pixels the frame buffer will take.
	 *	@return
	 *		The width in pixels.
	 */
	inline unsigned int getWidth() const
	{
		return mWidth;
	}

	/** Sets the width in pixels the frame buffer will take.
	 *	@param width
	 *		The width in pixels.
	 */
	inline void setWidth(const unsigned int width)
	{
		mWidth = width;
	}

	/** Gets the height in pixels the frame buffer will take.
	 *	@return
	 *		The height in pixels.
	 */
	inline unsigned int getHeight() const
	{
		return mHeight;
	}

	/** Sets the height in pixels the frame buffer will take.
	 *	@param height
	 *		The height in pixels.
	 */
	inline void setHeight(const unsigned int height)
	{
		mHeight = height;
	}

protected:

private:
	FrameBufferBindingMap mBindings;
	unsigned int mWidth;
	unsigned int mHeight;
};

}

#endif /* FRAMEBUFFERDECLARATION_H_ */
