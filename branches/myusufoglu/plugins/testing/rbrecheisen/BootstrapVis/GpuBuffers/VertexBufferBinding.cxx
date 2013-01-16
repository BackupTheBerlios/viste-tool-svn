/*
 * VertexBufferBinding.cpp
 *
 *  Created on: Feb 19, 2009
 *      Author: ron
 */

#include "VertexBufferBinding.h"

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

VertexBufferBinding::VertexBufferBinding(BindingType bindingType,
		ElementType elementType, unsigned int nrElements) :
	mBindingType(bindingType), mElementType(elementType), mNrElements(
			nrElements)
{
}

VertexBufferBinding::~VertexBufferBinding()
{
}

unsigned int VertexBufferBinding::getSizeInBytes() const
{
	unsigned int elementSize;

	switch (mElementType)
	{
	case ELM_FLOAT:
		elementSize = sizeof(GLfloat);
		break;
	default:
		elementSize = 0;
		break;
	}

	return elementSize * mNrElements;
}

}
