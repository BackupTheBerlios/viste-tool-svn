/*
 * FrameBufferBinding.cpp
 *
 *  Created on: Apr 08, 2009
 *      Author: ron
 */

#include "FrameBufferBinding.h"

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

FrameBufferBinding::FrameBufferBinding(BindingType bindingType, DataType dataType, unsigned int textureUnit) :
	mBindingType(bindingType), mDataType(dataType), mTextureUnit(textureUnit)
{
}

FrameBufferBinding::~FrameBufferBinding()
{
}

}
