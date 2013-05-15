/*
 * FrameBufferDeclaration.cpp
 *
 * 2009-04-08	Ron Otten
 * - First version
 */

#include "FrameBufferDeclaration.h"

namespace opengl
{

FrameBufferDeclaration::FrameBufferDeclaration():
	mWidth(0), mHeight(0)
{
}

FrameBufferDeclaration::FrameBufferDeclaration(const unsigned int width, const unsigned int height):
	mWidth(width), mHeight(height)
{
}

FrameBufferDeclaration::~FrameBufferDeclaration()
{
}

FrameBufferBinding FrameBufferDeclaration::createBinding(
	FrameBufferBinding::BindingType bindingType,
	FrameBufferBinding::DataType dataType,
	unsigned int textureUnit)
{
	FrameBufferBindingMap::const_iterator it = mBindings.find(bindingType);
	if (it != mBindings.end())
	{
		// Binding already exists
		// TODO: Throw exception instead?
		return it->second;
	}

	FrameBufferBinding binding(bindingType, dataType, textureUnit);

	mBindings.insert(std::make_pair(bindingType, binding));
	return binding;
}

FrameBufferBinding FrameBufferDeclaration::getBinding(FrameBufferBinding::BindingType bindingType)
{
	FrameBufferBindingMap::iterator it = mBindings.find(bindingType);
	if (it == mBindings.end())
	{
		// Binding doesn't exist
		// TODO: Throw exception instead?
		return FrameBufferBinding(FrameBufferBinding::BND_COLOR_ATTACHMENT, FrameBufferBinding::ELM_UNSIGNED_BYTE, 0);
	}

	return it->second;
}

bool FrameBufferDeclaration::containsBinding(FrameBufferBinding::BindingType bindingType)
{
	FrameBufferBindingMap::iterator it = mBindings.find(bindingType);
	return (it != mBindings.end());
}

void FrameBufferDeclaration::destroyBinding(FrameBufferBinding::BindingType bindingType)
{
	FrameBufferBindingMap::iterator it = mBindings.find(bindingType);
	if (it == mBindings.end())
	{
		// Binding doesn't exist
		// TODO: Throw exception instead?
		return;
	}

	mBindings.erase(it);
}

void FrameBufferDeclaration::destroyBinding(FrameBufferBinding binding)
{
	this->destroyBinding(binding.getBindingType());
}

}
