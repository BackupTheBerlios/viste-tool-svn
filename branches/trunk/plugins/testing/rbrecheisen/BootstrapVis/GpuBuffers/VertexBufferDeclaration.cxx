/*
 * VertexBufferDeclaration.cpp
 *
 * 2009-02-19	Ron Otten
 * - First version
 */

#include "VertexBufferDeclaration.h"

namespace opengl
{

VertexBufferDeclaration::VertexBufferDeclaration()
{
}

VertexBufferDeclaration::~VertexBufferDeclaration()
{
}

VertexBufferBinding VertexBufferDeclaration::createBinding(
		VertexBufferBinding::BindingType bindingType,
		VertexBufferBinding::ElementType elementType, unsigned int nrElements)
{
	VertexBufferBindingMap::const_iterator it = mBindings.find(bindingType);
	if (it != mBindings.end())
	{
		// Binding already exists
		// TODO: Throw exception instead?
		return it->second;
	}

	VertexBufferBinding binding(bindingType, elementType, nrElements);

	mBindings.insert(std::make_pair(bindingType, binding));
	return binding;
}

VertexBufferBinding VertexBufferDeclaration::getBinding(
		VertexBufferBinding::BindingType bindingType)
{
	VertexBufferBindingMap::iterator it = mBindings.find(bindingType);
	if (it == mBindings.end())
	{
		// Binding doesn't exist
		// TODO: Throw exception instead?
		return VertexBufferBinding(VertexBufferBinding::BND_VERTEX, VertexBufferBinding::ELM_FLOAT, 0);
	}

	return it->second;
}

void VertexBufferDeclaration::destroyBinding(
		VertexBufferBinding::BindingType bindingType)
{
	VertexBufferBindingMap::iterator it = mBindings.find(bindingType);
	if (it == mBindings.end())
	{
		// Binding doesn't exist
		// TODO: Throw exception instead?
		return;
	}

	mBindings.erase(it);
}

void VertexBufferDeclaration::destroyBinding(VertexBufferBinding binding)
{
	this->destroyBinding(binding.getBindingType());
}

unsigned int VertexBufferDeclaration::getSizeInBytes() const
{
	unsigned int size = 0;

	VertexBufferBindingMap::const_iterator it = mBindings.begin();
	for(; it != mBindings.end(); ++it)
	{
		size += it->second.getSizeInBytes();
	}

	return size;
}

}
