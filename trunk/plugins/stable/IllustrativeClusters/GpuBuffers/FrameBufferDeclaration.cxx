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
