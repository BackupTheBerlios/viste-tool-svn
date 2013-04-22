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
 * VertexBuffer.cpp
 *
 * 2009-02-19	Ron Otten
 * - First version
 */

#include "VertexBuffer.h"

namespace opengl
{

VertexBuffer::VertexBuffer() :
	mBufferHandle(0), mBytesPerElement(0)
{
}

VertexBuffer::VertexBuffer(VertexBufferDeclaration declaration) :
	mBufferHandle(0), mBytesPerElement(0)
{
	this->declare(declaration);
}

VertexBuffer::~VertexBuffer()
{
	if (glIsBuffer(mBufferHandle) == GL_TRUE)
	{
		this->unbind();
		this->feedbackUnbind();

		glDeleteBuffers(1, &mBufferHandle);
	}
}

void VertexBuffer::declare(VertexBufferDeclaration declaration)
{
	if (glIsBuffer(mBufferHandle) != GL_FALSE)
	{
		return;
	}

	glGenBuffers(1, &mBufferHandle);
	mDeclaration = declaration;
	mBytesPerElement = mDeclaration.getSizeInBytes();
}

void VertexBuffer::fill(void* data, unsigned int nrElements)
{
	if (glIsBuffer(mBufferHandle) != GL_TRUE)
	{
		return;
	}

	GLint originalBufferHandle = 0;
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &originalBufferHandle);

	glBindBuffer(GL_ARRAY_BUFFER, mBufferHandle);

	glBufferData(GL_ARRAY_BUFFER, nrElements * mBytesPerElement, data,
			GL_DYNAMIC_COPY);

	glBindBuffer(GL_ARRAY_BUFFER, originalBufferHandle);
}

void VertexBuffer::fillRange(void* data, unsigned int nrElements,
		unsigned int elementOffset)
{
	if (glIsBuffer(mBufferHandle) != GL_TRUE)
	{
		return;
	}

	GLint originalBufferHandle = 0;
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &originalBufferHandle);

	glBindBuffer(GL_ARRAY_BUFFER, mBufferHandle);

	glBufferSubData(GL_ARRAY_BUFFER, elementOffset * mBytesPerElement,
			nrElements * mBytesPerElement, data);

	glBindBuffer(GL_ARRAY_BUFFER, originalBufferHandle);
}

void VertexBuffer::bind(unsigned int elementStride, unsigned int elementOffset)
{
	if (glIsBuffer(mBufferHandle) != GL_TRUE)
	{
		return;
	}

	glBindBuffer(GL_ARRAY_BUFFER, mBufferHandle);

	GLintptr byteOffset = mBytesPerElement * elementOffset;
	GLintptr byteStride = mBytesPerElement * elementStride;

	GLintptr accumOffset = 0;
	VertexBufferDeclaration::VertexBufferBindingEnumerator bindings =
			mDeclaration.getBindingsEnumerator();
	while (bindings.moveNext())
	{
		VertexBufferBinding binding = bindings.getCurrentValue();

		GLenum dataType = static_cast<GLenum>(binding.getElementType());

		switch (binding.getBindingType())
		{
		case VertexBufferBinding::BND_VERTEX:
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(binding.getNrElements(), dataType, byteStride,
					reinterpret_cast<GLvoid*> (byteOffset + accumOffset));
			break;
		case VertexBufferBinding::BND_COLOR:
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(binding.getNrElements(), dataType, byteStride,
					reinterpret_cast<GLvoid*> (byteOffset + accumOffset));
			break;
		default:
			break;
		}

		accumOffset += binding.getSizeInBytes();
	}
}

void VertexBuffer::unbind()
{
	if (glIsBuffer(mBufferHandle) != GL_TRUE)
	{
		return;
	}

	GLint currentBufferHandle = 0;
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currentBufferHandle);

	if (static_cast<GLuint> (currentBufferHandle) == mBufferHandle)
	{
		VertexBufferDeclaration::VertexBufferBindingEnumerator bindings =
				mDeclaration.getBindingsEnumerator();

		while (bindings.moveNext())
		{
			VertexBufferBinding binding = bindings.getCurrentValue();
			switch (binding.getBindingType())
			{
			case VertexBufferBinding::BND_VERTEX:
				glDisableClientState(GL_VERTEX_ARRAY);
				break;
			case VertexBufferBinding::BND_COLOR:
				glDisableClientState(GL_COLOR_ARRAY);
				break;
			default:
				break;
			}
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void VertexBuffer::feedbackBind()
{
	if (glIsBuffer(mBufferHandle) != GL_TRUE)
	{
		return;
	}

	VertexBufferDeclaration::VertexBufferBindingEnumerator bindings =
			mDeclaration.getBindingsEnumerator();

	unsigned int nrBindings = mDeclaration.getNrBindings();
	GLint* attribs = new GLint[nrBindings * 3];
	GLint* attribsPtr = attribs;

	while (bindings.moveNext())
	{
		VertexBufferBinding binding = bindings.getCurrentValue();
		*attribsPtr = static_cast<GLenum>(binding.getBindingType());
		attribsPtr++;

		*attribsPtr = binding.getNrElements();
		attribsPtr++;

		*attribsPtr = 0; // Everything interleaved into the 1st buffer
		attribsPtr++;
	}

	glTransformFeedbackAttribsNV(nrBindings, attribs, GL_INTERLEAVED_ATTRIBS_NV);
	glBindBufferBaseNV(GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, mBufferHandle);
}

void VertexBuffer::feedbackUnbind()
{
	glBindBufferBaseNV(GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0);
}

}
