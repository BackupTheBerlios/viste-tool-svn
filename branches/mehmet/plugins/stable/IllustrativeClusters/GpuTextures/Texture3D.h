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
 * Texture3D.h
 *
 * 2009-02-24	Ron Otten
 * - First version
 */

#ifndef opengl_Texture3D_h
#define opengl_Texture3D_h

#include <GL/glew.h>
#include <GL/gl.h>

#include "TextureDeclaration.h"

namespace opengl
{

/** A high level representation of an OpenGL 3D texture. */
class Texture3D
{
public:
	Texture3D();
	Texture3D(TextureDeclaration declaration);
	virtual ~Texture3D();

	/** Creates the 3D texture and declares its format.
	 *  @remarks
	 *  	May only be called once and only if the parameterless constructor
	 *      was used to create the Texture3D instance.
	 *  @param declaration
	 *  	The declaration of the texture.
	 */
	void declare(TextureDeclaration declaration);

	/** Fills the 3D texture with data according to the given dimensions
	 *  @param data
	 *  	The texture's raw data, in accordance with the pixel format
	 *      and data type passed in with the texture's TextureDeclaration.
 	 *	@param xDim
 	 *		The X dimension of the texture.
 	 *	@param yDim
 	 *		The Y dimension of the texture.
 	 *	@param zDim
 	 *		The Z dimension of the texture.
 	 *	@eturn
 	 *		Whether the texture could be succesfully stored.
	 */
	bool fill(void* data, unsigned int xDim, unsigned int yDim, unsigned int zDim);

	/** Binds the texture. */
	void bind();

	/** Unbinds the texture. */
	void unbind();

private:
	GLuint mTextureHandle;
	TextureDeclaration mDeclaration;
	unsigned int mDimensions[3];
};

}

#endif /* TEXTURE3D_H_ */
