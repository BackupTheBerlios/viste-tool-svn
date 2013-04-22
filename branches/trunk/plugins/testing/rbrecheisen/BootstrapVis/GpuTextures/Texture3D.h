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
