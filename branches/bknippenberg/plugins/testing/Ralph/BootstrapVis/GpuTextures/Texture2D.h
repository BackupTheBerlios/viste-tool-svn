/*
 * Texture2D.h
 *
 * 2009-02-24	Ron Otten
 * - First version
 */

#ifndef opengl_Texture2D_h
#define opengl_Texture2D_h

#include <GL/glew.h>
#include <GL/gl.h>

#include "TextureDeclaration.h"

namespace opengl
{

/** A high level representation of an OpenGL 2D texture. */
class Texture2D
{
public:
	Texture2D();
	Texture2D(TextureDeclaration declaration);
	virtual ~Texture2D();

	/** Creates the 2D texture and declares its format.
	 *  @remarks
	 *  	May only be called once and only if the parameterless constructor
	 *      was used to create the Texture2D instance.
	 *  @param declaration
	 *  	The declaration of the texture.
	 */
	void declare(TextureDeclaration declaration);

	/** Fills the 2D texture with data according to the given dimensions
	 *  @param data
	 *  	The texture's raw data, in accordance with the pixel format
	 *      and data type passed in with the texture's TextureDeclaration.
 	 *	@param xDim
 	 *		The X dimension of the texture.
 	 *	@param yDim
 	 *		The Y dimension of the texture.
 	 *	@return
 	 *		Whether the texture could be succesfully stored.
	 */
	bool fill(void* data, unsigned int xDim, unsigned int yDim);

	/** Binds the texture. */
	void bind();

	/** Unbinds the texture. */
	void unbind();

protected:
	friend class FrameBuffer;
	inline GLuint getHandle() const
	{
		return mTextureHandle;
	}

private:
	GLuint mTextureHandle;
	TextureDeclaration mDeclaration;
	unsigned int mDimensions[2];
};

}

#endif /* TEXTURE3D_H_ */
