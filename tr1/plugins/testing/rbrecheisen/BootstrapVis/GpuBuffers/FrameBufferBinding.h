/*
 * FrameBufferBinding.h
 *
 * 2009-04-08	Ron Otten
 * - First version
 */

#ifndef opengl_FrameBufferBinding_h
#define opengl_FrameBufferBinding_h

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

class FrameBufferBinding
{
public:
	enum BindingType
	{
		BND_COLOR_ATTACHMENT = GL_COLOR_ATTACHMENT0_EXT,
		BND_DEPTH_ATTACHMENT = GL_DEPTH_ATTACHMENT_EXT
	};

	enum DataType
	{
		ELM_FLOAT = GL_FLOAT,
		ELM_UNSIGNED_BYTE = GL_UNSIGNED_BYTE
	};

	FrameBufferBinding(BindingType bindingType, DataType elementType, unsigned int textureUnit);

	virtual ~FrameBufferBinding();

	inline BindingType getBindingType() const
	{
		return mBindingType;
	}
	inline DataType getDataType() const
	{
		return mDataType;
	}
	inline unsigned int getTextureUnit() const
	{
		return mTextureUnit;
	}

private:
	BindingType mBindingType;
	DataType mDataType;
	unsigned int mTextureUnit;
};

}

#endif /* FRAMEBUFFERBINDING_H_ */
