/*
 * VertexBufferBinding.h
 *
 * 2009-02-19	Ron Otten
 * - First version
 */

#ifndef opengl_VertexBufferBinding_h
#define opengl_VertexBufferBinding_h

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

/** A binding target (e.g., position, color, etc.) for an OpenGL vertex buffer object.
 * 	@remarks
 *		Consumed by instances of the VertexBufferDeclaration class.
 */
class VertexBufferBinding
{
public:
	/** Enumerates the possible binding targets for OpenGL vertex buffer objects. */
	enum BindingType
	{
		BND_VERTEX = GL_POSITION,
		BND_COLOR = GL_PRIMARY_COLOR
	};

	/** Enumerates the possible data storage type for the binding target's elements. */
	enum ElementType
	{
		ELM_FLOAT = GL_FLOAT
	};

	/** Creates a new VertexBufferBinding
	 *	@param bindingType
	 *		The type of target to bind to.
	 *	@param elementType
	 *		The data storage type for the binding's elements.
	 *	@param nrElements
	 *		How many elements / components a single vertex will take
	 *		in the binding.
	 */
	VertexBufferBinding(BindingType bindingType, ElementType elementType,
			unsigned int nrElements);

	/** Destroys the VertexBufferBinding */
	virtual ~VertexBufferBinding();

	/** Gets the target the binding is bound to.
	 *	@return
	 *		The bound target.
	 */
	inline BindingType getBindingType() const
	{
		return mBindingType;
	}

	/** Gets the data storage type for the binding's elements.
	 *	@return
	 *		The data storage type.
	 */
	inline ElementType getElementType() const
	{
		return mElementType;
	}

	/** Gets the number of elements in the binding one vertex takes.
	 *	@return
	 *		The number of elements for one vertex.
	 */
	inline unsigned int getNrElements() const
	{
		return mNrElements;
	}

	/** Gets the amount of bytes this binding would take to store
	 *  one buffer element (vertex).
	 * 	@return
	 * 		The number of bytes.
	 */
	unsigned int getSizeInBytes() const;

private:
	BindingType mBindingType;
	ElementType mElementType;
	unsigned int mNrElements;
};

}

#endif /* VERTEXBUFFERBINDING_H_ */
