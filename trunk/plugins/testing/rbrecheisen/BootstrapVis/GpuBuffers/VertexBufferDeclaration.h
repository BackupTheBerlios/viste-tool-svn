/*
 * VertexBufferDeclaration.h
 *
 * 2009-02-19	Ron Otten
 * - First version
 */

#ifndef opengl_VertexBufferDeclaration_h
#define opengl_VertexBufferDeclaration_h

#include <map>
#include <vector>
#include "VertexBufferBinding.h"
#include "../Enumerators/MapEnumerator.h"

namespace opengl
{

/** A declaration of the structure for an OpenGL vertex buffer object.
 * 	@remarks
 * 		Consumed by instances of the VertexBuffer class.
 */
class VertexBufferDeclaration
{
public:
	/** Represents a mapping from a type of buffer binding to the actual VertexBufferBinding instance. */
	typedef std::map<VertexBufferBinding::BindingType, VertexBufferBinding> VertexBufferBindingMap;

	/** Allows enumeration over a VertexBufferBindingMap without public exposure of the map's iterators. */
	typedef MapEnumerator<VertexBufferBindingMap> VertexBufferBindingEnumerator;

	/** Creates a new VertexBufferDeclaration. */
	VertexBufferDeclaration();

	/** Destroys the VertexBufferDeclaration. */
	virtual ~VertexBufferDeclaration();

	/** Create a new buffer binding for the VertexBufferDeclaration.
	 * 	@param bindingType
	 * 		The type of buffer binding.
	 * 	@param elementType
	 * 		The type of data element the binding consists of.
	 * 	@param nrElements
	 * 		The number of data elements one buffer entry consists of.
	 * 	@return
	 * 		The newly created binding.
	 */
	VertexBufferBinding createBinding(VertexBufferBinding::BindingType bindingType, VertexBufferBinding::ElementType elementType, unsigned int nrElements);

	/** Removes a specific buffer binding from the VertexBufferDeclaration and destroys it.
	 * 	@param bindingType
	 * 		The buffer binding type to remove and destroy.
	 */
	void destroyBinding(VertexBufferBinding::BindingType bindingType);

	/** Removes a specific buffer binding from the VertexBufferDeclaration and destroys it.
	 * 	@param binding
	 * 		The buffer binding to remove and destroy.
	 */
	void destroyBinding(VertexBufferBinding binding);

	/** Retrieves a previously created buffer binding from the VertexBufferDeclaration.
	 * 	@param bindingType
	 * 		The type of the binding to retrieve.
	 * 	@return
	 * 		The requested binding.
	 */
	VertexBufferBinding getBinding(VertexBufferBinding::BindingType bindingType);

	/** Get an enumerator over the collection of buffer bindings
	 * 	the VertexBufferDeclaration contains.
	 *  @return
	 *  	The enumerator over the collection.
	 */
	inline VertexBufferBindingEnumerator getBindingsEnumerator()
	{
		return VertexBufferBindingEnumerator(mBindings);
	}

	/** Gets the number of buffer bindings the
	 * 	VertexBufferDeclaration contains.
	 * 	@return
	 * 		The number of bindings present.
	 */
	inline unsigned int getNrBindings()
	{
		return mBindings.size();
	}

	/** Gets the amount of bytes one buffer element in a buffer
	 * 	structured according to the VertexBufferDeclaration
	 * 	would take.
	 * 	@return
	 * 		The number of bytes.
	 */
	unsigned int getSizeInBytes() const;

protected:

private:
	VertexBufferBindingMap mBindings;
};

}

#endif /* VERTEXBUFFERDECLARATION_H_ */
