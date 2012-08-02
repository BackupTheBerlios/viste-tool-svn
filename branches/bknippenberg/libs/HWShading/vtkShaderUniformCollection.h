/**
 * vtkShaderUniformCollection.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkShaderUniformCollection_h
#define bmia_vtkShaderUniformCollection_h

#include <vtkCollection.h>
#include "vtkShaderUniform.h" // for inline methods/static casts

namespace bmia {

class vtkShaderUniformCollection : public vtkCollection
{
public:
  static vtkShaderUniformCollection* New();

  /**
   * Add a ShaderObject to the list
   */
  void AddItem(vtkShaderUniform* su)
    {
    this->vtkCollection::AddItem((vtkObject*)su);
    };

  /**
   * Get the next ShaderUniform in the list. Return NULL when at the end of the 
   * list.
   */
  vtkShaderUniform *GetNextItem()
    {
    return static_cast<vtkShaderUniform *>(this->GetNextItemAsObject());
    };

  /**
   * Get the ith shader object from the list.
   */
  vtkShaderUniform* GetItem(int i)
    {
    return static_cast<vtkShaderUniform *>(this->GetItemAsObject(i));
    };

protected:
  vtkShaderUniformCollection() {};
  ~vtkShaderUniformCollection() {};

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(vtkObject *o) { this->vtkCollection::AddItem(o); };

};

} // namespace bmia

#endif // bmia_vtkShaderObjectCollection_h
