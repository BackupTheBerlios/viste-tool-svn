/**
 * vtkShaderObjectCollection.h
 * by Tim Peeters
 *
 * 2005-05-04	Tim Peeters	First version
 */

#ifndef bmia_vtkShaderObjectCollection_h
#define bmia_vtkShaderObjectCollection_h

#include <vtkCollection.h>
#include "vtkShaderObject.h" // for inline methods/static casts

namespace bmia {

class vtkShaderObjectCollection : public vtkCollection
{
public:
  static vtkShaderObjectCollection* New();

  /**
   * Add a ShaderObject to the list
   */
  void AddItem(vtkShaderObject* so)
    {
    this->vtkCollection::AddItem((vtkObject*)so);
    };

  /**
   * Get the next ShaderObject in the list. Return NULL when at the end of the 
   * list.
   */
  vtkShaderObject *GetNextItem()
    {
    return static_cast<vtkShaderObject *>(this->GetNextItemAsObject());
    };

  /**
   * Get the ith shader object from the list.
   */
  vtkShaderObject* GetItem(int i)
    {
    return static_cast<vtkShaderObject *>(this->GetItemAsObject(i));
    };

protected:
  vtkShaderObjectCollection() {};
  ~vtkShaderObjectCollection() {};

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(vtkObject *o) { this->vtkCollection::AddItem(o); };

};

} // namespace bmia

#endif // bmia_vtkShaderObjectCollection_h
