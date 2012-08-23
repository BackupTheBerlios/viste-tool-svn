/**
 * vtkLookupTableCollection.h
 *
 * 2006-04-06	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkLookupTableCollection_h
#define bmia_vtkLookupTableCollection_h

#include <vtkCollection.h>
class vtkLookupTable;

namespace bmia {

class vtkLookupTableCollection : public vtkCollection
{
public:
  static vtkLookupTableCollection* New();

  /**
   * Add a lookup table to the list
   */
  void AddItem(vtkLookupTable* lut);

  /**
   * Get the next lookup table from the list. NULL is returned when the
   * collection is exhausted.
   */
  vtkLookupTable* GetNextItem();

  /**
   * Get the ith lookup table in the list.
   */
  vtkLookupTable *GetItem(int i);

protected:
   vtkLookupTableCollection();
   ~vtkLookupTableCollection();

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(vtkObject *o) { this->vtkCollection::AddItem(o); };

}; // class vtkLookupTableCollection

} // namespace bmia

#endif // bmia_vtkLookupTableCollection_h
