/**
 * vtkLookupTableCollection.h
 *
 * 2006-04-06	Tim Peeters
 * - First version
 */

#include "vtkLookupTableCollection.h"
#include <vtkLookupTable.h>
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkLookupTableCollection);

vtkLookupTableCollection::vtkLookupTableCollection()
{
  // nothing to do
}

vtkLookupTableCollection::~vtkLookupTableCollection()
{
  // emtpy
}

void vtkLookupTableCollection::AddItem(vtkLookupTable* lut)
{
  this->vtkCollection::AddItem((vtkObject*)lut);
}

vtkLookupTable* vtkLookupTableCollection::GetNextItem()
{
  return static_cast<vtkLookupTable *>(this->GetNextItemAsObject());
}

vtkLookupTable* vtkLookupTableCollection::GetItem(int i)
{
  return static_cast<vtkLookupTable *>(this->GetItemAsObject(i));;
}

} // namespace bmia
