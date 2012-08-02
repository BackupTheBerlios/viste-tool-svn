/*
 * vtkAbstractSubCanvasCollection.h
 *
 * 2005-01-12	Tim Peeters
 * - First version
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 */

#ifndef bmia_vtkAbstractSubCanvasCollection_h
#define bmia_vtkAbstractSubCanvasCollection_h

#include <vtkCollection.h>
#include "vtkAbstractSubCanvas.h"

namespace bmia {

class vtkAbstractSubCanvasCollection : public vtkCollection
{
public:
  vtkTypeRevisionMacro(vtkAbstractSubCanvasCollection, vtkCollection);
  static vtkAbstractSubCanvasCollection *New();

  // Description:
  // Add a dataset to the list.
  void AddItem(vtkAbstractSubCanvas *sc)
  {
    this->vtkCollection::AddItem((vtkObject *)sc);
  };
  
  // Description:
  // Get the next subcanvas in the list.
  vtkAbstractSubCanvas *GetNextItem()
  { 
    return static_cast<vtkAbstractSubCanvas *>(this->GetNextItemAsObject());
  };

  // Description:
  // Get the ith subcanvas in the list.
  vtkAbstractSubCanvas *GetItem(int i)
  { 
    return static_cast<vtkAbstractSubCanvas *>(this->GetItemAsObject(i));
  };

	/** Index of the currently maximized subcanvas. */
  
	int maximizedSubCanvas;
  
protected:
  vtkAbstractSubCanvasCollection();
  ~vtkAbstractSubCanvasCollection() {};

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(vtkObject *o) { this->vtkCollection::AddItem(o); };

private:
  vtkAbstractSubCanvasCollection(const vtkAbstractSubCanvasCollection&);  // Not implemented.
  void operator=(const vtkAbstractSubCanvasCollection&);  // Not implemented.

};

} // namespace bmia

#endif
