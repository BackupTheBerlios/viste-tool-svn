/*
 * vtkAbstractSubCanvasCollection.cxx
 *
 * 2005-01-12	Tim Peeters
 * - First version
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 */

#include "vtkAbstractSubCanvasCollection.h"
#include "vtkObjectFactory.h"

namespace bmia {
  vtkStandardNewMacro(vtkAbstractSubCanvasCollection);
  vtkCxxRevisionMacro(vtkAbstractSubCanvasCollection, "$Revision: 0.1 $");

  vtkAbstractSubCanvasCollection::vtkAbstractSubCanvasCollection()
  {
	  // No subcanvas is maximized on startup.
	  this->maximizedSubCanvas = -1;
  }
}
