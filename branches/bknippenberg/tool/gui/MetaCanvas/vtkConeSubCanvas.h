/**
 * vtkConeSubCanvas.h
 * by Tim Peeters
 *
 * 2005-01-10	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkConeSubCanvas_h
#define bmia_vtkConeSubCanvas_h

#include "vtkSubCanvas.h"

namespace bmia {

/**
 * Example class showing how to subclass vtkSubCanvas.
 */
class vtkConeSubCanvas : public vtkSubCanvas
{
public:
  //vtkTypeRevisionMacro(vtkSubCanvas, vtkConeSubCanvas);
  static vtkConeSubCanvas *New();

protected:
  vtkConeSubCanvas();
  ~vtkConeSubCanvas();

};

} // namespace bmia

#endif
