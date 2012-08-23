/**
 * vtkCylinderSubCanvas.h
 * by Tim Peeters
 *
 * 2005-01-10	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkCylinderSubCanvas_h
#define bmia_vtkCylinderSubCanvas_h

#include "vtkSubCanvas.h"

class vtkCylinderSource;

namespace bmia {

/**
 * Example class showing how to subclass vtkSubCanvas.
 */
class vtkCylinderSubCanvas : public vtkSubCanvas
{
public:
  vtkTypeRevisionMacro(vtkCylinderSubCanvas, vtkSubCanvas);
  static vtkCylinderSubCanvas *New();

  void SetCylinderResolution(int resolution);
  int GetCylinderResolution();

protected:
  vtkCylinderSubCanvas();
  ~vtkCylinderSubCanvas();

private:
  vtkCylinderSource* CylinderSource;

};

} // namespace bmia

#endif
