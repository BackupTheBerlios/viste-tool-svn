/**
 * vtkMetaCanvasInteractorStyle.h
 * by Tim Peeters
 *
 * 2005-01-06	Tim Peeters
 * - First version.
 */

#ifndef bmia_vtkMetaCanvasInteractorStyle_h
#define bmia_vtkMetaCanvasInteractorStyle_h

#include <vtkInteractorStyle.h>
#include "vtkMetaCanvas.h"

namespace bmia {

/**
 * General for interaction on a MetaCanvas.
 */
class vtkMetaCanvasInteractorStyle : public vtkInteractorStyle
{
public:
  static vtkMetaCanvasInteractorStyle* New();
  vtkTypeRevisionMacro(vtkMetaCanvasInteractorStyle,vtkInteractorStyle);

  /**
   * Sets the metacanvas of this metacanvas interactor style to the
   * specified one.
   */
  virtual void SetMetaCanvas(vtkMetaCanvas* metacanvas);

protected:
  vtkMetaCanvasInteractorStyle();
  ~vtkMetaCanvasInteractorStyle();

  vtkGetObjectMacro(MetaCanvas, vtkMetaCanvas); 

  // hide for users/compiler. Make sure the interactor is set
  // through SetMetaCanvas only.
  virtual void SetInteractor(vtkRenderWindowInteractor* interactor)
  {
    this->vtkInteractorStyle::SetInteractor(interactor);
  }

private:
  /**
   * The current MetaCanvas this interactor style is operating on.
   */
  vtkMetaCanvas* MetaCanvas;

  vtkMetaCanvasInteractorStyle(const vtkMetaCanvasInteractorStyle&);  // Not implemented.
  void operator=(const vtkMetaCanvasInteractorStyle&);  // Not implemented.
};

} // namespace bmia

#endif
