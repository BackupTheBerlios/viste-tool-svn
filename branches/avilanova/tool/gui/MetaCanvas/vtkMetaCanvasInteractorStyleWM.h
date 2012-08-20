/**
 * vtkMetaCanvasInteractorStyleWM.h
 * by Tim Peeters
 *
 * 2005-01-06	TimP	First version
 */

#ifndef bmia_vtkMetaCanvasInteractorStyleWM_h
#define bmia_vtkMetaCanvasInteractorStyleWM_h

#include "vtkMetaCanvasInteractorStyle.h"

namespace bmia {

#define VTK_MCMS_NOTHING	0	// not manipulating a subcanvas
#define VTK_MCMS_MOVING		1	// moving a subcanvas
#define VTK_MCMS_RESIZING	2	// resizing a subcanvas

// Metacanvas viewport corner.
#define VTK_MCVC_LOWER_LEFT	0
#define VTK_MCVC_LOWER_RIGHT	1
#define VTK_MCVC_UPPER_LEFT	2
#define VTK_MCVC_UPPER_RIGHT	3

/**
 * VTK meta canvas interactor style that acts like a "Window Manager" and can
 * be used to manipulate the various subcanvasses on the meta canvas by
 * dragging and resizing them.
 */
class vtkMetaCanvasInteractorStyleWM : public vtkMetaCanvasInteractorStyle
{
public:
  static vtkMetaCanvasInteractorStyleWM* New();

  virtual void OnLeftButtonDown();
  virtual void OnLeftButtonUp();
  virtual void OnRightButtonDown();
  virtual void OnRightButtonUp();
  virtual void OnMouseMove();

  virtual void SetMetaCanvas(vtkMetaCanvas* metacanvas);

protected:
  vtkMetaCanvasInteractorStyleWM();
  ~vtkMetaCanvasInteractorStyleWM();

private:
  vtkMetaCanvasInteractorStyleWM(const vtkMetaCanvasInteractorStyleWM&);  // Not implemented.
  void operator=(const vtkMetaCanvasInteractorStyleWM&);  // Not implemented.

  void SetCurrentManipulationStyle(int mstyle);

  /**
   * The current style of interaction. Either VTK_MCIS_MOVING,
   * VTK_MCIS_RESIZING, or VTK_MCIS_NOTHING.
   */
  int CurrentManipulationStyle;

  /**
   * The renderer of the subcanvas that is currently being manipulated.
   * NULL if none is being manipulated (if CurrentManipulationSTyle == VTK_MCMS_NOTHING).
   */
  vtkAbstractSubCanvas* ManipulatedSubCanvas;

  /**
   * Returns the corner of the specified viewport in which the given coordinate is.
   */
  int GetViewportCorner(double* viewport, double rx, double ry);

  /**
   * The current viewport corner. Useful when resizing.
   */
  int CurrentViewportCorner;

  /**
   * Set CurrentViewportCorner using GetViewportCorner();
   */
  void DetectCurrentViewportCorner();

};

} // namespace bmia

#endif
