/*
 * vtkMetaCanvasInteractorStyleForward.h
 *
 * 2005-01-06	Tim Peeters
 * - First version
 *
 * 2005-01-13	Tim Peeters
 * - Removed a lot of methods here that are no longer
 *   necessary due to changes in the metacanvas and subcanvas
 *   classes.
 *
 * 2006-10-03	Tim Peeters
 * - Made OnButtonDown() function virtual because I want to
 *   override it in vtkDTICanvasInteractorStyle.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 * 2011-02-21	Evert van Aart
 * - Maximizing subcanvasses can now also be done using F1-F5.
 *
 */

#ifndef bmia_vtkMetaCanvasInteractorStyleForward_h
#define bmia_vtkMetaCanvasInteractorStyleForward_h

#include "vtkMetaCanvasInteractorStyle.h"

#include <vtkTimerLog.h>

namespace bmia {

/**
 *  This metacanvas interactor style lets the events it receives be handled
 *  by the interactor style of the currently selected subcanvas. It also
 *  takes care of selecting different subcanvasses if the left mouse button
 *  is pushed over a subcanvas on the metacanvas.
 */
class vtkMetaCanvasInteractorStyleForward : public vtkMetaCanvasInteractorStyle
{
public:
  //vtkTypeRevisionMacro(vtkMetaCanvasInteractorStyle, vtkMetaCanvasInteractorStyleForward);
  static vtkMetaCanvasInteractorStyleForward* New();

  /**
   * Selects the subcanvas over which the event took place (if there is one),
   * and forwards all events (including the ButtonDown event) to that subcanvas.
   */

  virtual void OnKeyDown();
  virtual void OnLeftButtonDown();
  virtual void OnLeftButtonUp() 
  { 
  }
  virtual void OnMiddleButtonDown() { this->OnButtonDown(); }
  virtual void OnRightButtonDown() { 
	  this->OnButtonDown(); 
  }

  virtual void SetMetaCanvas(vtkMetaCanvas* metacanvas);

protected:
  vtkMetaCanvasInteractorStyleForward();
  ~vtkMetaCanvasInteractorStyleForward();

  /**
   * Call this if any of the mouse buttons is pressed.
   * Selects the subcanvas where the button was pressed.
   */
  virtual void OnButtonDown();

	/** Position of the mouse cursor at the time of the previous click. */
	int prevPos[2];

	/** Number of clicks of the LMB, used to detect double clicks. */
	int numberOfClicks;

	/** Timer used to detect double clicks. */
	vtkTimerLog * timer;

private:
  vtkMetaCanvasInteractorStyleForward(const vtkMetaCanvasInteractorStyleForward&);  // Not implemented.
  void operator=(const vtkMetaCanvasInteractorStyleForward&);  // Not implemented.

};

} // namespace bmia

#endif
