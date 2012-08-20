/**
 * vtkMetaCanvasInteractorStyleSwitch.h
 *
 * 2005-01-07	Tim Peeters
 * - First version.
 *
 * 2011-03-01	Evert van Aart
 * - "OnChar" now also listens to the "R"-key. This allows us to overwrite the
 *   camera-resetting behavior with our own. This is required for the planes in
 *   the 2D views.
 *
 */

#ifndef bmia_vtkMetaCanvasInteractorStyleSwitch_h
#define bmia_vtkMetaCanvasInteractorStyleSwitch_h

#include "vtkMetaCanvasInteractorStyle.h"

namespace bmia {

#define VTK_MCIS_FORWARD	0	// Forward events to subcanvasses
#define VTK_MCIS_WM		1	// Window manager style

class vtkMetaCanvasInteractorStyleForward;
class vtkMetaCanvasInteractorStyleWM;

/**
 * This class makes allows interactive switching between two
 * styles: metacanvas interactor style WM and metacanvas interactor
 * style Forward by pressing the 'm' key.
 */
class vtkMetaCanvasInteractorStyleSwitch : public vtkMetaCanvasInteractorStyle
{
public:
  static vtkMetaCanvasInteractorStyleSwitch *New();

  void SetMetaCanvas(vtkMetaCanvas* metacanvas);

  /**
   * Only care about the char event, which is used to switch between different
   * styles.
   */
  virtual void OnChar();

protected:
  vtkMetaCanvasInteractorStyleSwitch();
  ~vtkMetaCanvasInteractorStyleSwitch();

  vtkMetaCanvasInteractorStyleForward* StyleForward;
  vtkMetaCanvasInteractorStyleWM* StyleWM;

  int CurrentStyleNr;
  vtkMetaCanvasInteractorStyle* CurrentStyle;

  void SetCurrentStyle();

private:
  vtkMetaCanvasInteractorStyleSwitch(const vtkMetaCanvasInteractorStyleSwitch&);  // Not implemented.
  void operator=(const vtkMetaCanvasInteractorStyleSwitch&);  // Not implemented.

};

} // namespace bmia

#endif
