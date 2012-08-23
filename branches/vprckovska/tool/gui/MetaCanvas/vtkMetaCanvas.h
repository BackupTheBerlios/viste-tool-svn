/*
 * vtkMetaCanvas.h
 *
 * 2005-01-12	Tim Peeters
 * - First version
 *
 * 2005-07-15	Tim Peeters
 * - Added SetInteractorStyle() function
 *
 * 2005-11-14	Tim Peeters
 * - Renamed from vtkMetaCanvasSelect to vtkMetaCanvas
 *
 * 2005-11-15	Tim Peeters
 * - Keep an InteractorStyle variable and override SetInteractor()
 *   from superclass so that an interactor style will automatically be
 *   applied to the new interactor.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 * 2011-03-01	Evert van Aart
 * - Metacanvas now emits an event when the user resets the camera ('R').
 *
 */

#ifndef bmia_vtkMetaCanvas_h
#define bmia_vtkMetaCanvas_h

#include "vtkGenericMetaCanvas.h"

namespace bmia {

class vtkMetaCanvasInteractorStyle;

/**
 * Metacanvas where one subcanvas can be selected.
 * Also has support for interactor styles to interact with the metacanvas and
 * the subcanvasses.
 */
class vtkMetaCanvas : public vtkGenericMetaCanvas
{
public:
  static vtkMetaCanvas* New();

  /**
   * Set/Get the interactor style.
   */
  void SetInteractorStyle(vtkMetaCanvasInteractorStyle* style);

  /**
   * Returns the currently selected subcanvas or NULL if no subcanvas
   * is selected.
   */
  vtkGetObjectMacro(SelectedSubCanvas, vtkAbstractSubCanvas);

  /**
   * Automatically enable self interaction when a subcanvas is selected and 
   * disable it when it is deselected.
   * When calling this function, interaction is disabled for all subcanvasses,
   * except for the selected one, if there is one selected and interact==true.
   */
  void SetInteractOnSelect(bool interact);
  vtkBooleanMacro(InteractOnSelect, bool);

  /**
   * Selects the subcanvas that is on the location where the last event
   * took place.
   */
  void SelectPokedSubCanvas();

	/** Emit a custom event that the camera of the selected subcanvas must be 
		reset. If this event is handled by a callback (like the one for the
		Plane Visualization Plugin), a "1" is returned; otherwise, a "0" 
		is returned. */

  int ResetCameraOfPokedSubCanvas();

	/** Called when the user double-clicks on a subcanvas. Used to maximize 
		subcanvasses or restore the initial layout. */
  
	void DoubleClickedOnCanvas();


  /**
   * If the subcanvas being removed was the selected subcanvas, make sure
   * it is deselected before it is removed.
   */
  virtual void RemoveSubCanvas(vtkAbstractSubCanvas* subcanvas);

  /**
   * Set the new interactor and if there was an interactor style, use it
   * for the new interactor as well.
   */
  virtual void SetInteractor(vtkRenderWindowInteractor* rwi);

protected:
  vtkMetaCanvas();
  ~vtkMetaCanvas();

  /**
   * Select the subcanvas at the specified location,
   * or deselect any subcanvas if there is no subcanvas there.
   */
  void SelectSubCanvasAt(int x, int y);

  /**
   * Selects the specified subcanvas.
   */
  void SelectSubCanvas(vtkAbstractSubCanvas* subcanvas);

  /**
   * The current interactor style.
   */
  vtkMetaCanvasInteractorStyle* InteractorStyle;

private:
  vtkAbstractSubCanvas* SelectedSubCanvas;
  bool InteractOnSelect;
};

} // namespace bmia

#endif
