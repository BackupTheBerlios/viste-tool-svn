/**
 * vtkSubCanvas.h
 *
 * 2005-01-05	Tim Peeters
 * - First version
 *
 * 2005-01-12	Tim Peeters
 * - Made vtkSubCanvas a subclass of vtkAbstractSubCanvas
 *   and implemented SetInteract() and the unimplemented
 *   methods from vtkViewport.
 *
 * 2005-07-15	Tim Peeters
 * - Added Render().
 *
 * 2005-11-15	Tim Peeters
 * - Added SetBackground(r, g, b).
 *
 * 2006-02-20	Tim Peeters
 * - Made GetRenderer() function public instead of protected.
 *   This function is used by QDTIWindow to pass the renderer to
 *   QDTITrackingWidget so that it can add actors to the renderer.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 */

#ifndef bmia_vtkSubCanvas_h
#define bmia_vtkSubCanvas_h

#include "vtkAbstractSubCanvas.h"
#include <vtkRenderer.h>
#include <vtkInteractorStyle.h>

namespace bmia {

/**
 * Class for representing a subcanvas on a metacanvas.
 * Each subcanvas has its own renderer (and thus viewport)
 * and own interaction defined by a vtkInteractorStyle.
 */
class vtkSubCanvas : public vtkAbstractSubCanvas
{
public:
  vtkTypeRevisionMacro(vtkSubCanvas,vtkAbstractSubCanvas);
  /**
   * Creates a new subcanvas with no interactor style set and
   * a new (standard) renderer.
   */
  static vtkSubCanvas *New();

  /**
   * Enable/disable interaction for this subcanvas using the subcanvas's
   * interactorstyle. Implemented pure virtual function from
   * vtkAbstractSubCanvas.
   */
  virtual void SetInteract(bool i);

  virtual void SetViewport(double xmin, double ymin, double xmax, double ymax);
  virtual double* GetViewport();

  virtual void SetRenderWindow(vtkRenderWindow* rw);

  /**
   * (Re)renders this subcanvas by calling this->Renderer->Render();
   */
  virtual void Render();

  /**
   * Set the background color.
   */
  virtual void SetBackground(double r, double g, double b)
    {
    this->GetRenderer()->SetBackground(r, g, b);
    }

  vtkGetObjectMacro(Renderer, vtkRenderer);

  /**
   * Reset the camera of the subcanvas. Usually this is done by simply
   * resetting the camera of the renderer.
   */
  virtual void ResetCamera()
    {
    this->Renderer->ResetCamera();
    }

  void SetInteractorStyle(vtkInteractorStyle* istyle);
  vtkGetObjectMacro(InteractorStyle, vtkInteractorStyle);

	/** Return the viewport that was set when "SetViewPort" was called for
		the first time. All values are zero if "SetViewPort" has not yet
		been called. 
		@param vp	Four-element output array. */

	void getFirstViewPort(double * vp);

protected:

  vtkSubCanvas();
  ~vtkSubCanvas();

private:
  vtkSubCanvas(const vtkSubCanvas&);  // Not implemented.
  void operator=(const vtkSubCanvas&);  // Not implemented.

  vtkRenderer* Renderer;
  vtkInteractorStyle* InteractorStyle;

	/** True until "SetViewPort" is called for the first time. */
	
	bool firstSetViewport;

	/** Contains the viewport values used in the first call of "SetViewPort".
		If this function has not yet been called, all values will be zero. */

	double firstViewport[4];

};

} // namespace bmia

#endif
