/**
 * vtkAbstractSubCanvas.h
 * by Tim Peeters
 *
 * 2005-01-12	Tim Peeters
 * - First version
 *
 * 2005-06-03	Tim Peeters
 * - Use namespace bmia
 *
 * 2006-02-18	Tim Peeters
 * - Added functions double GetRelative{X,Y}(int)
 */

#ifndef bmia_vtkAbstractSubCanvas_h
#define bmia_vtkAbstractSubCanvas_h

#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <string>

namespace bmia {

/**
 * Abstract class representing a subcanvas. A subcanvas is a part
 * of a metacanvas with its own rendering and interaction defined.
 */
class vtkAbstractSubCanvas : public vtkObject
// not a subclass of vtkViewport because that requires implementation of too
// many virtual functions. Instead give it a vtkViewport variable and
// a GetViewport function. Subclasses 
{
public:
  vtkTypeRevisionMacro(vtkAbstractSubCanvas,vtkObject);

  /**
   * Enable/Disable this subcanvas's own interaction.
   */
  virtual void SetInteract(bool i) = 0;
  vtkGetMacro(Interact, bool);
  vtkBooleanMacro(Interact, bool);

  /**
   * Specify the viewport for the subcanvas to draw in the rendering window.
   * Coordinates are expressed as (xmin, ymin, xmax, ymax), where each
   * coordinate is 0 <= coordinate <= 1.0.
   */
  virtual void SetViewport (double xmin, double ymin, double xmax, double ymax) = 0;
  virtual void SetViewport (double viewport[4])
    {
    this->SetViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }
  virtual double * GetViewport() = 0;
  virtual void GetViewport (double data[4])
    {
    double* viewport = this->GetViewport();
    for (int i=0; i < 4; i++) { data[i] = viewport[i]; }
    }

  // Description:
  // Is a given display point in this Viewport's viewport.
  virtual bool IsInViewport(int x, int y);

  /**
   * Get/Set the interactor to use for interaction,
   * if interaction is enabled.
   */
  void SetInteractor(vtkRenderWindowInteractor* i);
  vtkGetObjectMacro(Interactor, vtkRenderWindowInteractor);

  /**
   * Get/Set the render window to render on.
   */
  vtkSetObjectMacro(RenderWindow, vtkRenderWindow);
  vtkGetObjectMacro(RenderWindow, vtkRenderWindow);

  /**
   * Get the relative (0.0..1.0) coordinates inside the subcanvas from
   * the absolute x/y pixel coordinates in the renderwindow.
   */
  double GetRelativeX(int x);
  double GetRelativeY(int y);

  /**
   * Reset the camera in the subcanvas.
   */
  virtual void ResetCamera() {};

  std::string subCanvasName;

protected:
  vtkAbstractSubCanvas();
  virtual ~vtkAbstractSubCanvas();

  /**
   * Specifies whether this subcanvas's own interaction is enabled or
   * disabled.
   */
  bool Interact;

  vtkRenderWindowInteractor* Interactor;
  vtkRenderWindow* RenderWindow;

private:
  vtkAbstractSubCanvas(const vtkAbstractSubCanvas&);  // Not implemented.
  void operator=(const vtkAbstractSubCanvas&);  // Not implemented.

};

} // namespace bmia

#endif
