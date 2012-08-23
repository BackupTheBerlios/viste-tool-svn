/*
 * vtkMedicalCanvas.h
 *
 * 2010-09-16	Tim Peeters
 * - First version.
 *
 * 2011-02-28	Evert van Aart
 * - Added support for setting the background color with or without gradient.
 *
 */

#ifndef bmia_vtkMedicalCanvas_h
#define bmia_vtkMedicalCanvas_h

#include "vtkMetaCanvas.h"

namespace bmia {

class vtkSubCanvas;

/**
 * The vtkMedicalCanvas is a subclass of vtkMetaCanvas that
 * provides one large 3D view and 3 smaller 2D views. Its goal
 * is to be used as a canvas for viewing medical data, with
 * the 2D views showing cross-sections of the 3D volume rendered
 * in the 3D view.
 *
 * TODO: add support for changing the layout.
 * TODO: add support for changing interactor styles??
 */
class vtkMedicalCanvas : public vtkMetaCanvas
{
public:
  static vtkMedicalCanvas* New();

  /**
   * Return the renderer of the 3D SubCanvas.
   */
  vtkRenderer* GetRenderer3D();

  /**
   * Return the big subcanvas, meant for 3D visualization.
   */
  vtkSubCanvas* GetSubCanvas3D();

  /**
   * Return one of the three smaller subcanvasses, meant for
   * slice visualization in 2D.
   */
  vtkSubCanvas* GetSubCanvas2D(int i);

	/** Set the background color and turn off the gradient. 
		@param r	Primary color (red). 
		@param g	Primary color (green).
		@param b	Primary color (blue). */

	void setBackgroundColor(double r, double g, double b);

	/** Set the background gradient, using two input RGB colors.
	@param r1	Primary color (red). 
	@param g1	Primary color (green).
	@param b1	Primary color (blue). 
	@param r2	Secondary color (red). 
	@param g2	Secondary color (green).
	@param b2	Secondary color (blue). */

	void setGradientBackground(double r1, double g1, double b1, double r2, double g2, double b2);


protected:
  vtkMedicalCanvas();
  ~vtkMedicalCanvas();

private:
  vtkSubCanvas* SubCanvas3D;
  vtkSubCanvas* SubCanvas2D[3];

}; // class vtkMedicalCanvas
} // namespace bmia
#endif // bmia_vtkMedicalCanvas_h
