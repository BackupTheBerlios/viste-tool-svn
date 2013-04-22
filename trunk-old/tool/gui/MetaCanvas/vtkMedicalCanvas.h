/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
