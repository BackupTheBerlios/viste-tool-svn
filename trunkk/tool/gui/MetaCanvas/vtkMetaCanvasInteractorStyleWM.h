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
