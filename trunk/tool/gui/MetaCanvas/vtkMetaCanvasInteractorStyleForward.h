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
