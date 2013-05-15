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
