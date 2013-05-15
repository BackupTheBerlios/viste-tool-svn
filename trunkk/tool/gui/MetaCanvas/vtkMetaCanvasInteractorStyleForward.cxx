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
 * vtkMetaCanvasInteractorStyleForward.cxx
 *
 * 2005-01-06	Tim Peeters
 * - First version
 *
 * 2005-01-11	Tim Peeters
 * - Reorganized and added SetCurrentInteractorStyle()
 *   function (without parameters)
 * 
 * 2005-01-13	Tim Peeters
 * - A lot of simplifications because of changes in
 *   metacanvas and subcanvas classes/functionality.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 * 2011-02-21	Evert van Aart
 * - Maximizing subcanvasses can now also be done using F1-F5.
 *
 */

#include "vtkMetaCanvasInteractorStyleForward.h"
#include <vtkObjectFactory.h> // for vtkStandardNewMacro
#include <QString>
#include <assert.h>

namespace bmia {

vtkStandardNewMacro(vtkMetaCanvasInteractorStyleForward);
//vtkCxxRevisionMacro(vtkMetaCanvasInteractorStyleForward, "$Revision 0.1 $");

vtkMetaCanvasInteractorStyleForward::vtkMetaCanvasInteractorStyleForward()
{
	// Set default values
	this->prevPos[0] = -1000;
	this->prevPos[1] = -1000;
	this->numberOfClicks = 0;

	// Create a timer and start it
	this->timer = vtkTimerLog::New();
	this->timer->StartTimer();
}

vtkMetaCanvasInteractorStyleForward::~vtkMetaCanvasInteractorStyleForward()
{
  // nothing
}


void vtkMetaCanvasInteractorStyleForward::OnKeyDown()
{
	// Get the key symbol
	char * key = this->Interactor->GetKeySym();

	// Convert to a QString for easy handling
	QString QKey = QString(key);

	// F1: Restore subcanvasses to original size
	if (QKey == "F1")
		this->GetMetaCanvas()->RestoreCanvasSizes();
	// F2: Upper 2D view
	if (QKey == "F2")
		this->GetMetaCanvas()->MaximizeSubCanvas(3);
	// F3: Middle 2D view
	if (QKey == "F3")
		this->GetMetaCanvas()->MaximizeSubCanvas(2);
	// F4: Lower 2D view
	if (QKey == "F4")
		this->GetMetaCanvas()->MaximizeSubCanvas(1);
	// F5: 3D view
	if (QKey == "F5")
		this->GetMetaCanvas()->MaximizeSubCanvas(0);
}

void vtkMetaCanvasInteractorStyleForward::OnLeftButtonDown()
{
	// Increment the number of LMB clicks
	this->numberOfClicks++;

	// Get the position of the cursor
	int pickPosition[2];
	this->GetInteractor()->GetEventPosition(pickPosition);

	// Compute the distance between this click and the previous one
	int xdist = pickPosition[0] - this->prevPos[0];
	int ydist = pickPosition[1] - this->prevPos[1];

	// Update the previous pick position
	this->prevPos[0] = pickPosition[0];
	this->prevPos[1] = pickPosition[1];

	// Compute the total click distance
	int moveDistance = (int) sqrt((double)(xdist * xdist + ydist * ydist));

	// Stop the timer, and get the elapsed time between two LMB clicks
	this->timer->StopTimer();
	double dt = this->timer->GetElapsedTime();

	// For this to register as a double-click, the distance moved between clicks
	// should be less than five pixels, and the time should be less than half
	// a second. If this is not a double-click, use the default "OnButtonDown".

	if(moveDistance > 5 || dt > 0.5)
	{ 
		this->numberOfClicks = 1;
		this->OnButtonDown();
	}

	// We've got a double-click! Time to resize the subcanvasses.
	if(this->numberOfClicks == 2)
	{
		this->numberOfClicks = 0;
		this->GetMetaCanvas()->DoubleClickedOnCanvas();
	}

	// Start the timer again
	this->timer->StartTimer();
}

void vtkMetaCanvasInteractorStyleForward::OnButtonDown()
{
  //cout << this<<"(vtkMetaCanvasInteractorStyleForward)::OnButtonDown()"<<endl;
  if ( this->GetMetaCanvas() == NULL )
    {
    return;
    }

  this->GetMetaCanvas()->SelectPokedSubCanvas();
}

void vtkMetaCanvasInteractorStyleForward::SetMetaCanvas(vtkMetaCanvas* metacanvas)
{
  // note:
  // don't return if this->MetaCanvas == metacanvas, because
  // vtkMetaCanvasInteractorStyle::SelectMetaCanvas(metacanvas) must still be called.
  // this is to make sure the right interactor is being used if the metacanvas's
  // has changed.

  vtkMetaCanvas* mc = this->GetMetaCanvas();
  if ( mc != NULL )
    {
    mc->InteractOnSelectOff();
    }

  this->vtkMetaCanvasInteractorStyle::SetMetaCanvas(metacanvas);
  mc = this->GetMetaCanvas();

  if ( mc != NULL )
    {
    mc->InteractOnSelectOn();
    }
}

} // namespace bmia
