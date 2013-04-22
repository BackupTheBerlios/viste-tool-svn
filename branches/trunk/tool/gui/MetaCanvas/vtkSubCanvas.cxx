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
 * vtkSubCanvas.cxx
 *
 * 2005-01-05	Tim Peeters
 * - First version
 *
 * 2005-05-31	Tim Peeters
 * - Set the interactor style in the constructor. This way default interaction
 *   is used if no interactor style is explicitly defined.
 *
 * 2005-07-16	Tim Peeters
 * - Implemented Render() function.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 */

#include "vtkSubCanvas.h"
#include <vtkObjectFactory.h>
#include "vtkInteractorStyleSwitchFixed.h"

namespace bmia {

vtkStandardNewMacro(vtkSubCanvas);
vtkCxxRevisionMacro(vtkSubCanvas, "$Revision: 0.1 $");

vtkSubCanvas::vtkSubCanvas()
{
  this->Renderer = vtkRenderer::New();
  //this->Renderer->TwoSidedLightingOff(); // needed to use Phong or Gouraud shading // doesn't work :S
  this->Renderer->SetBackground(0.5, 0.5, 0.5);
  this->InteractorStyle = NULL;
  this->Interactor = NULL;
  this->Interact = false;

  vtkInteractorStyleSwitchFixed* style = vtkInteractorStyleSwitchFixed::New();
  this->SetInteractorStyle(style);
  style->Delete(); style = NULL;

	// Viewport has not yet been initialized
	this->firstSetViewport = true;

	// Default viewport values
	this->firstViewport[0] = 0.0;
	this->firstViewport[1] = 0.0;
	this->firstViewport[2] = 0.0;
	this->firstViewport[3] = 0.0;
}

vtkSubCanvas::~vtkSubCanvas()
{
  vtkDebugMacro(<<"Destroying subcanvas "<<this);
  this->SetInteract(false);
  this->SetInteractorStyle(NULL);
  this->Renderer->Delete();
  this->Renderer = NULL;
}

void vtkSubCanvas::SetInteractorStyle(vtkInteractorStyle* istyle)
{
  vtkDebugMacro(<<"Setting interactor style to "<<istyle);
  if (this->InteractorStyle == istyle)
  {
    return;
  }

  bool i = this->Interact;
  this->SetInteract(false);

  if (this->InteractorStyle != NULL)
    {
    this->InteractorStyle->UnRegister(this);
    }

  this->InteractorStyle = istyle;
  if (this->InteractorStyle != NULL)
    {
    this->InteractorStyle->Register(this);
    }

  this->SetInteract(i);
//  this->Modified();
}

void vtkSubCanvas::SetInteract(bool i)
{
  vtkDebugMacro(<<"Setting interact to "<<i);

  if (this->Interact == i)
    {
    return;
    }

  this->Interact = i;
  if (this->InteractorStyle == NULL)
    {
    return;
    }
  if (this->Interactor == NULL)
    {
    return;
    }
  if (this->Interact)
    { // switching from no interaction to interaction
    this->InteractorStyle->SetInteractor(this->Interactor);
    this->InteractorStyle->SetDefaultRenderer(this->Renderer);
    this->InteractorStyle->SetCurrentRenderer(this->Renderer);
    }
  else
    { // switching from interaction to no interaction
    this->InteractorStyle->SetInteractor(NULL);
    this->InteractorStyle->SetDefaultRenderer(NULL);
    //this->InteractorStyle->SetCurrentRenderer(NULL); // why? changes on
			// interaction anyway if no default renderer is set.
    }
}

void vtkSubCanvas::SetViewport(double xmin, double ymin, double xmax, double ymax)
{
	// If this is the first time that this function is called, store the viewport.
	if (this->firstSetViewport)
	{
		this->firstViewport[0] = xmin;
		this->firstViewport[1] = ymin;
		this->firstViewport[2] = xmax;
		this->firstViewport[3] = ymax;
		this->firstSetViewport = false;
	}

  this->Renderer->SetViewport(xmin, ymin, xmax, ymax);
//  this->Modified();
}


void vtkSubCanvas::getFirstViewPort(double * vp)
{
	// Copy first viewport to the output
	vp[0] = this->firstViewport[0];
	vp[1] = this->firstViewport[1];
	vp[2] = this->firstViewport[2];
	vp[3] = this->firstViewport[3];
}

double* vtkSubCanvas::GetViewport()
{
  return this->Renderer->GetViewport();
}

void vtkSubCanvas::SetRenderWindow(vtkRenderWindow* rw)
{
  if (this->RenderWindow == rw)
    {
    return;
    }

  if (this->RenderWindow != NULL)
    {
    this->RenderWindow->RemoveRenderer(this->Renderer);
    }

  this->vtkAbstractSubCanvas::SetRenderWindow(rw);

  if (this->RenderWindow != NULL)
    {
    this->RenderWindow->AddRenderer(this->Renderer);
    }
}

void vtkSubCanvas::Render()
{
  this->Renderer->Render();
  if (this->RenderWindow) this->RenderWindow->Render();
}

} // namespace bmia
