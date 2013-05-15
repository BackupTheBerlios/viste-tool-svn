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
 * vtkAbstractSubCanvas.cxx
 * by Tim Peeters
 *
 * 2005-01-12	Tim Peeters
 * - First version
 *
 * 2006-02-17	Tim Peeters
 * - Added functions double GetRelative{X,Y}(int)
 */

#include "vtkAbstractSubCanvas.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkCxxRevisionMacro(vtkAbstractSubCanvas, "$Revision: 1.93 $");

vtkAbstractSubCanvas::vtkAbstractSubCanvas()
{
  this->RenderWindow = NULL;
  this->Interactor = NULL;
  this->Interact = false;
  this->subCanvasName = std::string("Unnamed");
}

vtkAbstractSubCanvas::~vtkAbstractSubCanvas()
{
  // make sure no interactor is still associated with this subcanvas.
  this->SetInteractor(NULL);
}

void vtkAbstractSubCanvas::SetInteractor(vtkRenderWindowInteractor* i)
{
  vtkDebugMacro(<<"Setting interactor to "<<i);
  bool istatus = this->Interact;
  this->SetInteract(false);

  if (this->Interactor != NULL)
    {
    this->Interactor->UnRegister(this);
    }

  this->Interactor = i;
  if (this->Interactor != NULL)
    {
    this->Interactor->Register(this);
    }

  this->SetInteract(istatus);
  //this->Modified();
}

bool vtkAbstractSubCanvas::IsInViewport(int x, int y)
{
  if (this->RenderWindow != NULL)
    {
    int * size = this->RenderWindow->GetSize();
    double * viewport = this->GetViewport();

    if ((viewport[0]*size[0] <= x)&&
        (viewport[2]*size[0] >= x)&&
        (viewport[1]*size[1] <= y)&&
        (viewport[3]*size[1] >= y))
      {
      return true;
      }
    }
  return false;
}

double vtkAbstractSubCanvas::GetRelativeX(int x)
{
  if (this->RenderWindow == NULL)
   {
   vtkWarningMacro(<<"Cannot compute relative X coordinate without a render window!");
   return -1.0;
   }

  int * size = this->RenderWindow->GetSize();
  double* viewport = this->GetViewport();

  return (double)x - viewport[0]*(double)size[0];
}

double vtkAbstractSubCanvas::GetRelativeY(int y)
{
  if (this->RenderWindow == NULL)
    {
    vtkWarningMacro(<<"Cannot compute relative Y coordinate without a render window!");
    return -1.0;
    }

  int * size = this->RenderWindow->GetSize();
  double* viewport = this->GetViewport();

  return (double)y - viewport[1]*(double)size[1];
}

} // namespace bmia
