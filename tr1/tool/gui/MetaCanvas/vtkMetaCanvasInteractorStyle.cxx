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
 * vtkMetaCanvasInteractorStyle.cxx
 * by Tim Peeters
 *
 * 2005-01-06	Tim Peeters
 * - First version
 */

#include "vtkMetaCanvasInteractorStyle.h"
#include <vtkObjectFactory.h> // needed for vtkStandardNewMacro
#include <assert.h>

// without this line, this->Interactor->GetEventPosition() does not work:
//#include <vtkRenderWindowInteractor.h>

namespace bmia {

vtkStandardNewMacro(vtkMetaCanvasInteractorStyle);
vtkCxxRevisionMacro(vtkMetaCanvasInteractorStyle, "$Revision: 1.93 $");


vtkMetaCanvasInteractorStyle::vtkMetaCanvasInteractorStyle()
{
  this->MetaCanvas = NULL;
}

vtkMetaCanvasInteractorStyle::~vtkMetaCanvasInteractorStyle()
{
  this->SetMetaCanvas(NULL);
}

void vtkMetaCanvasInteractorStyle::SetMetaCanvas(vtkMetaCanvas* metacanvas)
{
  vtkDebugMacro(<<"Setting metacanvas to "<<metacanvas);
  if (this->MetaCanvas == metacanvas)
  {
    if (this->MetaCanvas == NULL)
      {
      // this->MetaCanvas was NULL and stays NULL.
      return;
      }
    if (this->MetaCanvas->GetInteractor() == this->GetInteractor())
      {
      return;
      }
  }

  if (this->MetaCanvas != NULL)
    {
    this->SetInteractor(this->MetaCanvas->GetInteractor());
    this->MetaCanvas->UnRegister(this);
    }
  
  this->MetaCanvas = metacanvas;
  if (this->MetaCanvas != NULL)
    {
    this->MetaCanvas->Register(this);
    this->SetInteractor(this->MetaCanvas->GetInteractor());
//    this->MetaCanvas->InteractOnSelectOn();
    }
  else
    { // this->MetaCanvas == NULL
    this->SetInteractor(NULL);
    }
  // Linking of events can be done with the SetInteractor() method.
  // The interactor must be defined independently from the metacanvas.
}


} // namespace bmia
