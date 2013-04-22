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
 * vtkConeSubCanvas.cxx
 * by Tim Peeters
 *
 * 2005-01-10	Tim Peeters
 * - First version
 */

#include "vtkConeSubCanvas.h"
#include <vtkObjectFactory.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkConeSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>

namespace bmia {

//vtkCxxRevisionMacro(vtkConeSubCanvas, "$Revision: 0.12 $");
vtkStandardNewMacro(vtkConeSubCanvas);

vtkConeSubCanvas::vtkConeSubCanvas()
{
  vtkConeSource *cone = vtkConeSource::New();
  cone->SetResolution(80);
  vtkPolyDataMapper *coneMapper = vtkPolyDataMapper::New();
  coneMapper->SetInput(cone->GetOutput());

  vtkActor *coneActor = vtkActor::New();
  coneActor->SetMapper(coneMapper);
  coneActor->GetProperty()->SetColor(1.0, 0.0, 1.0);

  this->GetRenderer()->AddActor(coneActor);
  this->GetRenderer()->SetBackground(0.5,0,0);

  vtkInteractorStyleTrackballCamera* istyle = vtkInteractorStyleTrackballCamera::New();
  this->SetInteractorStyle(istyle);

  istyle->Delete();
  coneActor->Delete();
  coneMapper->Delete();
  cone->Delete();
}

vtkConeSubCanvas::~vtkConeSubCanvas()
{
  //cout<<"~vtkConeSubCanvas()"<<endl;
  // everything that was created in the constructor (with New())
  // was also destroyed there.
}

} // namespace bmia
