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

#include "vtkCylinderSubCanvas.h"
#include <vtkObjectFactory.h>
//#include <vtkInteractorStyleJoystickCamera.h>
//#include <vtkInteractorStyleSwitch.h>
#include "vtkInteractorStyleSwitchFixed.h"
//#include <vtkInteractorStyleSwitch.h>
#include <vtkCylinderSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>

namespace bmia {

vtkStandardNewMacro(vtkCylinderSubCanvas);
vtkCxxRevisionMacro(vtkCylinderSubCanvas, "$Revision: 0.1 $");

vtkCylinderSubCanvas::vtkCylinderSubCanvas()
{
  this->CylinderSource = vtkCylinderSource::New();
  this->CylinderSource->SetResolution(7);

  vtkPolyDataMapper *cylinderMapper = vtkPolyDataMapper::New();
  cylinderMapper->SetInput(this->CylinderSource->GetOutput());

  vtkActor* cylinderActor = vtkActor::New();
  cylinderActor->SetMapper(cylinderMapper);
  cylinderActor->GetProperty()->SetColor(1.0, 0, 0);

  this->GetRenderer()->AddActor(cylinderActor);
  this->GetRenderer()->SetBackground(0.2, 0.1, 0.1);

  //vtkInteractorStyleJoystickCamera* istyle = vtkInteractorStyleJoystickCamera::New();
  vtkInteractorStyleSwitchFixed* istyle = vtkInteractorStyleSwitchFixed::New();
  //vtkInteractorStyleSwitch* istyle = vtkInteractorStyleSwitch::New();
  this->SetInteractorStyle(istyle);

  istyle->Delete();
  cylinderActor->Delete();
  cylinderMapper->Delete();
  //cylinder->Delete();
}

vtkCylinderSubCanvas::~vtkCylinderSubCanvas()
{
  this->CylinderSource->Delete();
  this->CylinderSource = NULL;
}

int vtkCylinderSubCanvas::GetCylinderResolution()
{
  return this->CylinderSource->GetResolution();
}

void vtkCylinderSubCanvas::SetCylinderResolution(int resolution)
{
  //cout<<"vtkCylinderSubCanvas::SetCylinderResolution("<<resolution<<")"<<endl;
  vtkDebugMacro(<<"Setting cylinder resolution to "<<resolution);
  this->CylinderSource->SetResolution(resolution);
  this->Modified();
}

} // namespace bmia
