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
