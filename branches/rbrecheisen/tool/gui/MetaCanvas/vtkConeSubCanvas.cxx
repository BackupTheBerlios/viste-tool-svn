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
