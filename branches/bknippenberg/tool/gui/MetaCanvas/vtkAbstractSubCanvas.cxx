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
