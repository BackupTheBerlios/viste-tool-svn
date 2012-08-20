/**
 * vtkMetaCanvasInteractorStyleWM.cxx
 * by Tim Peeters
 *
 * 2005-01-06	Tim Peeters
 * - First version
 *
 * 2005-01-13 	Tim Peeters
 * - Small changes due to changes in vtkMetaCanvas.
 */

#include "vtkMetaCanvasInteractorStyleWM.h"
#include <vtkObjectFactory.h> // for vtkStandardNewMacro
#include <assert.h>
#include <vtkRenderWindowInteractor.h>

namespace bmia {

vtkStandardNewMacro(vtkMetaCanvasInteractorStyleWM);

vtkMetaCanvasInteractorStyleWM::vtkMetaCanvasInteractorStyleWM()
{
  this->CurrentManipulationStyle = VTK_MCMS_NOTHING;
}

vtkMetaCanvasInteractorStyleWM::~vtkMetaCanvasInteractorStyleWM()
{
  // nothing to do.
}

void vtkMetaCanvasInteractorStyleWM::SetCurrentManipulationStyle(int mstyle)
{
  assert( this->GetMetaCanvas() != NULL );
  assert( (mstyle == VTK_MCMS_MOVING) || (mstyle == VTK_MCMS_RESIZING) ||
          (mstyle == VTK_MCMS_NOTHING) );

  this->CurrentManipulationStyle = mstyle;
  if ( mstyle != VTK_MCMS_NOTHING )
    {
    this->GetMetaCanvas()->SelectPokedSubCanvas();
    if ( (mstyle == VTK_MCMS_RESIZING) && (this->GetMetaCanvas()->GetSelectedSubCanvas()) )
      {
      this->DetectCurrentViewportCorner();
      }
    }
}

void vtkMetaCanvasInteractorStyleWM::OnLeftButtonDown()
{
  this->SetCurrentManipulationStyle(VTK_MCMS_MOVING);
}

void vtkMetaCanvasInteractorStyleWM::OnLeftButtonUp()
{
  this->SetCurrentManipulationStyle(VTK_MCMS_NOTHING);
}

void vtkMetaCanvasInteractorStyleWM::OnRightButtonDown()
{
  this->SetCurrentManipulationStyle(VTK_MCMS_RESIZING);
}

void vtkMetaCanvasInteractorStyleWM::OnRightButtonUp()
{
  this->SetCurrentManipulationStyle(VTK_MCMS_NOTHING);
}

void vtkMetaCanvasInteractorStyleWM::OnMouseMove()
{
  assert( this->GetMetaCanvas() != NULL );

  if ( this->CurrentManipulationStyle == VTK_MCMS_NOTHING )
    {
    return;
    }

  vtkAbstractSubCanvas* sc = this->GetMetaCanvas()->GetSelectedSubCanvas();
  if ( sc == NULL )
    {
    return;
  }

  int eventX; int eventY;
  int lastEventX; int lastEventY;
  double* viewport;
  int* windowSize;
  double dx; double dy; // amount of change in window-relative coordinates.

  vtkRenderWindowInteractor* rwi = this->GetInteractor();
  assert( rwi != NULL );

  rwi->GetEventPosition(eventX, eventY);
  rwi->GetLastEventPosition(lastEventX, lastEventY);

  vtkRenderWindow* rw = this->GetMetaCanvas()->GetRenderWindow();
  assert( rw != NULL );

  windowSize = rw->GetSize();
  assert( sc->GetRenderWindow() != NULL );
  viewport = sc->GetViewport();
  // viewport == {xmin, ymin, xmax, ymax}

  assert( windowSize[0] >= 1 ); assert( windowSize[1] >= 1 );

  dx = -((double)(lastEventX - eventX))/((double)(windowSize[0]));
  dy = -((double)(lastEventY - eventY))/((double)(windowSize[1]));
  
  switch ( this->CurrentManipulationStyle )
    {
    case VTK_MCMS_MOVING:
      viewport[0] += dx; viewport[1] += dy;
      viewport[2] += dx; viewport[3] += dy;

      // make sure the subcanvas viewport stays inside the metacanvas bounds:
      if ( viewport[0] < 0 ) { viewport[2] -= viewport[0]; viewport[0] = 0; }
      if ( viewport[1] < 0 ) { viewport[3] -= viewport[1]; viewport[1] = 0; }
      if ( viewport[2] > 1 ) { viewport[0] += (1-viewport[2]); viewport[2] = 1; }
      if ( viewport[3] > 1 ) { viewport[1] += (1-viewport[3]); viewport[3] = 1; }
      break;
    case VTK_MCMS_RESIZING:
      switch ( this->CurrentViewportCorner )
        {
        case VTK_MCVC_UPPER_LEFT:
          if (viewport[2]-viewport[0]-dx>0.05)
            {
	    viewport[0] += dx; if (viewport[0] < 0) viewport[0] = 0;
	    }
	  if (viewport[3]+dy-viewport[1]>0.05)
            {
            viewport[3] += dy; if (viewport[3] > 1) viewport[3] = 1;
            }
          break;
        case VTK_MCVC_UPPER_RIGHT:
          if (viewport[2]+dx-viewport[0]>0.05)
            {
	    viewport[2] += dx; if (viewport[2] > 1) viewport[2] = 1;
            }
          if (viewport[3]+dy-viewport[1]>0.05)
            {
	    viewport[3] += dy; if (viewport[3] > 1) viewport[3] = 1;
            }
          break;
        case VTK_MCVC_LOWER_LEFT:
          if (viewport[2]-viewport[0]-dx>0.05)
            {
            viewport[0] += dx; if (viewport[0] < 0) viewport[0] = 0;
            }
          if (viewport[3]-viewport[1]-dy>0.05)
            {
            viewport[1] += dy; if (viewport[1] < 0) viewport[1] = 0;
            }
          break;
        case VTK_MCVC_LOWER_RIGHT:
          if (viewport[2]+dx-viewport[0]>0.05)
            {
            viewport[2] += dx; if (viewport[2] > 1) viewport[2] = 1;
            }
          if (viewport[3]-viewport[1]-dy>0.05)
            {
            viewport[1] += dy; if (viewport[1] < 0) viewport[1] = 0;
            } 
          break;
        }
      break;
    }

  sc->SetViewport(viewport);
  rw->Render(); // r->Render() doesn't seem to do the trick.
}

void vtkMetaCanvasInteractorStyleWM::DetectCurrentViewportCorner()
{
  assert( this->GetMetaCanvas() != NULL );

  double* viewport; int* windowSize;
  int eventX; int eventY;
  double rx; double ry; // relative X and Y (in the interval [0,1])

  vtkRenderWindow* rw = this->GetMetaCanvas()->GetRenderWindow();
  assert( rw != NULL );

  vtkAbstractSubCanvas* sc = this->GetMetaCanvas()->GetSelectedSubCanvas();
  // this method is only called by SetCurrentManipulationStyle() if
  // there is a selected subcanvas.
  assert( sc != NULL );

  vtkRenderWindowInteractor* rwi = this->GetInteractor();
  assert( rwi != NULL );

  rwi->GetEventPosition(eventX, eventY);
  windowSize = rw->GetSize();
  viewport = sc->GetViewport();

  assert(windowSize[0] > 0); assert(windowSize[1] > 0);
  rx = ((double)eventX)/((double)(windowSize[0]));
  ry = ((double)eventY)/((double)(windowSize[1]));

  this->CurrentViewportCorner = this->GetViewportCorner(viewport, rx, ry);

}

int vtkMetaCanvasInteractorStyleWM::GetViewportCorner(double* viewport, double rx, double ry)
{
  int result;
  double centerx = (viewport[2] - viewport[0])/2.0;
  double centery = (viewport[3] - viewport[1])/2.0;
  double dx = rx - centerx - viewport[0];
  double dy = ry - centery - viewport[1];
  if (dx < 0)
    {
    if (dy < 0)
      {
      result = VTK_MCVC_LOWER_LEFT;
      }
    else // dy >= 0
      {
      result = VTK_MCVC_UPPER_LEFT;
      }
    }
  else // dx >= 0
    {
    if (dy < 0)
      {
      result = VTK_MCVC_LOWER_RIGHT;
      }
    else
      {
      result = VTK_MCVC_UPPER_RIGHT;
      }
    }

  return result;
}

void vtkMetaCanvasInteractorStyleWM::SetMetaCanvas(vtkMetaCanvas* metacanvas)
{
  // note:
  // don't return if this->MetaCanvas == metacanvas, because
  // vtkMetaCanvasInteractorStyle::SelectMetaCanvas(metacanvas) must still be called.
  // this is to make sure the right interactor is being used if the metacanvas's
  // has changed.

  this->vtkMetaCanvasInteractorStyle::SetMetaCanvas(metacanvas);

  if ( this->GetMetaCanvas() != NULL )
    {
    this->GetMetaCanvas()->InteractOnSelectOff();
    }
}

} // namespace bmia
