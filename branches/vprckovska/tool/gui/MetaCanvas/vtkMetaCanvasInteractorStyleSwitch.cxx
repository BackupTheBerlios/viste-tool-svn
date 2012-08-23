/**
 * vtkMetaCanvasInteractorStyleSwitch.cxx
 *
 * 2005-01-07	Tim Peeters
 * - First version.
 *
 * 2011-03-01	Evert van Aart
 * - "OnChar" now also listens to the "R"-key. This allows us to overwrite the
 *   camera-resetting behavior with our own. This is required for the planes in
 *   the 2D views.
 *
 */

#include "vtkMetaCanvasInteractorStyleSwitch.h"

#include "vtkMetaCanvasInteractorStyleForward.h"
#include "vtkMetaCanvasInteractorStyleWM.h"

#include <vtkObjectFactory.h>
#include <assert.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCallbackCommand.h>


namespace bmia {

vtkStandardNewMacro(vtkMetaCanvasInteractorStyleSwitch);

vtkMetaCanvasInteractorStyleSwitch::vtkMetaCanvasInteractorStyleSwitch()
{
  this->StyleForward = vtkMetaCanvasInteractorStyleForward::New();
  this->StyleWM = vtkMetaCanvasInteractorStyleWM::New();
  this->CurrentStyleNr = VTK_MCIS_FORWARD;
  this->CurrentStyle = NULL;
}

vtkMetaCanvasInteractorStyleSwitch::~vtkMetaCanvasInteractorStyleSwitch()
{
  assert( this->StyleForward != NULL );
  this->StyleForward->Delete();
  this->StyleForward = NULL;

  assert( this->StyleWM != NULL );
  this->StyleWM->Delete();
  this->StyleForward = NULL;
}

void vtkMetaCanvasInteractorStyleSwitch::SetMetaCanvas(vtkMetaCanvas* metacanvas)
{
  this->vtkMetaCanvasInteractorStyle::SetMetaCanvas(metacanvas);
  //this->StyleForward->SetMetaCanvas(metacanvas);
  //this->StyleWM->SetMetaCanvas(metacanvas);
  this->SetCurrentStyle();
}

void vtkMetaCanvasInteractorStyleSwitch::OnChar()
{
  assert( this->Interactor != NULL );
  switch(this->Interactor->GetKeyCode())
    {
    case 'm':
    case 'M':
      if (this->CurrentStyleNr == VTK_MCIS_FORWARD)
        {
        this->CurrentStyleNr = VTK_MCIS_WM;
        }
      else
        {
        this->CurrentStyleNr = VTK_MCIS_FORWARD;
        }

      // they do this in vtkInteractorStyleSwitch
      // and without it subcanvas interaction some times breaks after switching
      // from Forward to WM style while a subcanvas with a JoystickCameraStyle
      // is selected.
      this->EventCallbackCommand->SetAbortFlag(1);

      this->SetCurrentStyle();
      break;

		// Reset the camera.   
		case 'r':
		case 'R':

			// Emit a signal that the camera needs to be reset
			if (this->GetMetaCanvas()->ResetCameraOfPokedSubCanvas() == 1)
			{
				// We overwrite the default camera resetting behavior, so we set
				// the abort flag to avoid other callback handling this event.
				this->EventCallbackCommand->SetAbortFlag(1);
			}

			// If the camera reset event was not handled, we do not set the
			// abort flag, so the default handler will take care of things.
			break;
    }
}

void vtkMetaCanvasInteractorStyleSwitch::SetCurrentStyle()
{
  //cout<<"vtkMetaCanvasInteractorStyleSwitch::SetCurrentStyle()\n";
// commented out because this check should not be performed if this
// method is called by SetMetaCanvas(). Otherwise it does no harm.
//  if (((this->CurrentStyleNr == VTK_MCIS_FORWARD) &&
//       (this->CurrentStyle == this->StyleForward)) ||
//      ((this->CurrentStyleNr == VTK_MCIS_WM) &&
//       (this->CurrentStyle == this->StyleWM)))
//    {
//    return;
//    }

  // the style must really be set/changed now.

  // note: this function is also called from SetMetaCanvas(..), so
  // this->GetMetaCanvas() may return NULL.

  if (this->CurrentStyle != NULL)
    {  // unlink the current style from the interactor if needed
      this->CurrentStyle->SetMetaCanvas(NULL);
    }

  //assert( this->Interactor != NULL );

  switch(this->CurrentStyleNr)
    {
    case VTK_MCIS_FORWARD:
      this->CurrentStyle = this->StyleForward;
      break;
    case VTK_MCIS_WM:
      this->CurrentStyle = this->StyleWM;
      break;
    }
    this->CurrentStyle->SetMetaCanvas(this->GetMetaCanvas());
}

/*
void vtkMetaCanvasInteractorStyleSwitch::SetInteractor(vtkRenderWindowInteractor *iren)
{
  if(iren == this->Interactor)
    {
    return;
    }
  // if we already have an Interactor then stop observing it
  if(this->Interactor)
    {
    this->Interactor->RemoveObserver(this->EventCallbackCommand);
    }
  this->Interactor = iren;
  // from vtkInteractorStyleSwitch:
  // add observers for each of the events handled in ProcessEvents
  if(iren)
    {
    iren->AddObserver(vtkCommand::CharEvent, 
                      this->EventCallbackCommand,
                      this->Priority);
  
    iren->AddObserver(vtkCommand::DeleteEvent, 
                      this->EventCallbackCommand,
                      this->Priority);
//    iren->AddObserver(vtkCommand::TimerEvent,
//                      this->EventCallbackCommand,
//                      this->Priority);

//    iren->AddObserver(vtkCommand::AnyEvent,
//		      this->EventCallbackCommand,
//                      this->Priority);
    }
  this->SetCurrentStyle();
}
*/

} // namespace bmia
