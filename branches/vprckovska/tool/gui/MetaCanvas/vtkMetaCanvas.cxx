/*
 * vtkMetaCanvas.cxx
 *
 * 2005-01-12	Tim Peeters
 * - First version
 *
 * 2005-07-15	Tim Peeters
 * - Implemented SetInteractorStyle() functions.
 * - Add initialization of interactor style in constructor.
 *
 * 2005-11-14	Tim Peeters
 * - Renamed from vtkMetaCanvasSelect to vtkMetaCanvas.
 *
 * 2005-11-15	Tim Peeters
 * - Added SetInteractor() that carries the old interactor style
 *   along to the new RenderWindowInteractor (if both exist).
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 * 2011-03-01	Evert van Aart
 * - Metacanvas now emits an event when the user resets the camera ('R').
 *
 */

#include "vtkMetaCanvas.h"
#include <vtkObjectFactory.h> // for vtkStandardNewMacro
#include <assert.h>
#include "vtkMetaCanvasUserEvents.h"
#include "vtkMetaCanvasInteractorStyle.h"
#include "vtkMetaCanvasInteractorStyleSwitch.h"
#include <vtkRenderWindowInteractor.h>

namespace bmia {

vtkStandardNewMacro(vtkMetaCanvas);

vtkMetaCanvas::vtkMetaCanvas()
{
  this->SelectedSubCanvas = NULL;
  this->InteractOnSelect = true;
  this->InteractorStyle = NULL;

  vtkMetaCanvasInteractorStyleSwitch* istyle = vtkMetaCanvasInteractorStyleSwitch::New();
  this->SetInteractorStyle(istyle);
  istyle->Delete(); istyle = NULL;
}

vtkMetaCanvas::~vtkMetaCanvas()
{
  this->SelectedSubCanvas = NULL;
}

void vtkMetaCanvas::SelectSubCanvasAt(int x, int y)
{
  vtkDebugMacro(<<"Selecting subcanvas at ("<<x<<", "<<y<<").");
  this->SelectSubCanvas(this->GetSubCanvasAt(x, y));
}

void vtkMetaCanvas::SelectSubCanvas(vtkAbstractSubCanvas* subcanvas)
{
  vtkDebugMacro(<<"Selecting subcanvas "<<subcanvas);
  if (this->SelectedSubCanvas != subcanvas)
    {
    if (this->InteractOnSelect)
      {
      if (this->SelectedSubCanvas != NULL)
        {
        this->SelectedSubCanvas->InteractOff();
        }
      if (subcanvas != NULL)
        {
        subcanvas->InteractOn();
        }
      }
    this->SelectedSubCanvas = subcanvas;
    this->InvokeEvent(vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVAS_SELECTED,
		      this->SelectedSubCanvas);
    }
}


int vtkMetaCanvas::ResetCameraOfPokedSubCanvas()
{
	// First, select the subcanvas under the mouse cursor
	this->SelectPokedSubCanvas();

	// Invoke the custom event for resetting the camera, and return if it was
	// aborted. Callbacks listening to this event should set the abort flag 
	// if they handled it. In our case, the callback class of the Plane
	// Visualization plugin will set the flag if and only if the selected
	// subcanvas was one of the 2D views; this way, resetting the camera for
	// the 3D view is done in the usual way, while resetting a 2D camera
	// is done using the function in the PlanesVisPlugin.

	return this->InvokeEvent(vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVAS_CAMERA_RESET,
		this->SelectedSubCanvas);
}

void vtkMetaCanvas::SetInteractOnSelect(bool interact)
{
  if (this->InteractOnSelect == interact)
    {
    return;
    }

  this->InteractOnSelect = interact;
  if (this->SelectedSubCanvas != NULL)
    {
    vtkAbstractSubCanvasCollection* scc = this->GetSubCanvasses();
    assert( scc != NULL );
    for (int i=0; i < scc->GetNumberOfItems(); i++)
      {
      assert( scc->GetItem(i) != NULL );
      scc->GetItem(i)->InteractOff();
      }
    this->SelectedSubCanvas->SetInteract(this->InteractOnSelect);
    }
}

void vtkMetaCanvas::SelectPokedSubCanvas()
{
  if ( this->Interactor == NULL )
    {
    this->SelectSubCanvas(NULL);
    return;
    }

  assert( this->Interactor != NULL );

  int eventX; int eventY;
  this->Interactor->GetEventPosition(eventX, eventY);
  this->SelectSubCanvasAt(eventX, eventY);
}


void vtkMetaCanvas::DoubleClickedOnCanvas()
{
	// Do nothing if no interactor exists
	if (this->Interactor == NULL)
	{
		this->SelectSubCanvas(NULL);
		return;
	}

	// Resize the subcanvas which was double-clicked
	int eventX; int eventY;
	this->Interactor->GetEventPosition(eventX, eventY);
	this->ResizeSubCanvasAt(eventX, eventY);
}

void vtkMetaCanvas::RemoveSubCanvas(vtkAbstractSubCanvas* subcanvas)
{
  vtkDebugMacro(<<"Removing subcanvas "<<subcanvas);
  if (this->SelectedSubCanvas == subcanvas)
    {
    this->SelectSubCanvas(NULL);
    }
  this->vtkGenericMetaCanvas::RemoveSubCanvas(subcanvas);
}

void vtkMetaCanvas::SetInteractorStyle(vtkMetaCanvasInteractorStyle* style)
{
  if (this->InteractorStyle == style)
    {
    return;
    }

  if (this->InteractorStyle) this->InteractorStyle->UnRegister(this);
   
  this->InteractorStyle = style;
  if (this->InteractorStyle)
    {
    this->InteractorStyle->Register(this);
    this->InteractorStyle->SetMetaCanvas(this);
    }

  if (this->GetRenderWindowInteractor())
    {
    this->GetRenderWindowInteractor()->SetInteractorStyle(this->InteractorStyle);
    }
}

void vtkMetaCanvas::SetInteractor(vtkRenderWindowInteractor* rwi)
{
  // switch the rwi
  this->vtkGenericMetaCanvas::SetInteractor(rwi);

  // use the old interactor style for the new rwi, if both are not NULL
  if (this->InteractorStyle)
    {
    if (this->GetInteractor())
      {
      this->GetInteractor()->SetInteractorStyle(this->InteractorStyle);
      } // if (this->GetInteractor())
    this->InteractorStyle->SetMetaCanvas(this);
    } // if (this->InteractorStyle)
}

} // namespace bmia
