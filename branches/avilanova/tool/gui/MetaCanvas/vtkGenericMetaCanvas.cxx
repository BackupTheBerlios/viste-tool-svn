/*
 * vtkGenericMetaCanvas.cxx
 *
 * 2005-01-05	Tim Peeters
 * - First version
 *
 * 2005-01-12	Tim Peeters
 * - Removed selection related parts
 * - Use new vtkAbstractSubCanvas methods
 * - Use vtkAbstractSubCanvasCollection for subcanvasses
 *
 * 2005-07-15	Tim Peeters
 * - In constructor, initialize with a standard new vtkRenderWindowInteractor
 *   instead of having this->Interactor == NULL.
 * - Implemented Start() function
 *
 * 2005-11-14	Tim Peeters
 * - Renamed from vtkMetaCanvas to vtkGenericMetaCanvas
 *
 * 2005-11-15	Tim Peeters
 * - Modified OnEvent(object, event, data) so that the renderwindow is only
 *   re-rendered if it was already visible. Otherwise it may be mapped on the
 *   screen already when setting up the subcanvasses and it should not be
 *   rendered yet.
 *
 * 2006-02-13	Tim Peeters
 * - Make it possible to change the renderwindow.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 * 2011-02-21	Evert van Aart
 * - Increased support for maximization of subcanvasses.
 *
 */

#include "vtkGenericMetaCanvas.h"
#include <vtkObjectFactory.h>
#include <assert.h>
#include "vtkMetaCanvasUserEvents.h"
#include "vtkSubCanvas.h"

namespace bmia {

vtkStandardNewMacro(vtkGenericMetaCanvas);

vtkGenericMetaCanvas::vtkGenericMetaCanvas()
{
  //this->RenderWindow = vtkRenderWindow::New();

  this->SubCanvasses = vtkAbstractSubCanvasCollection::New();
  assert( this->SubCanvasses->GetNumberOfItems() == 0 );
  this->BackgroundRenderer = vtkRenderer::New();
  this->Interactor = NULL;

  // Create a RenderWindow
  this->RenderWindow = NULL;
  vtkRenderWindow* rw = vtkRenderWindow::New();
  this->SetRenderWindow(rw);
  rw->Delete(); rw = NULL;

  // Make sure a background is always rendered
//  this->RenderWindow->AddRenderer(this->BackgroundRenderer);
  this->BackgroundRenderer->SetViewport(0,0,1,1);
  this->BackgroundRenderer->SetBackground(0,0,0);

  this->CallbackCommand = vtkCallbackCommand::New();
  this->CallbackCommand->SetClientData(this);
  this->CallbackCommand->SetCallback(vtkGenericMetaCanvas::OnEvent);

  vtkRenderWindowInteractor* rwi = vtkRenderWindowInteractor::New();
  this->SetInteractor(rwi);
  rwi->Delete(); rwi = NULL;
}

vtkGenericMetaCanvas::~vtkGenericMetaCanvas()
{
  this->SetInteractor(NULL);
  //this->RenderWindow->Delete();
  //this->RenderWindow = NULL;
  this->SetRenderWindow(NULL);
  while( this->SubCanvasses->GetNumberOfItems() > 0 )
    {
    this->RemoveSubCanvas(SubCanvasses->GetItem(0));
    }
  this->SubCanvasses->Delete();
  this->SubCanvasses = NULL;
  this->BackgroundRenderer->Delete();
  this->BackgroundRenderer = NULL;
}

void vtkGenericMetaCanvas::AddSubCanvas(vtkAbstractSubCanvas* subcanvas)
{
  assert( subcanvas != NULL );

  // Add the subcanvas to SubCanvasses
  this->SubCanvasses->AddItem(subcanvas);
  // the subcanvas is registered by SubCanvasses, so there is no need
  // to do that here as well.

  subcanvas->SetRenderWindow(this->RenderWindow);
  subcanvas->SetInteractor(this->Interactor);
  subcanvas->AddObserver(vtkCommand::ModifiedEvent, this->CallbackCommand);

  this->InvokeEvent(vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVAS_ADDED,
			subcanvas);
}

void vtkGenericMetaCanvas::RemoveSubCanvas(vtkAbstractSubCanvas* subcanvas)
{
  assert( subcanvas != NULL );
  subcanvas->RemoveObserver(this->CallbackCommand);
  subcanvas->SetRenderWindow(NULL);
  subcanvas->SetInteractor(NULL);
  this->SubCanvasses->RemoveItem(subcanvas);
  this->InvokeEvent(vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVAS_REMOVED,
			NULL); // note that subcanvas may have been destructed
}

void vtkGenericMetaCanvas::SetInteractor(vtkRenderWindowInteractor* rwi)
{
  vtkDebugMacro(<<"Setting interactor...");
  if (this->Interactor != rwi)
    {

    // The interactor is Registered by the render window, which is only available
    // inside this class. So there is no need to (Un)Register() it here.
    // But I'll do it anyway to be sure ;)

    if (this->Interactor != NULL)
      {
      this->Interactor->UnRegister(this);
      }

    this->Interactor = rwi;

    if (this->Interactor != NULL)
      {
      this->Interactor->Register(this);
      }
  
    this->RenderWindow->SetInteractor(this->Interactor);
    for (int k = 0; k < this->SubCanvasses->GetNumberOfItems(); k++)
      {
      assert( this->SubCanvasses->GetItem(k) != NULL );
      this->SubCanvasses->GetItem(k)->SetInteractor(rwi);
      }
    }
}

vtkAbstractSubCanvas* vtkGenericMetaCanvas::GetSubCanvasAt(int x, int y)
{
  vtkAbstractSubCanvas* result = NULL;
  int i = this->SubCanvasses->GetNumberOfItems();

  vtkAbstractSubCanvas* subcanvas = NULL;
  while ( (result == NULL) && (i > 0) )
    {
    i--;
    subcanvas = this->SubCanvasses->GetItem(i);
    assert( subcanvas != NULL );
    if ( subcanvas->IsInViewport(x, y) )
      {
      result = subcanvas;
      }
    }
  return result;
}


void vtkGenericMetaCanvas::MaximizeSubCanvas(int ID)
{
	// Current subcanvas and maximized subcanvas
	vtkAbstractSubCanvas * subcanvas = NULL;
	vtkAbstractSubCanvas * result = NULL;

	// Get the number of canvasses
	int numberOfCanvasses = this->SubCanvasses->GetNumberOfItems();

	// Loop through all subcanvasses
	for (int i = 0; i < numberOfCanvasses; ++i)
	{
		subcanvas = this->SubCanvasses->GetItem(i);

		// Skip non-existent canvasses
		if (subcanvas == NULL)
			continue;


		// Store a pointer to the canvas that needs to be maximized
		if (i == ID)
		{
			result = subcanvas;
			continue;
		}

		// Minimize all other subcanvasses
		subcanvas->SetViewport(0.0, 0.0, 0.0, 0.0);
	}

	// Maximize selected subcanvas
	result->SetViewport(0.0, 0.0, 1.0, 1.0);

	// Store index of selected subcanvas
	this->SubCanvasses->maximizedSubCanvas = ID;

	// Tell anyone who's interested that we've resized the subcanvasses
	this->InvokeEvent(vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVASSES_RESIZED, NULL);

	// Redraw the canvas
	this->GetRenderWindow()->Render();
}

void vtkGenericMetaCanvas::RestoreCanvasSizes()
{
	// Current subcanvas
	vtkAbstractSubCanvas * subcanvas = NULL;

	// Get the number of canvasses
	int numberOfCanvasses = this->SubCanvasses->GetNumberOfItems();

	// Loop through all subcanvasses
	for (int i = 0; i < numberOfCanvasses; ++i)
	{
		// Get the current subcanvas
		subcanvas = this->SubCanvasses->GetItem(i);

		if (subcanvas == NULL)
			continue;

		// Restore the initial viewport
		double vp[4];
		((vtkSubCanvas * ) subcanvas)->getFirstViewPort(vp);
		subcanvas->SetViewport(vp[0], vp[1], vp[2], vp[3]);
	}

	// No maximized subcanvas
	this->SubCanvasses->maximizedSubCanvas = -1;

	// Tell anyone who's interested that we've resized the subcanvasses
	this->InvokeEvent(vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVASSES_RESIZED, NULL);

	// Redraw the canvas
	this->GetRenderWindow()->Render();
}

void vtkGenericMetaCanvas::ResizeSubCanvasAt(int x, int y)
{
	// Current subcanvas
	vtkAbstractSubCanvas * subcanvas = NULL;

	// Selected subcanvas
	vtkAbstractSubCanvas * result = NULL;

	// Index of the selected subcanvas in the subcanvas collection
	int resultID = -1;

	int numberOfCanvasses = this->SubCanvasses->GetNumberOfItems();

	// Loop through all subcanvasses
	for (int i = 0; i < numberOfCanvasses; ++i)
	{
		// Get the current subcanvas
		subcanvas = this->SubCanvasses->GetItem(i);

		if (subcanvas == NULL)
			continue;

		// Check if this is the selected subcanvas
		if (subcanvas->IsInViewport(x, y))
		{
			result = subcanvas;
			resultID = i;
			break;
		}
	}

	// Do nothing if no subcanvas was selected
	if (result == NULL)
		return;

	// If the selected subcanvas is not maximized, maximize it now
	if (this->SubCanvasses->maximizedSubCanvas != resultID)
	{
		// Loop through all subcanvasses
		for (int i = 0; i < numberOfCanvasses; ++i)
		{
			subcanvas = this->SubCanvasses->GetItem(i);

			// Skip non-existant and selected subcanvasses
			if (subcanvas == NULL || i == resultID)
				continue;

			// Minimize subcanvas
			subcanvas->SetViewport(0.0, 0.0, 0.0, 0.0);
		}

		// Maximize selected subcanvas
		result->SetViewport(0.0, 0.0, 1.0, 1.0);

		// Store index of selected subcanvas
		this->SubCanvasses->maximizedSubCanvas = resultID;
	}
	// Otherwise, the selected subcanvas was already maximized,
	// so we go back to the initial layout
	else
	{
		// Loop through all subcanvasses
		for (int i = 0; i < numberOfCanvasses; ++i)
		{
			subcanvas = this->SubCanvasses->GetItem(i);

			if (subcanvas == NULL)
				continue;

			// Restore the initial viewport
			double vp[4];
			((vtkSubCanvas * ) subcanvas)->getFirstViewPort(vp);
			subcanvas->SetViewport(vp[0], vp[1], vp[2], vp[3]);
		}

		// No maximized subcanvas
		this->SubCanvasses->maximizedSubCanvas = -1;
	}

	// Tell anyone who's interested that we've resized the subcanvasses
	this->InvokeEvent(vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVASSES_RESIZED, NULL);
}

void vtkGenericMetaCanvas::OnEvent(vtkObject* object,
			    unsigned long event,
			    void* clientdata,
			    void* calldata)
{
  // vtkDebugMacro not possible because this is a static function. If output
  // is needed, use cout.
  //vtkDebugMacro(<<"object == "<<object<<", event == "<<event<<", clientdata =="
	//			<<clientdata<<", calldata == "<<calldata<<".");

  vtkGenericMetaCanvas* self = reinterpret_cast<vtkGenericMetaCanvas *>( clientdata );
  self->OnEvent(object, event, calldata);
}

void vtkGenericMetaCanvas::OnEvent(vtkObject* object,
			    unsigned long event,
			    void* calldata)
{
  vtkDebugMacro(<<"object =="<<object<<", event == "<<event<<", calldata == "
				<<calldata<<".");

  if (!this->RenderWindow) return;

  switch(event)
  {
  case (vtkCommand::ModifiedEvent):
    // ModifiedEvent from one of the subcanvasses.
    if (this->RenderWindow->GetMapped())
      { // the renderwindow is already on-screen. But it changed, so re-render it.
      this->RenderWindow->Render();
      }
    // if the renderwindow was not visible, don't re-render it.
  }
}

void vtkGenericMetaCanvas::Start()
{
  if (this->Interactor) this->Interactor->Start();
}

void vtkGenericMetaCanvas::SetSize(int h, int v)
{
  if (this->RenderWindow) this->RenderWindow->SetSize(h, v);
}

void vtkGenericMetaCanvas::SetRenderWindow(vtkRenderWindow* rw)
{
  if (this->RenderWindow == rw) return;

  vtkDebugMacro(<<"Setting RenderWindow to "<<rw);

  if (this->RenderWindow != NULL)
    {
    // XXX The line below was commented out because this gives an access
    // violation error on windows when starting dtitool.exe. Very strange.
	// FIX THIS!!
    this->RenderWindow->UnRegister(this);
    // I think it was fixed by setting this->RenderWindow to NULL properly in
    // the constructor before calling SetRenderWindow. Not tested on windows yet.
    }

  this->RenderWindow = rw;

  if (this->RenderWindow)
    this->RenderWindow->Register(this);
    this->RenderWindow->AddRenderer(this->BackgroundRenderer);

    this->SubCanvasses->InitTraversal();
    vtkAbstractSubCanvas* sc = this->SubCanvasses->GetNextItem();
    while (sc != NULL)
      {
      sc->SetRenderWindow(this->RenderWindow);
      sc = this->SubCanvasses->GetNextItem();
      } // while
  //object == NULL

//  this->RenderWindow->SetInteractor(this->Interactor);
}

void vtkGenericMetaCanvas::ResetAllCameras()
{
  this->SubCanvasses->InitTraversal();
  vtkAbstractSubCanvas* sc = this->SubCanvasses->GetNextItem();
  while (sc != NULL)
    {
    sc->ResetCamera();
    sc = this->SubCanvasses->GetNextItem();
    } // while
}

} // namespace bmia
