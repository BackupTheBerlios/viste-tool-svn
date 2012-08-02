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
