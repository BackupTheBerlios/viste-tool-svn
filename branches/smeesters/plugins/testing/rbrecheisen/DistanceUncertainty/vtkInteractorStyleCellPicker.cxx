#include <vtkInteractorStyleCellPicker.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkSphereSource.h>
#include <vtkCellPicker.h>
#include <vtkProperty.h>
#include <vtkPoints.h>

vtkCxxRevisionMacro( vtkInteractorStyleCellPicker, "$Revision: 1.0 $" );
vtkStandardNewMacro( vtkInteractorStyleCellPicker );

//////////////////////////////////////////////////////////////////////
vtkInteractorStyleCellPicker::vtkInteractorStyleCellPicker()
{
	this->Renderer = 0;
	this->EventHandler = 0;
	this->PickActor = this->CreateDefaultPickActor();

	this->PickedPosition[0] = -1;
	this->PickedPosition[1] = -1;
	this->PickedPosition[2] = -1;
}

//////////////////////////////////////////////////////////////////////
vtkInteractorStyleCellPicker::~vtkInteractorStyleCellPicker()
{
}

//////////////////////////////////////////////////////////////////////
void vtkInteractorStyleCellPicker::OnLeftButtonDown()
{
	if( ! this->GetRenderer() )
		return;

	int position[2];
	this->Interactor->GetEventPosition( position );

	vtkCellPicker * picker = vtkCellPicker::New();
	int picked = picker->Pick( position[0], position[1], 0, this->GetRenderer() );
	if( picked )
	{
        if( this->Interactor->GetShiftKey() && this->Interactor->GetControlKey() )
		{
			double point[3];
			vtkPoints * points = picker->GetPickedPositions();
			if( points->GetNumberOfPoints() > 0 )
			{
				points->GetPoint( 0, point );
				this->PickActor->SetPosition( point );

				if( ! this->GetRenderer()->HasViewProp( this->PickActor ) )
				{
					this->GetRenderer()->AddViewProp( this->PickActor );
				}

				this->PickedPosition[0] = point[0];
				this->PickedPosition[1] = point[1];
				this->PickedPosition[2] = point[2];

				this->EventHandler->Execute( 0, 0, this->PickedPosition );
			}

			this->GetRenderer()->GetRenderWindow()->Render();
		}
	}

	picker->Delete();
}

//////////////////////////////////////////////////////////////////////
void vtkInteractorStyleCellPicker::SetEventHandler( vtkCommand * handler )
{
	this->EventHandler = handler;
}

//////////////////////////////////////////////////////////////////////
vtkCommand * vtkInteractorStyleCellPicker::GetEventHandler()
{
	return this->EventHandler;
}

//////////////////////////////////////////////////////////////////////
void vtkInteractorStyleCellPicker::SetRenderer( vtkRenderer * renderer )
{
	this->Renderer = renderer;
}

//////////////////////////////////////////////////////////////////////
vtkRenderer * vtkInteractorStyleCellPicker::GetRenderer()
{
	return this->Renderer;
}

//////////////////////////////////////////////////////////////////////
void vtkInteractorStyleCellPicker::SetPickActorToDefault()
{
	if( this->PickActor )
		this->PickActor->Delete();
	this->PickActor = this->CreateDefaultPickActor();
}

//////////////////////////////////////////////////////////////////////
void vtkInteractorStyleCellPicker::SetPickActor( vtkActor * actor )
{
	if( this->PickActor )
		this->PickActor->Delete();
	this->PickActor = actor;
}

//////////////////////////////////////////////////////////////////////
vtkActor * vtkInteractorStyleCellPicker::GetPickActor()
{
	return this->PickActor;
}

//////////////////////////////////////////////////////////////////////
double * vtkInteractorStyleCellPicker::GetPickedPosition()
{
	return this->PickedPosition;
}

//////////////////////////////////////////////////////////////////////
void vtkInteractorStyleCellPicker::GetPickedPosition( double position[3] )
{
	position[0] = this->PickedPosition[0];
	position[1] = this->PickedPosition[1];
	position[2] = this->PickedPosition[2];
}

//////////////////////////////////////////////////////////////////////
vtkActor * vtkInteractorStyleCellPicker::CreateDefaultPickActor()
{
	vtkSphereSource * sphere = vtkSphereSource::New();
	sphere->SetRadius( 2 );
	sphere->SetThetaResolution( 64 );
	sphere->SetPhiResolution( 32 );

	vtkPolyDataMapper * mapper = vtkPolyDataMapper::New();
	mapper->SetInput( sphere->GetOutput() );
	sphere->Delete();

	vtkActor * actor = vtkActor::New();
	actor->GetProperty()->SetColor( 0, 1, 0 );
	actor->SetMapper( mapper );
	mapper->Delete();

	return actor;
}
