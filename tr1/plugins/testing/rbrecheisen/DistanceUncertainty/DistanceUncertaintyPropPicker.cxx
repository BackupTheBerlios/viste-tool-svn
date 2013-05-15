#include <DistanceUncertaintyPropPicker.h>

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPropPicker::DistanceUncertaintyPropPicker()
	{
		this->Plugin = 0;

		this->PropPicker = vtkPropPicker::New();
		this->PropPicker->AddObserver( vtkCommand::StartPickEvent, this );
		this->PropPicker->AddObserver( vtkCommand::PickEvent, this );
		this->PropPicker->AddObserver( vtkCommand::EndPickEvent, this );

		this->CellPicker = vtkCellPicker::New();
		this->CellPicker->AddObserver( vtkCommand::StartPickEvent, this );
		this->CellPicker->AddObserver( vtkCommand::PickEvent, this );
		this->CellPicker->AddObserver( vtkCommand::EndPickEvent, this );

		this->Enabled = true;

		vtkSphereSource * sphere = vtkSphereSource::New();
		sphere->SetRadius( 2 );
		sphere->SetThetaResolution( 64 );
		sphere->SetPhiResolution( 32 );

		vtkPolyDataMapper * mapper = vtkPolyDataMapper::New();
		mapper->SetInput( sphere->GetOutput() );
		sphere->Delete();

		this->PickActor = vtkActor::New();
		this->PickActor->SetMapper( mapper );
		this->PickActor->GetProperty()->SetColor( 0, 1, 0 );
		mapper->Delete();
	}

	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPropPicker::~DistanceUncertaintyPropPicker()
	{
		this->Plugin = 0;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPropPicker::Execute( vtkObject * caller, unsigned long eventId, void * callData )
	{
		std::cout << "DistanceUncertaintyPropPicker::Execute()" << std::endl;
		if( ! this->GetPlugin() || ! this->IsEnabled() )
			return;

		if( eventId == vtkCommand::EndPickEvent || eventId == vtkCommand::PickEvent || eventId == vtkCommand::StartPickEvent )
		{
			double * pickPos = this->PropPicker->GetPickPosition();
			std::cout << "DistanceUncertaintyPropPicker::Execute() pickPos = " << pickPos[0] << " " << pickPos[1] << " " << pickPos[2] << std::endl;

			vtkRenderer * renderer = this->Plugin->getRenderer3D();
			if( ! renderer->HasViewProp( this->PickActor ) )
			{
				renderer->AddViewProp( this->PickActor );
			}

			this->Plugin->render();
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPropPicker::SetPlugin( DistanceUncertaintyPlugin * plugin )
	{
		this->Plugin = plugin;
		this->Plugin->getInteractor()->SetPicker( this->PropPicker );
	}

	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPlugin * DistanceUncertaintyPropPicker::GetPlugin()
	{
		return this->Plugin;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPropPicker::SetEnabled( bool enabled )
	{
		this->Enabled = enabled;
	}

	///////////////////////////////////////////////////////////////////////////
	bool DistanceUncertaintyPropPicker::IsEnabled()
	{
		return this->Enabled;
	}
}
