/** Includes */

#include "vtkInteractorStyleTrackballCellEndPicker.h"

namespace bmia
{		

//-----------------------------[ Constructor ]-----------------------------\\

vtkInteractorStyleTrackballCellEndPicker::vtkInteractorStyleTrackballCellEndPicker()
{
	// // Initial setting for parameters
	this->NumberOfEnds=0;
	this->PickedDataSet_Index=-1;
	this->PickedFiber_Index=-1;
	this->HasTempEnd1=0;
	this->HasTempEnd2=0;
}

//------------------------------[ Destructor ]-----------------------------\\

vtkInteractorStyleTrackballCellEndPicker::~vtkInteractorStyleTrackballCellEndPicker()
{
	// Destory
	if(this->NumberOfEnds==1)
	{
		this->End1->Delete();
	
		this->EndMapper1->Delete();

		this->EndActor1->Delete();
	}
	else if (this->NumberOfEnds==2)
	{
		this->End1->Delete();
	
		this->EndMapper1->Delete();

		this->EndActor1->Delete();

		this->End2->Delete();
	
		this->EndMapper2->Delete();

		this->EndActor2->Delete();
	}
}

vtkStandardNewMacro(vtkInteractorStyleTrackballCellEndPicker);			

//--------------------------[ SetRenderProcess ]---------------------------\\

void vtkInteractorStyleTrackballCellEndPicker::SetRender(vtkRenderer * renderer)
{
	// Get the vtkRenderer
	this->Renderer = renderer;
}


//-------------------------------[ SetGUI ]--------------------------------\\

void vtkInteractorStyleTrackballCellEndPicker::SetGUI(QWidget * gui)
{
	// Get the GUI
	this->GUI = gui;
}


//---------------------------[ SetComparedDataSets ]-----------------------\\

void vtkInteractorStyleTrackballCellEndPicker::SetComparedDataSets(QList<data::DataSet*> comparedDataSets)
{
	// Get the datasets which contains all the loaded "fiber" type datasets.
	this->ComparedDataSets=comparedDataSets;
}


//-------------------------[ SetPickedDataSet_Index ]----------------------\\

void vtkInteractorStyleTrackballCellEndPicker::SetPickedDataSet_Index(int index)
{
	// Get the index of the already picked dataset
	this->PickedDataSet_Index = index;
}

//--------------------------[ SetPickedFiber_Index ]------------------------\\

void vtkInteractorStyleTrackballCellEndPicker::SetPickedFiber_Index(int index)
{
	// Get the index of the picked fiber 
	this->PickedFiber_Index = index;
}

//-----------------------------[ OnLeftButtonDown ]------------------------\\

void vtkInteractorStyleTrackballCellEndPicker::OnLeftButtonDown()
{
	// Get the cell picker from the interactor
	vtkCellPicker * CellPicker = vtkCellPicker::SafeDownCast(Interactor->GetPicker());

	if (!CellPicker)
		return;

	// Perform pick action on the click event position. Returns 1 if successfully 
	// picked, and 0 otherwise.

	int Picked = CellPicker->Pick(	this->Interactor->GetEventPosition()[0], 
									this->Interactor->GetEventPosition()[1], 
									0,  // Always zero
									this->Renderer);

	// Perform if pick action succeeded
	if(Picked)
	{
		// Check the number of endpoints. NumberOfEnds counts the number of confirmed endpoint.
		if (this->NumberOfEnds >= 2)
		{
			QMessageBox::warning(this->GUI,"Fiber Cutting Plugin", "Both end points have already been set!",QMessageBox::Ok,QMessageBox::Ok);
			return;
		}

		// Get the picked fiber bundle's polydata.
		this->PickedDataSet_PolyData = vtkPolyData::SafeDownCast(CellPicker->GetDataSet());

		if (this->PickedDataSet_PolyData == NULL)
			return;

		// Get some property of the picked fiber bundle
		int no_Points1 = this->PickedDataSet_PolyData->GetNumberOfPoints();
		int no_Points2 = this->ComparedDataSets[this->PickedDataSet_Index]->getVtkPolyData()->GetNumberOfPoints();

		int no_Cells1 = this->PickedDataSet_PolyData->GetNumberOfCells();
		int no_Cells2 = this->ComparedDataSets[this->PickedDataSet_Index]->getVtkPolyData()->GetNumberOfCells();

		// Check if the picked fiber bundle is equal to working-on fiber bundle and also if the picked endpoint is on the picked fiber.
		if(no_Points1 == no_Points2 && no_Cells1 == no_Cells2  && CellPicker->GetCellId()==this->PickedFiber_Index)
		{
			// Check the number of confirmed endpoints
			if(this->NumberOfEnds == 0)
			{
				// Check if unconfirmed endpoint in cache, if so, delete it for new endpoint. 
				if(HasTempEnd1)
				{
					this->Renderer->RemoveActor(EndActor1);
					this->End1->Delete();
					this->EndMapper1->Delete();
					this->EndActor1->Delete();
				}

				// Create a array for picked point coordinate
				double PickedPoint[3];
				CellPicker->GetPickPosition(PickedPoint);

				this->PointIndex[NumberOfEnds] = CellPicker->GetPointId();
				
				// Create a new actor for endpoint and visualize it the render window
				this->End1 = vtkSphereSource::New();  
				this->End1 ->SetRadius(1);

				this->EndMapper1 = vtkPolyDataMapper::New();
				this->EndMapper1->SetInput(this->End1->GetOutput());
				this->EndActor1 = vtkActor::New();
				this->EndActor1->SetMapper(this->EndMapper1);
				this->EndActor1->SetPosition(PickedPoint[0],PickedPoint[1],PickedPoint[2]);
				this->EndActor1->GetProperty()->SetColor(1.0, 0.0, 0.0);
				this->EndActor1->SetPickable(0);
				
				this->Renderer->AddActor(EndActor1);
				
				// Set up the flag for unconfirmed endpoint
				this->HasTempEnd1 = 1;

			} // if [first end point]

			// Check the number of confirmed endpoint.
			else if(this->NumberOfEnds == 1)
			{
				// Check if unconfirmed endpoint in cache, if so, delete it for new endpoint. 
				if(this->HasTempEnd2)
				{
					this->Renderer->RemoveActor(EndActor2);
					this->End2->Delete();
					this->EndMapper2->Delete();
					this->EndActor2->Delete();
				}

				// Create a array for picked point coordinate
				double PickedPoint[3];
				CellPicker->GetPickPosition(PickedPoint);

				this->PointIndex[NumberOfEnds] = CellPicker->GetPointId();

				// Create a new actor for endpoint and visualize it the render window
				this->End2 = vtkSphereSource::New();  
				this->End2 ->SetRadius(1);

				this->EndMapper2 = vtkPolyDataMapper::New();
				this->EndMapper2->SetInput(this->End2->GetOutput());
				this->EndActor2 = vtkActor::New();
				this->EndActor2->SetMapper(this->EndMapper2);
				this->EndActor2->SetPosition(PickedPoint[0],PickedPoint[1],PickedPoint[2]);
				this->EndActor2->GetProperty()->SetColor(1.0, 0.0, 0.0);
				this->EndActor2->SetPickable(0);
				
				this->Renderer->AddActor(EndActor2);

				// Set up the flag for unconfirmed endpoint
				this->HasTempEnd2 = 1;

			} // if [second end point]

		} // if [valid point on fiber]

	} // if [Picked]

	// Forward events
	vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}


//---------------------------------[ AutoEnd1 ]----------------------------\\

void vtkInteractorStyleTrackballCellEndPicker::AutoEnd1()
{
	// Check the number of confirmed endpoint.
	if(this->NumberOfEnds==0)
	{
		// Check if unconfirmed endpoint in cache, if so, detele it for new endpoint. 
		if(this->HasTempEnd1)
		{
			this->Renderer->RemoveActor(EndActor1);
			this->End1->Delete();
			this->EndActor1->Delete();
			this->EndMapper1->Delete();
		}

		// Get the origianl endpoint of the picked fiber, and make a mark on this point location.
		vtkPolyData * polydata = this->ComparedDataSets[this->PickedDataSet_Index]->getVtkPolyData();
		vtkCell * Cell = polydata->GetCell(this->PickedFiber_Index);
		this->PointIndex[this->NumberOfEnds]=Cell->GetPointId(Cell->GetNumberOfPoints()-1);
		this->End1 = vtkSphereSource::New();  
		this->End1 ->SetRadius(1);
	
		this->EndMapper1 = vtkPolyDataMapper::New();
		this->EndMapper1->SetInput(this->End1->GetOutput());
		this->EndActor1 = vtkActor::New();
		this->EndActor1->SetMapper(this->EndMapper1);

		double EndPointLoc[3];
		Cell->GetPoints()->GetPoint(Cell->GetNumberOfPoints()-1,EndPointLoc);
		this->EndActor1->SetPosition(EndPointLoc[0],EndPointLoc[1],EndPointLoc[2]);
		this->EndActor1->GetProperty()->SetColor(1.0, 0.0, 0.0);
		this->EndActor1->SetPickable(0);
	
		this->Renderer->AddActor(this->EndActor1);
		this->HasTempEnd1=1;
	}

	// Check the number of confirmed endpoint.
	if(this->NumberOfEnds==1)
	{
		// Check if unconfirmed endpoint in cache, if so, detele it for new endpoint. 
		if(this->HasTempEnd2)
		{
			this->Renderer->RemoveActor(EndActor2);
			this->End2->Delete();
			this->EndActor2->Delete();
			this->EndMapper2->Delete();
		}

		// Get the origianl endpoint of the picked fiber, and make a mark on this point location.
		vtkPolyData * polydata = this->ComparedDataSets[this->PickedDataSet_Index]->getVtkPolyData();
		vtkCell * Cell = polydata->GetCell(this->PickedFiber_Index);
		this->PointIndex[this->NumberOfEnds]=Cell->GetPointId(Cell->GetNumberOfPoints()-1);
		this->End2 = vtkSphereSource::New();  
		this->End2 ->SetRadius(1);

		this->EndMapper2 = vtkPolyDataMapper::New();
		this->EndMapper2->SetInput(this->End1->GetOutput());
		this->EndActor2 = vtkActor::New();
		this->EndActor2->SetMapper(this->EndMapper1);

		double EndPointLoc[3];
		Cell->GetPoints()->GetPoint(Cell->GetNumberOfPoints()-1,EndPointLoc);
		this->EndActor2->SetPosition(EndPointLoc[0],EndPointLoc[1],EndPointLoc[2]);
		this->EndActor2->GetProperty()->SetColor(1.0, 0.0, 0.0);
		this->EndActor2->SetPickable(0);
	
		this->Renderer->AddActor(this->EndActor2);
		this->HasTempEnd2=1;

	}
}


//---------------------------------[ AutoEnd2 ]----------------------------\\

void vtkInteractorStyleTrackballCellEndPicker::AutoEnd2()
{
	// Check the number of confirmed endpoint.
	if(this->NumberOfEnds==0)
	{
		// Check if unconfirmed endpoint in cache, if so, detele it for new endpoint. 
		if(this->HasTempEnd1)
		{
			this->Renderer->RemoveActor(EndActor1);
			this->End1->Delete();
			this->EndActor1->Delete();
			this->EndMapper1->Delete();
		}

		// Get the origianl endpoint of the picked fiber, and make a mark on this point location.
		vtkPolyData * polydata = this->ComparedDataSets[this->PickedDataSet_Index]->getVtkPolyData();
		vtkCell * Cell = polydata->GetCell(this->PickedFiber_Index);
		this->PointIndex[this->NumberOfEnds]=Cell->GetPointId(0);
		this->End1 = vtkSphereSource::New();  
		this->End1 ->SetRadius(1);

		this->EndMapper1 = vtkPolyDataMapper::New();
		this->EndMapper1->SetInput(this->End1->GetOutput());
		this->EndActor1 = vtkActor::New();
		this->EndActor1->SetMapper(this->EndMapper1);

		double EndPointLoc[3];
		Cell->GetPoints()->GetPoint(0,EndPointLoc);
		this->EndActor1->SetPosition(EndPointLoc[0],EndPointLoc[1],EndPointLoc[2]);
		this->EndActor1->GetProperty()->SetColor(1.0, 0.0, 0.0);
		this->EndActor1->SetPickable(0);
	
		this->Renderer->AddActor(this->EndActor1);
		this->HasTempEnd1=1;
	}

	// Check the number of confirmed endpoint.
	if(this->NumberOfEnds==1)
	{
		// Check if unconfirmed endpoint in cache, if so, detele it for new endpoint. 
		if(this->HasTempEnd2)
		{
			this->Renderer->RemoveActor(EndActor2);
			this->End2->Delete();
			this->EndActor2->Delete();
			this->EndMapper2->Delete();
		}

		// Get the origianl endpoint of the picked fiber, and make a mark on this point location.
		vtkPolyData * polydata = this->ComparedDataSets[this->PickedDataSet_Index]->getVtkPolyData();
		vtkCell * Cell = polydata->GetCell(this->PickedFiber_Index);
		this->PointIndex[this->NumberOfEnds]=Cell->GetPointId(0);
		this->End2 = vtkSphereSource::New();  
		this->End2 ->SetRadius(1);

		this->EndMapper2 = vtkPolyDataMapper::New();
		this->EndMapper2->SetInput(this->End1->GetOutput());
		this->EndActor2 = vtkActor::New();
		this->EndActor2->SetMapper(this->EndMapper1);

		double EndPointLoc[3];
		Cell->GetPoints()->GetPoint(0,EndPointLoc);
		this->EndActor2->SetPosition(EndPointLoc[0],EndPointLoc[1],EndPointLoc[2]);
		this->EndActor2->GetProperty()->SetColor(1.0, 0.0, 0.0);
		this->EndActor2->SetPickable(0);
	
		this->Renderer->AddActor(this->EndActor2);
		this->HasTempEnd2=1;
	}
}

}// namespace bmia
		