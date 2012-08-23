/** Includes */

#include "vtkInteractorStyleTrackballCellPicker.h"

namespace bmia
{		


//-----------------------------[ Constructor ]-----------------------------\\

vtkInteractorStyleTrackballCellPicker::vtkInteractorStyleTrackballCellPicker()
{
	// Initial setting for parameters
	this->PickedFiber_Mapper = vtkPolyDataMapper::New();		
	this->PickedFiber_Actor  = vtkActor::New();
	this->PickedFiber_PolyData = vtkPolyData::New();
	this->HasGotFiber =0;
	this->HasPickedDataSet_Index=-1;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkInteractorStyleTrackballCellPicker::~vtkInteractorStyleTrackballCellPicker()
{
	// Destroy
	this->Renderer->RemoveActor(this->PickedFiber_Actor);

	this->PickedFiber_Actor->Delete(); 
	this->PickedFiber_Mapper->Delete(); 
	this->PickedFiber_PolyData->Delete();
}


vtkStandardNewMacro(vtkInteractorStyleTrackballCellPicker);


//--------------------------[ SetRenderProcess ]---------------------------\\

void vtkInteractorStyleTrackballCellPicker::SetRenderProcess(vtkRenderer * renderer)
{
	// Get the vtkRenderer and Add the PickedFiber_Actor to Renderer
	this->Renderer = renderer;
	this->Renderer->AddActor(this->PickedFiber_Actor);

	// Set up mapper
	this->PickedFiber_Actor->SetMapper(this->PickedFiber_Mapper);
	
	// Set up the color of the actor
	this->PickedFiber_Actor->GetProperty()->SetColor(1.0,1.0,1.0);
}


//---------------------------[ SetComparedDataSets ]-----------------------\\

void vtkInteractorStyleTrackballCellPicker::SetComparedDataSets(QList<data::DataSet*> comparedDataSets)
{
	// Get the datasets which contains all the loaded "fiber" type datasets.
	this->ComparedDataSets=comparedDataSets;
}


//------------------------[ SetHasPickedDataSet_Index ]--------------------\\

void vtkInteractorStyleTrackballCellPicker::SetHasPickedDataSet_Index(int index)
{
	// Get the index of the already picked dataset
	this->HasPickedDataSet_Index = index;
}


//-----------------------------[ OnLeftButtonDown ]------------------------\\

void vtkInteractorStyleTrackballCellPicker::OnLeftButtonDown()
{
	/*
	std::cout<<endl;
	std::cout<<"     LeftButton: " << this->Interactor->GetEventPosition()[0] << " " << this->Interactor->GetEventPosition()[1] << std::endl;
	*/

	// Get the cellpicker from the interactor
	vtkCellPicker *CellPicker = vtkCellPicker::SafeDownCast(Interactor->GetPicker());

	// Perform pick action on the click event position.Return 1 is successfully picked, otherwise 0.
	int Picked = CellPicker->Pick(this->Interactor->GetEventPosition()[0], 
								   this->Interactor->GetEventPosition()[1], 
								   0,  // always zero
								   this->Renderer);	

	// Initial flag setting. Tis flag is designed to check if picked is fiber
	int PickedIsFiber=0;

	// Perform if pick action successed.
	if(Picked)
	{
		// Get the picked polydata.
		this->PickedDataSet_PolyData = vtkPolyData::SafeDownCast(CellPicker->GetDataSet());

		if (!(this->PickedDataSet_PolyData))
			return;

		// Check HasPickedDataSet_Index. 
		// If it is the first time performing the whole workflow, HasPickedDataSet_Index should be -1; otherwise it would indicate the index of working-on dataset.
		if(this->HasPickedDataSet_Index==-1)
		{
			// Check which dataset in ComparedDataSets list is equal to the picked dataset
			for(int i=0; i<this->ComparedDataSets.length();i++)
			{
				vtkPolyData * ComparedDataSet_PolyData = this->ComparedDataSets[i]->getVtkPolyData();

				int no_Points1 = this->PickedDataSet_PolyData->GetNumberOfPoints();
				int no_Points2 = ComparedDataSet_PolyData->GetNumberOfPoints();
				
				int no_Cells1 = this->PickedDataSet_PolyData->GetNumberOfCells();
				int no_Cells2 = ComparedDataSet_PolyData->GetNumberOfCells();
				
				if(no_Points1 == no_Points2 && no_Cells1 == no_Cells2)
				{
					PickedIsFiber=1;
					this->PickedDataSet_Index = i;
				}
			}	
		}
		else
		{
			int no_Points1 = this->PickedDataSet_PolyData->GetNumberOfPoints();
			int no_Points2 = ComparedDataSets[this->HasPickedDataSet_Index]->getVtkPolyData()->GetNumberOfPoints();

			int no_Cells1 = this->PickedDataSet_PolyData->GetNumberOfCells();
			int no_Cells2 = ComparedDataSets[this->HasPickedDataSet_Index]->getVtkPolyData()->GetNumberOfCells();

			if(no_Points1 == no_Points2 && no_Cells1 == no_Cells2)
			{
				PickedIsFiber=1;
				this->PickedDataSet_Index=this->HasPickedDataSet_Index;
			}
		}
		// Highlight the picked fiber if successfully picked
		if(PickedIsFiber)
		{
			this->HasGotFiber=1;						
			
			// Get the cell id
			int CellId = CellPicker->GetCellId();
			this->PickedFiber_Index = CellId;
			
			/*
			std::cout<<"     PICK Fiber:"<<CellId;
			*/

			// Create a new IdList
			vtkIdList * list =vtkIdList::New();
			list->SetNumberOfIds(1);
			list->SetId(0, CellId);

			// Copy the picked fiber and visualize it.
			this->PickedFiber_PolyData->Allocate();
			this->PickedFiber_PolyData->CopyCells(PickedDataSet_PolyData,list);
			this->PickedFiber_Mapper->SetInput(this->PickedFiber_PolyData);	

			if (this->PickedFiber_PolyData->GetNumberOfCells()==1)
			{
				/*
				cout<<" Point Index:"<<CellPicker->GetPointId()<<endl;
				*/
			}
			else
			{
				/*
				cout<<"     Error!";
				*/
			}	

			list->Delete();
		}

		else
		{
			/*
			std::cout<<"     PICKED IS NOT FIBER."<<std::endl;
			*/
		}
	}
	else
	{
		/*
		std::cout<<"     NO PICK."<<std::endl;
		*/
	}

	// Forward events
	vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

} // namespace bmia
		