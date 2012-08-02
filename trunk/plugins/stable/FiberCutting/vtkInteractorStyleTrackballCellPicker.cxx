/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
		