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

/* 
 * FiberCuttingPlugin.cxx
 *
 * 2010-11-15	Yang YU
 * - First Version.
 *
 * 2010-11-16	Yang YU
 * - Realized basic function of toolbox (next/previous page, plain text, setting parameters).
 *
 * 2010-11-16	Yang YU
 * - Added fiber data loading and visualization (Data Page).
 *
 * 2010-11-25	Yang YU
 * - Realized the point picker.
 *
 * 2011-03-14	Evert van Aart
 * - Version 1.0.0.
 * - Changes to the tutorial text in the GUI and the text of the error/warning dialogs. 
 * 
 * 2011-03-28	Evert van Aart
 * - Version 1.1.0.
 * - Increased stability of plugin.
 * - Rewording of GUI elements and message boxes.
 * - Redesign of GUI, enable/disable buttons to make workflow more clear.
 * - Unconfirmed points now show up in red, confirmed points are green. 
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.1.1.
 * - When saving output fibers, the plugin now automatically selects the 
 *   data directory defined in the default profile. 
 *
 * 2011-04-21	Evert van Aart
 * - Version 1.1.2.
 * - Removed the progress bar for the output writer, since it is pretty much
 *   instantaneous anyway.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.1.3.
 * - Improved attribute handling.
 *
 * 2011-06-06	Evert van Aart
 * - Version 1.1.4.
 * - Fixed crash when deleting fiber data sets from the Fiber Visualization plugin.
 *
 */


/** Includes */

#include "FiberCuttingPlugin.h"


class vtkMedicalCanvas;


namespace bmia{


//-----------------------------[ Constructor ]-----------------------------\\

FiberCuttingPlugin::FiberCuttingPlugin() : AdvancedPlugin("Fiber Cutting") 
{
	// Create a new assembly
	this->assembly =vtkPropAssembly::New();

	// Set up GUI
	this->widget = new QWidget();
	this->ui = new Ui::FiberCuttingForm();
	this->ui->setupUi(this->widget);

	// Set up initial state of the toolBox
	this->ui->toolBox->setVisible(0);
	this->ui->toolBox->setItemEnabled(0,0);
	this->ui->toolBox->setItemEnabled(1,0);
	this->ui->toolBox->setItemEnabled(2,0);	

	this->AutoEndFlag=0;
	this->ModifiedFiberIndex=0;

	this->styleTrackball    = vtkInteractorStyleTrackballCamera::New();
	this->styleTrackballCP  = NULL;
	this->styleTrackballCEP = NULL;

	// Connect SIGNAL and SLOT
	connect( this->ui->FiberCutting_StartButton, SIGNAL(clicked()), this, SLOT(StartButton()));

	connect( this->ui->FiberSelectPage_NextButton,SIGNAL(clicked()), this, SLOT(FiberSelectPage_NextButton()));
	
	connect( this->ui->FiberCuttingPage_AutoEndButton,SIGNAL(clicked()),this,SLOT(AutoEnd()));
	connect( this->ui->FiberCuttingPage_ConfirmButton,SIGNAL(clicked()),this,SLOT(ConfirmEndPoint()));
	connect( this->ui->FiberCuttingPage_ClearAllButton,SIGNAL(clicked()),this,SLOT(ClearAll()));

	connect( this->ui->FiberCuttingPage_PreviousButton,SIGNAL(clicked()),this, SLOT(FiberCuttingPage_PreviousButton()));
	connect( this->ui->FiberCuttingPage_NextButton,SIGNAL(clicked()), this, SLOT(FiberCuttingPage_NextButton()));

	connect( this->ui->SaveRepeatPage_RepeatButton,SIGNAL(clicked()), this, SLOT(SaveRepeatPage_RepeatButton()));
	connect( this->ui->SaveRepeatPage_QuitButton,SIGNAL(clicked()),this,SLOT(SaveRepeatPage_QuitButton()));
	connect( this->ui->SaveRepeatPage_SaveButton,SIGNAL(clicked()),this,SLOT(SaveRepeatPage_SaveButton()));
}


//------------------------------[ Destructor ]-----------------------------\\

FiberCuttingPlugin::~FiberCuttingPlugin()
{
	// Remove the GUI
	delete this->widget;
	this->widget=NULL;
	
	// Delete ui
	delete this->ui;
	
	// Delete the assembly
	this->assembly->Delete();

	this->styleTrackball->Delete();
}


//------------------------------[ getVtkProp ]-----------------------------\\

vtkProp* FiberCuttingPlugin::getVtkProp()
{
	return this->assembly;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * FiberCuttingPlugin::getGUI()
{
	return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void FiberCuttingPlugin::dataSetAdded(data::DataSet * ds)
{
	Q_ASSERT(ds);

	// Check if the data set is of the "fibers" type
	if (ds->getKind() == "fibers")
	{
		// Check if the data set is existed
		if (!(this->dataSets.contains(ds)))
		{
			// Check if the polydata of this dataset is null 
			if(!ds->getVtkPolyData())
			{
				this->core()->out()->showMessage("Input fiber data set does not contain VTK polydata!");
				return;
			}
						
			// Add this data set to the "dataSets" list
			this->dataSets.append(ds);
						
			// Show this data set's name on "FiberSelectedPage_ListWidget"
			this->ui->FiberSelectPage_ListWidget->addItem(ds->getName());

			// Check if the data set is added in the "FiberSelectedPage"
			if(this->ui->toolBox->currentIndex() == 0 && this->styleTrackballCP)
			{
				// Update the "ComparedDataSets" for "styleTrackballCP"
				this->styleTrackballCP->SetComparedDataSets(this->dataSets);
			}

			// Check if the data set is added in the "FiberCuttingPage"
			if(this->ui->toolBox->currentIndex() == 1 && this->styleTrackballCP && this->styleTrackballCEP)
			{
				// Update the "ComparedDataSets" for both "styleTrackballCP" and "styleTrackballCEP"
				this->styleTrackballCP->SetComparedDataSets(this->dataSets);
				this->styleTrackballCEP->SetComparedDataSets(this->dataSets);
			}
		
			return;
		}
	}
}


//---------------------------[ dataSetChanged ]----------------------------\\


void FiberCuttingPlugin::dataSetChanged(data::DataSet * ds)
{
	if (ds->getKind() != "fibers")
		return;

	if (this->dataSets.contains(ds) == false)
		return;

	int dsIndex = this->dataSets.indexOf(ds);

	// If we're currently picking end points, and we've selected the fibers that
	// are being changed, we simply reset the toolbox
	if (this->ui->toolBox->currentIndex() == 1 && this->styleTrackballCP->PickedDataSet_Index)
	{
		this->SaveRepeatPage_QuitButton();
	}

	// Get the list item of the fiber set
	QListWidgetItem * currentListItem = this->ui->FiberSelectPage_ListWidget->item(dsIndex);

	// Update the fiber set name
	if (currentListItem)
		currentListItem->setText(ds->getName());
}


//---------------------------[ dataSetRemoved ]----------------------------\\

void FiberCuttingPlugin::dataSetRemoved(data::DataSet *ds)
{
	if (ds->getKind() != "fibers")
		return;

	if (this->dataSets.contains(ds) == false)
		return;

	int dsIndex = this->dataSets.indexOf(ds);

	// Get the list item of the fiber set
	this->ui->FiberSelectPage_ListWidget->takeItem(dsIndex);
	this->dataSets.removeAt(dsIndex);

	// Reset the toolbox, just to be sure
	this->SaveRepeatPage_QuitButton();
}


//-----------------------------[ PageNext ]--------------------------------\\

void FiberCuttingPlugin::PageNext ()
{
	// Get current toolBox index number
	int index = ui->toolBox->currentIndex();

	// Turn to the next page. Make only the next page "enable", while other pages "disable".
	for (int i = 0; i < 3; i++)
	{

		if(i == (index + 1) % 3)
		{
			this->ui->toolBox->setItemEnabled(i, true);
			this->ui->toolBox->setCurrentIndex(i);
		}
		else
		{
			this->ui->toolBox->setItemEnabled(i, false);
		}
	}
}


//----------------------------[ PagePrevious ]-----------------------------\\

void FiberCuttingPlugin::PagePrevious()
{
	// Get current toolBox index number
	int index = ui->toolBox->currentIndex();

	// Turn to the previous page. Make only the previous page "enable", while other pages "disable".
	for (int i = 0; i < 3; i++)
	{
		if(i == (index - 1))
		{
			this->ui->toolBox->setItemEnabled(i, true);
			this->ui->toolBox->setCurrentIndex(i);
		}
		else
		{
			this->ui->toolBox->setItemEnabled(i, false);
		}
	}
}


//-----------------------------[ StartButton ]-----------------------------\\

void FiberCuttingPlugin::StartButton()
{	
	// Check if the "dataSets" list is null
	if(this->dataSets.length()==0)
	{
		QMessageBox::warning(this->getGUI(), "Fiber Cutting Plugin", "No available fiber data set!");
		return;
	}
	else
	{
		// Set option for toolBox
		this->ui->toolBox->setVisible(true);
		this->ui->toolBox->setItemEnabled(0, true);
		this->ui->toolBox->setItemEnabled(1, false);
		this->ui->toolBox->setItemEnabled(2, false);
		this->ui->FiberCutting_StartButton->setEnabled(false);

		// Create a new styleTrackballCP 
		this->styleTrackballCP = vtkInteractorStyleTrackballCellPicker::New();																								
		
		// Set up the options for styleTrackballCP
		this->styleTrackballCP->SetRenderProcess(this->fullCore()->canvas()->GetRenderer3D());		
		this->styleTrackballCP->SetComparedDataSets(this->dataSets);

		// Set this styleTrackballCP to subcanvas
		vtkSubCanvas * subcanvas = this->fullCore()->canvas()->GetSubCanvas3D();
		subcanvas->SetInteractorStyle(this->styleTrackballCP);	

		// Create a cellPicker and set up its options
		vtkCellPicker * cellPicker = vtkCellPicker::New();                              
		cellPicker->SetTolerance(0.001);

		// Set this cellPicker to subcanvas
		subcanvas->GetInteractor()->SetPicker(cellPicker);
		cellPicker->Delete();
	}

}


//---------------------[ FiberSelectPage_NextButton ]----------------------\\

void FiberCuttingPlugin::FiberSelectPage_NextButton()
{
	// Check if successfully picked a fiber  
	if(this->styleTrackballCP->HasGotFiber == 0)
	{
		QMessageBox::warning(this->getGUI(), "Fiber Cutting Plugin", "No fiber selected!");
		return;
	}
	else
	{
		// Turn to next page
		this->PageNext();

		// Set GUI controls
		this->ui->fiberCuttingLabel->setText("Set 1st End Point");
		this->ui->FiberCuttingPage_NextButton->setEnabled(false);
		this->ui->FiberCuttingPage_ClearAllButton->setEnabled(false);
		this->ui->FiberCuttingPage_ConfirmButton->setEnabled(true);
		this->ui->FiberCuttingPage_AutoEndButton->setEnabled(true);

		// Create a styleTrackballCEP
		this->styleTrackballCEP = vtkInteractorStyleTrackballCellEndPicker::New();

		// Set up styleTrackballCEP's options
		this->styleTrackballCEP->SetRender(this->fullCore()->canvas()->GetRenderer3D());
		this->styleTrackballCEP->SetGUI(this->getGUI());
		this->styleTrackballCEP->SetComparedDataSets(this->dataSets);
		this->styleTrackballCEP->SetPickedDataSet_Index(this->styleTrackballCP->PickedDataSet_Index);
		this->styleTrackballCEP->SetPickedFiber_Index(this->styleTrackballCP->PickedFiber_Index);

		// Set styleTrackballCEP to subcanvas
		vtkSubCanvas* subcanvas = this->fullCore()->canvas()->GetSubCanvas3D();		
		subcanvas->SetInteractorStyle(this->styleTrackballCEP);		

		// Create a cellPicker and set up its options
		vtkCellPicker * cellPicker = vtkCellPicker::New();
		cellPicker->SetTolerance(0.001);

		// Set this cellPicker to subcanvas
		subcanvas->GetInteractor()->SetPicker(cellPicker);
		cellPicker->Delete();
	}
}


//---------------------------[ ConfirmEndPoint ]---------------------------\\

void FiberCuttingPlugin::ConfirmEndPoint()
{
	if (this->styleTrackballCEP->HasTempEnd1 == 0 && this->styleTrackballCEP->HasTempEnd2 == 0)
	{
		QMessageBox::warning(this->getGUI(), "Fiber Cutting Plugin", "Please pick a point first.");
		return;
	}

	if(this->styleTrackballCEP->HasTempEnd1 == 1 && this->styleTrackballCEP->NumberOfEnds == 0)
	{
		// When there is one unconfirmed endpoints as well as the number of confirmed endpoint is 0, update the NumberOfEnds to 1
		this->styleTrackballCEP->NumberOfEnds = 1;

		this->styleTrackballCEP->EndActor1->GetProperty()->SetColor(0.0, 1.0, 0.0);
		this->core()->render();
		this->ui->fiberCuttingLabel->setText("Set 2nd End Point");
		this->ui->FiberCuttingPage_ClearAllButton->setEnabled(true);
	}
	if(this->styleTrackballCEP->HasTempEnd2 == 1 && this->styleTrackballCEP->NumberOfEnds == 1)
	{
		// When there is one unconfirmed endpoints as well as the number of confirmed endpoint is 1, update the NumberOfEnds to 2
		this->styleTrackballCEP->NumberOfEnds = 2;
		this->styleTrackballCEP->EndActor2->GetProperty()->SetColor(0.0, 1.0, 0.0);
		this->core()->render();
		this->ui->fiberCuttingLabel->setText("End Points Set!");
		this->ui->FiberCuttingPage_NextButton->setEnabled(true);
		this->ui->FiberCuttingPage_ConfirmButton->setEnabled(false);
		this->ui->FiberCuttingPage_AutoEndButton->setEnabled(false);
	}
}


//-------------------------------[ AutoEnd ]-------------------------------\\

void FiberCuttingPlugin::AutoEnd()
{
	// Check if already two endpoints have been identified
	if(this->styleTrackballCEP->NumberOfEnds == 2)
	{
		QMessageBox::warning(this->getGUI(), "Fiber Cutting Plugin", "There are already two points.");
		return;
	}
	else
	{
		// If AutoEndFlag is 0, perform automatic endpoint marking to the first endpoint. Else if AutoEndFlag is 1, perform automatic endpoint marking to the second endpoint.
		if(this->AutoEndFlag == 0)
		{
			this->styleTrackballCEP->AutoEnd1();
			// change the AutoEndFlag to point to the second endpoint 
			this->AutoEndFlag = 1;
		}
		else if(this->AutoEndFlag == 1)
		{
			this->styleTrackballCEP->AutoEnd2();
			// change the AutoEndFlag to point to the first endpoint
			this->AutoEndFlag = 0;
		}

		this->core()->render();
	}
}


//-------------------------------[ ClearAll ]------------------------------\\

void FiberCuttingPlugin::ClearAll()
{
	this->ui->FiberCuttingPage_PreviousButton->click();
	this->ui->FiberSelectPage_NextButton->click();
}


//-------------------[ FiberCuttingPage_PreviousButton ]-------------------\\

void FiberCuttingPlugin::FiberCuttingPage_PreviousButton()
{
	// Turn to previous page
	this->PagePrevious();

	// Reset the label
	this->ui->fiberCuttingLabel->setText("Set 1st End Point");

	// Remove possible marked endpoint(s)
	if(this->styleTrackballCEP->HasTempEnd1==1)
	{
		this->fullCore()->canvas()->GetRenderer3D()->RemoveActor(this->styleTrackballCEP->EndActor1);
	}
	if(this->styleTrackballCEP->HasTempEnd2==1)
	{
		this->fullCore()->canvas()->GetRenderer3D()->RemoveActor(this->styleTrackballCEP->EndActor2);
	}

	// Remove InteractoStyle
	this->styleTrackballCEP->Delete();

	// Set up option for styleTrackballCP, and set it to the subcanvas
	this->styleTrackballCP->SetRenderProcess(this->fullCore()->canvas()->GetRenderer3D());
	vtkSubCanvas* subcanvas = this->fullCore()->canvas()->GetSubCanvas3D();	
	subcanvas->SetInteractorStyle(this->styleTrackballCP);		

	// Create a new cellpicker
	vtkCellPicker * cellPicker = vtkCellPicker::New();
	cellPicker->SetTolerance(0.001);

	subcanvas->GetInteractor()->SetPicker(cellPicker);

	// Re-render the widget
	this->core()->render();

	cellPicker->Delete();
}


//--------------------[ FiberCuttingPage_NextButton ]----------------------\\

void FiberCuttingPlugin::FiberCuttingPage_NextButton()
{
	// Check the number of selected endpoints
	if (this->styleTrackballCEP->NumberOfEnds < 2)
	{
		QMessageBox::warning(this->getGUI(),"Fiber Cutting Plugin","You have to decide the endpoints of the selected fiber.");
		return;
	}

	// Get the index of the selected endpoints
	int ptId1 = this->styleTrackballCEP->PointIndex[0];
	int ptId2 = this->styleTrackballCEP->PointIndex[1];

	// Check if they are the same
	if(ptId1 == ptId2)
	{
		QMessageBox::warning(this->getGUI(), "Fiber Cutting Plugin", "The begin- and end-points are the same.");

		// Go back one page, and immediately go forward again
		this->FiberCuttingPage_PreviousButton();
		this->FiberSelectPage_NextButton();
		return;
	}

	// Turn to the next page
	this->PageNext();

	// Remove actors
	this->fullCore()->canvas()->GetRenderer3D()->RemoveActor(this->styleTrackballCP->PickedFiber_Actor);
	this->fullCore()->canvas()->GetRenderer3D()->RemoveActor(this->styleTrackballCEP->EndActor1);
	this->fullCore()->canvas()->GetRenderer3D()->RemoveActor(this->styleTrackballCEP->EndActor2);

	// Switch the order of the two points if necessary
	if(ptId1 > ptId2)
	{
		int ptIdTemp;
		ptIdTemp = ptId1;
		ptId1 = ptId2;
		ptId2 = ptIdTemp;
	}

	// Calculate the number of remaining points in the updated fiber
	int NumberOfPoints = ptId2 - ptId1 + 1;
	
	// Get the picked fiber's original polydata
	vtkPolyData * originalPolyData = this->styleTrackballCEP->ComparedDataSets[this->styleTrackballCEP->PickedDataSet_Index]->getVtkPolyData();

	// Get the points of the original polydata
	vtkPoints * originalPoints = originalPolyData->GetPoints();

	// Get the lines of the original polydata
	vtkCellArray * originalLines = originalPolyData->GetLines();

	// Create a new polydata to store the modified polydata
	vtkPolyData * newPolyData  = vtkPolyData::New();

	// Create a new point array for the output
	vtkPoints * newPoints = vtkPoints::New();
	newPoints->SetDataType(originalPoints->GetDataType());
	newPolyData->SetPoints(newPoints);
	newPoints->Delete();

	// Create a new cell array for the output
	vtkCellArray * newLines = vtkCellArray::New();
	newPolyData->SetLines(newLines);
	newLines->Delete();

	// Get the cell data arrays of the in- and output, and prepare them for copying.
	vtkCellData * originalCellData  = originalPolyData->GetCellData();
	vtkCellData * newCellData = newPolyData->GetCellData();
	originalCellData->CopyAllOn();
	newCellData->CopyAllocate(originalCellData);

	// Get the point data arrays of the in- and output, and prepare them for copying.
	vtkPointData * originalPointData  = originalPolyData->GetPointData();
	vtkPointData * newPointData = newPolyData->GetPointData();
	originalPointData->CopyAllOn();
	newPointData->CopyAllocate(originalPointData);

	vtkIdType numberOfPoints;
	vtkIdType * pointList;
	int lineId = 0;

	// Current fiber point coordinates
	double X[3];

	// Initialize traversal of the input fibers
	originalLines->InitTraversal();

	// Loop through all input fibers
	for (vtkIdType lineId = 0; lineId < originalPolyData->GetNumberOfLines(); ++lineId)
	{
		// Get the number of point and the list of point IDs for the next fiber
		originalLines->GetNextCell(numberOfPoints, pointList);

		// We need at least two fiber points
		if (numberOfPoints < 2)
			continue;

		// If this is not the selected fiber, we copy it wholesale
		if (lineId != this->styleTrackballCEP->PickedFiber_Index)
		{
			// Create a list of the output point IDs
			vtkIdList * outList = vtkIdList::New();

			// Loop through the fiber points
			for (vtkIdType pointId = 0; pointId < numberOfPoints; ++pointId)
			{
				// Get the point coordinates from the input
				originalPolyData->GetPoint(pointList[pointId], X);

				// Store the point in the output
				vtkIdType newPointId = newPoints->InsertNextPoint(X);

				// Add the new point ID to the list
				outList->InsertNextId(newPointId);

				// Copy the point data from input to output
				newPointData->CopyData(originalPointData, pointList[pointId], newPointId);
			}

			// Create a new lines using the list of point IDs
			vtkIdType newLineId = newLines->InsertNextCell(outList);

			// Copy the cell data of the input fiber to the output fiber
			newCellData->CopyData(originalCellData, lineId, newLineId);

			// Delete the list of point IDs
			outList->Delete();

		} // if [not the selected fiber]
		else
		{
			// Create a list of the output point IDs
			vtkIdList * outList = vtkIdList::New();

			// Determines whether or not points should be copied to the output
			bool doCopy = false;

			// Loop through the fiber points
			for (vtkIdType pointId = 0; pointId < numberOfPoints; ++pointId)
			{
				// If copying is off, and we encounter one of the end-points, turn copying on
				if ((pointList[pointId] == ptId1 || pointList[pointId] == ptId2) && doCopy == false)
					doCopy = true;
				// If copying is on, and we encounter one of the end-points, turn copying off
				else if ((pointList[pointId] == ptId1 || pointList[pointId] == ptId2) && doCopy == true)
					doCopy = false;

				// Move to the next point if copying is off
				if (doCopy == false)
					continue;

				// Get the point coordinates from the input
				originalPolyData->GetPoint(pointList[pointId], X);

				// Store the point in the output
				vtkIdType newPointId = newPoints->InsertNextPoint(X);

				// Add the new point ID to the list
				outList->InsertNextId(newPointId);

				// Copy the point data from input to output
				newPointData->CopyData(originalPointData, pointList[pointId], newPointId);
			}

			// Create a new lines using the list of point IDs
			vtkIdType newLineId = newLines->InsertNextCell(outList);

			// Copy the cell data of the input fiber to the output fiber
			newCellData->CopyData(originalCellData, lineId, newLineId);

			// Delete the list of point IDs
			outList->Delete();

		} // else [not the selected fiber]

	} // for [all input fibers]

	// Update the data set
	data::DataSet * targetDS = this->dataSets[this->styleTrackballCEP->PickedDataSet_Index];

	targetDS->updateData(newPolyData);

	// Make sure the fibers will be visible
	targetDS->getAttributes()->addAttribute("isVisible", 1.0);

	// Make sure the pipeline will be rebuilt
	targetDS->getAttributes()->addAttribute("updatePipeline", 1.0);

	// Tell the data manager that we've changed the data set
	this->core()->data()->dataSetChanged(targetDS);

	// Record the picked fiber index into modified fiber list
	if (this->ModifiedFiberList.contains(this->styleTrackballCEP->PickedFiber_Index) == false)
	{
		this->ModifiedFiberList.append(this->styleTrackballCEP->PickedFiber_Index);
	}

	// Get the subcanvas and set styleTrackball to this subcanvas
	vtkSubCanvas* subcanvas = this->fullCore()->canvas()->GetSubCanvas3D();		
	subcanvas->SetInteractorStyle(this->styleTrackball);

	this->core()->render();
}


//--------------------[ SaveRepeatPage_RepeatButton ]----------------------\\

void FiberCuttingPlugin::SaveRepeatPage_RepeatButton()
{		
	// Get the index of the picked dataset
	int limitdataSets_index = this->styleTrackballCP->PickedDataSet_Index;
	this->styleTrackballCP->Delete();
	this->styleTrackballCEP->Delete();
	
	// Turn to the next page
	PageNext();

	// Create a new interactorstyle and set it options
	this->styleTrackballCP = vtkInteractorStyleTrackballCellPicker::New();					
	this->styleTrackballCP->SetRenderProcess(this->fullCore()->canvas()->GetRenderer3D());	
	this->styleTrackballCP->SetComparedDataSets(this->dataSets);
	this->styleTrackballCP->SetHasPickedDataSet_Index(limitdataSets_index);

	// Get the subcanvas and set its interactorstyle
	vtkSubCanvas* subcanvas = this->fullCore()->canvas()->GetSubCanvas3D();		
	subcanvas->SetInteractorStyle(this->styleTrackballCP);	
}


//----------------------[ SaveRepeatPage_QuitButton ]----------------------\\

void FiberCuttingPlugin::SaveRepeatPage_QuitButton()
{
	// Delete styleTrackballCP and styleTrackballCEP
	if (this->styleTrackballCP)
	{
		this->styleTrackballCP->Delete();
		this->styleTrackballCP = NULL;
	}

	if (this->styleTrackballCEP)
	{
		this->styleTrackballCEP->Delete();
		this->styleTrackballCEP = NULL;
	}

	// Set option for GUI
	this->ui->toolBox->setVisible(0);
	this->ui->FiberCutting_StartButton->setEnabled(1);

	this->ModifiedFiberList.clear();
}
	

//----------------------[ SaveRepeatPage_SaveButton ]----------------------\\

void FiberCuttingPlugin::SaveRepeatPage_SaveButton()
{
	// Get the picked fiber's original polydata
	vtkPolyData * originalPolyData = this->styleTrackballCEP->ComparedDataSets[this->styleTrackballCEP->PickedDataSet_Index]->getVtkPolyData();

	// Get the points of the original polydata
	vtkPoints * originalPoints = originalPolyData->GetPoints();

	// Get the lines of the original polydata
	vtkCellArray * originalLines = originalPolyData->GetLines();

	// Create a new polydata to store the modified polydata
	vtkPolyData * newPolyData  = vtkPolyData::New();

	// Create a new point array for the output
	vtkPoints * newPoints = vtkPoints::New();
	newPoints->SetDataType(originalPoints->GetDataType());
	newPolyData->SetPoints(newPoints);
	newPoints->Delete();

	// Create a new cell array for the output
	vtkCellArray * newLines = vtkCellArray::New();
	newPolyData->SetLines(newLines);
	newLines->Delete();

	// Get the cell data arrays of the in- and output, and prepare them for copying.
	vtkCellData * originalCellData  = originalPolyData->GetCellData();
	vtkCellData * newCellData = newPolyData->GetCellData();
	originalCellData->CopyAllOn();
	newCellData->CopyAllocate(originalCellData);

	// Get the point data arrays of the in- and output, and prepare them for copying.
	vtkPointData * originalPointData  = originalPolyData->GetPointData();
	vtkPointData * newPointData = newPolyData->GetPointData();
	originalPointData->CopyAllOn();
	newPointData->CopyAllocate(originalPointData);

	vtkIdType numberOfPoints;
	int lineId = 0;

	// Current fiber point coordinates
	double X[3];

	// Loop through all modified fibers
	for (vtkIdType lineId = 0; lineId < this->ModifiedFiberList.size(); ++lineId)
	{
		// Get the cell of the current fiber
		vtkCell * cell = originalPolyData->GetCell(this->ModifiedFiberList.at(lineId));
		numberOfPoints = cell->GetNumberOfPoints();

		// Create a list of the output point IDs
		vtkIdList * outList = vtkIdList::New();

		// Loop through the fiber points
		for (vtkIdType i = 0; i < numberOfPoints; ++i)
		{
			// Get the ID of the "i"-th point of the fiber
			vtkIdType pointId = cell->GetPointId(i);

			// Get the point coordinates from the input
			originalPoints->GetPoint(pointId, X);

			// Store the point in the output
			vtkIdType newPointId = newPoints->InsertNextPoint(X);

			// Add the new point ID to the list
			outList->InsertNextId(newPointId);

			// Copy the point data from input to output
			newPointData->CopyData(originalPointData, pointId, newPointId);
		}

		// Create a new lines using the list of point IDs
		vtkIdType newLineId = newLines->InsertNextCell(outList);

		// Copy the cell data of the input fiber to the output fiber
		newCellData->CopyData(originalCellData, this->ModifiedFiberList.at(lineId), newLineId);

		// Delete the list of point IDs
		outList->Delete();

	} // for [all modified fibers]

	// Get the default data directory
	QDir dataDir = this->core()->getDataDirectory();

	// Open a file dialog to get a filename
	QString fileName = QFileDialog::getSaveFileName(NULL, "Write Fibers", dataDir.absolutePath(), "Fibers (*.fbs)");

	// Check if the filename is correct
	if (fileName.isEmpty())
		return;

	// Convert the QString to a character array
	QByteArray ba = fileName.toAscii();
	char * fileNameChar = ba.data();

	// Create a polydata writer
	vtkPolyDataWriter * writer = vtkPolyDataWriter::New();
		
	// Configure the writer
	writer->SetFileName(fileNameChar);
	writer->SetInput(newPolyData);
	writer->SetFileTypeToASCII();
		
	// Write output file
	writer->Write();

	// Delete the writer and the new polydata
	writer->Delete();
	newPolyData->Delete();
}


}//namespace bmia

Q_EXPORT_PLUGIN2(libFiberCuttingPlugin,bmia::FiberCuttingPlugin)