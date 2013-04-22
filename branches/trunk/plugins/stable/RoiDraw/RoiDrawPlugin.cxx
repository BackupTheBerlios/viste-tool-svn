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
 * RoiDrawPlugin.h
 *
 * 2011-02-16	Evert van Aart
 * - First version. This is a new approach to the ROI Drawing plugin by Tim 
 *   Peeters. The previous version was plagued by a number of bugs, most of
 *   which were due to the use of a separate drawing window, which is not
 *   fully supported by VTK. This plugin moves drawing the ROIs back to the
 *   main window, which is more stable and more user friendly.
 *
 * 2011-03-09	Evert van Aart
 * - Version 1.0.0.
 * - Automatically set seeding type to "Distance" for loaded ROIs.
 * - Added support for grouping fibers.
 *
 * 2011-03-14	Evert van Aart
 * - Version 1.0.1.
 * - Fixed ROIs occasionnally not showing up in the 2D views.
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.2.
 * - When saving ROIs, plugin now automatically selects the data directory
 *   as defined by the default profile. 
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.1.0.
 * - Outsourced seeding options to the new Seeding plugin.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.1.1.
 * - Improved attribute handling.
 *
 */


/** Includes */

#include "RoiDrawPlugin.h"


namespace bmia {


RoiDrawPlugin::RoiDrawPlugin() : AdvancedPlugin("ROI Edit")
{
	// Create the GUI
	this->widget = new QWidget();
	this->ui = new Ui::RoiForm();
	this->ui->setupUi(this->widget);

	// Create the assembly, which holds all ROI actors
	this->assembly = vtkPropAssembly::New();

	// No ROI selected by default
	this->selectedRoi = -1;

	// No canvas selected by default
	this->currentSubCanvas = -1;

    // leave the option for seeding from voxels disabled until
    // there are data sets that can be chosen here.
    this->ui->renameButton->setEnabled(false);
    this->ui->deleteButton->setEnabled(false);
    this->ui->drawButton->setEnabled(false);

    connect(this->ui->roiComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(selectROI(int)));
    connect(this->ui->drawButton, SIGNAL(clicked()), this, SLOT(toggleDrawing()));
    connect(this->ui->newButton, SIGNAL(clicked()), this, SLOT(createNewROI()));
    connect(this->ui->renameButton, SIGNAL(clicked()), this, SLOT(rename()));
    connect(this->ui->deleteButton, SIGNAL(clicked()), this, SLOT(deleteRoi()));
	connect(this->ui->saveROIButton, SIGNAL(clicked()), this, SLOT(saveROI()));
	connect(this->ui->groupButton, SIGNAL(clicked()), this, SLOT(groupROIs()));
}

void RoiDrawPlugin::init()
{
	// Create a callback function for when the user selects a new subcanvas
	this->callBack = RoiDrawPluginCallback::New();
	this->callBack->plugin = this;
	this->fullCore()->canvas()->AddObserver(vtkCommand::UserEvent + 
						BMIA_USER_EVENT_SUBCANVAS_SELECTED, this->callBack);

	// Finish initialization
	this->slices[0] = NULL;
	this->slices[1] = NULL;
	this->slices[2] = NULL;
	this->isDrawing = false;

}

RoiDrawPlugin::~RoiDrawPlugin()
{
	delete this->widget;

	if (this->assembly)
		this->assembly->Delete();

	if (this->callBack)
		this->callBack->Delete();

	// Clear the lists of input data sets
	this->roiData.clear();
	this->volumeData.clear();

	// Loop through all tracers
	for (int i = 0; i < this->tracers.size(); ++i)
	{
		// Delete all existing tracing widgets
		TracerInformation currentTracer = this->tracers.at(i);

		if (currentTracer.widgets[0])
			currentTracer.widgets[0]->Delete();
		if (currentTracer.widgets[1])
			currentTracer.widgets[1]->Delete();
		if (currentTracer.widgets[2])
			currentTracer.widgets[2]->Delete();
	}

	// Clear the list of tracers
	this->tracers.clear();
}

QWidget * RoiDrawPlugin::getGUI()
{
    return this->widget;
}

vtkProp * RoiDrawPlugin::getVtkProp()
{
	return this->assembly;
}


void RoiDrawPlugin::enableControls()
{
	this->ui->roiComboBox->setEnabled(true);
	this->ui->newButton->setEnabled(true);

	// True if ROIs are available
	bool roiAvailable = this->selectedRoi != -1;

	this->ui->nameEdit->setEnabled(roiAvailable);
	this->ui->renameButton->setEnabled(roiAvailable);
	this->ui->deleteButton->setEnabled(roiAvailable);
	this->ui->saveROIButton->setEnabled(roiAvailable);
	this->ui->groupButton->setEnabled(roiAvailable);

	// True if the slices have an input available
	bool slicesSet = false;
	
	if (this->slices[0] && this->slices[1] && this->slices[2])
	{
		if (this->slices[0]->GetInput() && this->slices[1]->GetInput() && this->slices[2]->GetInput())
		{
			slicesSet = roiAvailable;
		}
	}

	this->ui->drawButton->setEnabled(slicesSet);

	// True if DTI images are available
	bool volumeAvailable = roiAvailable && this->volumeData.size() != 0;
}


void RoiDrawPlugin::dataSetAdded(data::DataSet * ds)
{
	if (!ds)
		return;

	// Regions of Interest
	if (ds->getKind() == "regionOfInterest")
	{
		// Get the polydata of the ROI
		vtkPolyData * roiPolyData = ds->getVtkPolyData();

		if (!roiPolyData)
			return;

		// Add the data set to the list
		this->roiData.append(ds);

		// Build a pipeline for rendering the ROI in the 3D view
		vtkPolyDataMapper * mapper = vtkPolyDataMapper::New();
		mapper->ScalarVisibilityOff();
		mapper->SetInput(roiPolyData);

		// Create an actor for the ROI
		vtkActor * actor = vtkActor::New();
		actor->GetProperty()->SetColor(0.0, 1.0, 0.0);
		actor->GetProperty()->SetLineWidth(2.0);
		actor->SetMapper(mapper);
		mapper->Delete();

		// Create a new structure for the 2D tracers
		TracerInformation newTracers;

		// Initialize it, and add it to the list
		newTracers.widgets[0] = NULL;
		newTracers.widgets[1] = NULL;
		newTracers.widgets[2] = NULL;
		newTracers.currentView = -1;
		this->tracers.append(newTracers);

		// Temporary VTK object
		vtkObject * obj = NULL;

		// Check if the ROI data set contains a transformation matrix
		if (ds->getAttributes()->getAttribute("transformation matrix", obj))
		{
			// If so, add it to the actor
			actor->SetUserMatrix(vtkMatrix4x4::SafeDownCast(obj));
		}

		// Add the actor to the assembly
		this->assembly->AddPart(actor);
		actor->Delete();

		// Add the new data set to the combo box
		this->ui->roiComboBox->addItem(ds->getName());

		// Select the newly added ROI
		this->ui->roiComboBox->setCurrentIndex(this->ui->roiComboBox->count() - 1);

		// Enable or disable controls
		this->enableControls();

	} // if [ROIs]

	// DTI image
	else if (ds->getKind() == "DTI")
	{
		// Check if the data set contains a VTK image data object
		if (!(ds->getVtkImageData()))
			return;

		// Add the data set to the list
		this->volumeData.append(ds);

		// Enable or disable controls
		this->enableControls();

	} // else [DTI image]

	// Slice actors
	else if (ds->getKind() == "sliceActor")
	{
		// Get the VTK object
		vtkObject * obj = ds->getVtkObject();

		if (!obj)
			return;

		// Cast the object to an actor
		vtkImageSliceActor * actor = static_cast<vtkImageSliceActor *>(obj);

		if (!actor)
			return;

		// Get the slice orientation
		int dir = actor->GetSliceOrientation();

		// Store the pointer to the slice
		if (dir == 0 || dir == 1 || dir == 2)
		{
			this->slices[dir] = actor;
		}
	}
}


void RoiDrawPlugin::dataSetChanged(data::DataSet * ds)
{
	if (!ds)
		return;

	// Region of Interest
	if (ds->getKind() == "regionOfInterest" && this->roiData.contains(ds))
	{
		// Get the index of the ROI
		int roiIndex = this->roiData.indexOf(ds);

		// Update the name in the combo box
		this->ui->roiComboBox->setItemText(roiIndex, ds->getName());

		// Get the actor of the ROI
		vtkActor * actor = this->getROIActor(roiIndex);
		Q_ASSERT(actor);

		// Get the associated mapper
		vtkPolyDataMapper * mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
		Q_ASSERT(mapper);

		// Change the input of the mapper
		mapper->SetInput(ds->getVtkPolyData());

		// Update the transformation matrix, if present
		vtkObject * obj = NULL;
		if (ds->getAttributes()->getAttribute("transformation matrix", obj))
		{
			actor->SetUserMatrix(vtkMatrix4x4::SafeDownCast(obj));
		}

		// Re-render the scene
		this->core()->render();
	
	} // if [ROIs]

	// DTI image
	else if (ds->getKind() == "DTI" && this->volumeData.contains(ds))
	{
		// Get the index of the DTI image
		int dtiIndex = this->volumeData.indexOf(ds);

	} // else [DTI image]

	// Slice actors
	else if (ds->getKind() == "sliceActor")
	{
		// At the moment, we do not need to do anything here, since the pointer
		// to the slice actor will never change. If, in the future, one of the
		// plugins does modify this pointer, we here need to check for this
		// modification, and reset the active tracer if needed.
	}
}


void RoiDrawPlugin::dataSetRemoved(data::DataSet * ds)
{
	if (!ds)
		return;

	// Region of Interest
	if (ds->getKind() == "regionOfInterest" && this->roiData.contains(ds))
	{
		// Get the index of the ROI
		int roiIndex = this->roiData.indexOf(ds);

		// Deselect all ROIs
		this->selectROI(-1);

		// Get the actor, and remove it from the assembly
		vtkActor * actor = this->getROIActor(roiIndex);
		Q_ASSERT(actor);

		// Remove the ROI from the data set
		this->roiData.removeAt(roiIndex);

		// Delete the tracers of this ROI
		if (this->tracers[roiIndex].widgets[0])
			this->tracers[roiIndex].widgets[0]->Delete();
		if (this->tracers[roiIndex].widgets[1])
			this->tracers[roiIndex].widgets[1]->Delete();
		if (this->tracers[roiIndex].widgets[2])
			this->tracers[roiIndex].widgets[2]->Delete();
		this->tracers.remove(roiIndex);

		this->assembly->RemovePart(actor);

		// Remove the item from the GUI combo box
		this->ui->roiComboBox->removeItem(roiIndex);

		// Enable or disable controls
		this->enableControls();

		// Render the screen if we just removed the last ROI
		if (this->roiData.isEmpty())
			this->core()->render();

	} // if [ROIs]

	// DTI image
	else if (ds->getKind() == "DTI" && this->volumeData.contains(ds))
	{
		// Get the index of the DTI image
		int dtiIndex = this->volumeData.indexOf(ds);

		// Remove the DTI image from the list
		this->volumeData.removeAt(dtiIndex);

		// Get the image data pointer
		vtkImageData * dtiImage = ds->getVtkImageData();
		vtkObject * obj = NULL;

		// Enable or disable controls
		this->enableControls();

	} // else [DTI image]

	else if (ds->getKind() == "sliceActor")
	{
		// Get the VTK object
		vtkObject * obj = ds->getVtkObject();

		if (!obj)
			return;

		// Cast the object to an actor
		vtkImageSliceActor * actor = static_cast<vtkImageSliceActor *>(obj);

		if (!actor)
			return;

		// Get the slice orientation
		int dir = actor->GetSliceOrientation();

		// Set the pointer to NULL
		if (dir == 0 || dir == 1 || dir == 2)
		{
			this->slices[dir] = NULL;
		}

		// Disable the "Draw" button
		this->enableControls();
	}
}


void RoiDrawPlugin::selectROI(int index)
{
    // Check if we've got a selected ROI
	if (this->selectedRoi != -1)
	{
		// If so, deselect it
		vtkActor * actor = this->getROIActor(this->selectedRoi);
		Q_ASSERT(actor);
		actor->GetProperty()->SetColor(0.0, 1.0, 0.0);

		// Turn the tracers of the current ROI off
		if (this->tracers[this->selectedRoi].widgets[0])
			this->tracers[this->selectedRoi].widgets[0]->Off();
		if (this->tracers[this->selectedRoi].widgets[1])
			this->tracers[this->selectedRoi].widgets[1]->Off();
		if (this->tracers[this->selectedRoi].widgets[2])
			this->tracers[this->selectedRoi].widgets[2]->Off();

		// Clear the name edit control
		this->ui->nameEdit->setText("");

		this->selectedRoi = -1;
	}

	if (index < 0 || index >= this->roiData.size())
		return;

	// Select the new ROI
	this->selectedRoi = index;
	vtkActor * actor = this->getROIActor(this->selectedRoi);
	Q_ASSERT(actor);

	// Color the ROI
	actor->GetProperty()->SetColor(1.0, 1.0, 0.0);

	// Get the current view containing the tracing widget
	int currentView = this->tracers[this->selectedRoi].currentView;

	if (currentView != -1)
	{
		// If a tracing widget exists for this ROI, turn it on, but leave
		// the interaction off. This will make the widget visible in the
		// corresponding 2D view, but disables interaction until the user
		// presses the "Start drawing" button.

		this->tracers[this->selectedRoi].widgets[currentView]->On();
		this->tracers[this->selectedRoi].widgets[currentView]->InteractionOff();
	}

	// Copy the name of the selected ROI to the name edit control
	this->ui->nameEdit->setText(this->getSelectedRoi()->getName());

	// Get the ROI data set and its attributes
	data::DataSet * ds = this->roiData.at(selectedRoi);

	if (!ds)
		return;

	// Re-render the scene
	this->core()->render();
}


void RoiDrawPlugin::createNewROI()
{
	// Set the default name for the new ROI
	int numberOfROIs = this->roiData.size();
	QString newDataName = QString("New ROI %1").arg(numberOfROIs + 1);

	// Create an empty polydata object
	vtkPolyData * polyData = vtkPolyData::New();

	// Create a data set with the default name and the empty data
	data::DataSet * ds = new data::DataSet(newDataName, "regionOfInterest", polyData);
	polyData->Delete();

	// Add the new ROI data set to the data manager
	this->core()->data()->addDataSet(ds);
}


vtkActor * RoiDrawPlugin::getROIActor(int index)
{
	// Get the parts of the assembly
    vtkPropCollection * collection = this->assembly->GetParts();
   
	// Error checking
	if (!collection)
		return NULL;

    if (collection->GetNumberOfItems() != this->roiData.size())
		return NULL;

	if (index < 0 || index >= collection->GetNumberOfItems())
		return NULL;

	// Get the VTK object
    vtkObject * obj = collection->GetItemAsObject(index);

	if (!obj)
		return NULL;

	// Cast the object to an actor and return it
	return vtkActor::SafeDownCast(obj);
}


void RoiDrawPlugin::toggleDrawing()
{
	// Enable drawing
	if (this->isDrawing == false)
	{
		// Get the index of the current ROI
		int roiIndex = this->selectedRoi;

		if (roiIndex < 0 || roiIndex >= this->roiData.size() || this->roiData.size() != this->tracers.size())
			return;

		// Activate drawing
		this->isDrawing = true;

		// Select the current 2D view, or if no 2D is active, select the top one.
		if (this->tracers[roiIndex].currentView == -1)
		{
			if (this->currentSubCanvas != -1)
				this->activateSubCanvas(this->currentSubCanvas);
			else
				this->activateSubCanvas(2);
		}

		// If the user switched canvasses while drawing was off, we first switch to
		// the currently selected canvas.

		if (this->currentSubCanvas != -1)
		{
			this->activateSubCanvas(this->currentSubCanvas);
		}

		// Get the current tracer, and turn its interaction on
		vtkImageTracerWidget2 * tracer = this->tracers[roiIndex].widgets[this->tracers[roiIndex].currentView];
		tracer->InteractionOn();

		// Change the text of the drawing button
		this->ui->drawButton->setText("Stop Drawing");

		// Disable all other controls
		this->ui->deleteButton->setEnabled(false);
		this->ui->nameEdit->setEnabled(false);
		this->ui->newButton->setEnabled(false);
		this->ui->renameButton->setEnabled(false);
		this->ui->saveROIButton->setEnabled(false);
		this->ui->groupButton->setEnabled(false);
		this->ui->roiComboBox->setEnabled(false);
	}

	// Disable drawing
	else
	{
		int roiIndex = this->selectedRoi;

		if (roiIndex < 0 || roiIndex >= this->roiData.size() || this->roiData.size() != this->tracers.size())
			return;

		// Get the current tracer, and turn its interaction off
		vtkImageTracerWidget2 * tracer = this->tracers[roiIndex].widgets[this->tracers[roiIndex].currentView];
		tracer->InteractionOff();

		// Get the ROI drawn by the user
		vtkPolyData * drawnRoi = vtkPolyData::New();
		tracer->GetPath(drawnRoi);

		// Get the points of the ROI
		vtkPoints * points = drawnRoi->GetPoints();

		vtkIdList * roiList = vtkIdList::New();

		// Check if we've got points
		if (points)
		{
			double p[3];
			
			// Loop through all points
			for (vtkIdType ptId = 0; ptId < points->GetNumberOfPoints(); ++ptId)
			{
				// Re-order the points. By default, the tracing widget creates one line per
				// set of subsequent points. However, throughout the tool, we use one line per
				// ROI polygon. So, here we put all points of the ROI into one line array.

				points->GetPoint(ptId, p);
				roiList->InsertNextId(ptId);
			}

			// Set the new line
			vtkCellArray * lines = vtkCellArray::New();
			lines->InsertNextCell(roiList);
			drawnRoi->SetLines(lines);
			lines->Delete();
			roiList->Delete();

			// Get the slice location from the actor
			vtkImageSliceActor * actor = this->slices[this->tracers[roiIndex].currentView];
			int axis = actor->GetSliceOrientation();
			float depth = actor->GetSliceLocation();

			// Project the ROI to the current plane
			for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
			{
				points->GetPoint(i, p);
				p[axis] = depth;
				points->SetPoint(i, p);
			}

			// Update the ROI, using the transformation matrix	
			if (this->slices[0])
				this->updateROI(drawnRoi, this->slices[0]->GetMatrix());

			// Delete the polydata path
			drawnRoi->Delete(); 
			drawnRoi = NULL;
		}

		// Change the button text, and re-enable the controls
		this->ui->drawButton->setText("Start Drawing");
		this->enableControls();

		// Done drawing
		this->isDrawing = false;
	}
}


void RoiDrawPlugin::updateROI(vtkPolyData * newData, vtkMatrix4x4 * m)
{
	// Do nothing if no ROI has been selected
	if (this->selectedRoi == -1) 
		return;

	// Get the data set of the selected ROI
	data::DataSet * ds = this->getSelectedRoi();
	Q_ASSERT(ds);

	vtkObject * obj = NULL;

	// If we've got a transformation matrix...
	if (m)
	{
		// ...copy it to a new matrix...
		vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
		mCopy->DeepCopy(m);

		// ...and add it to the data set
		ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(mCopy));
	}

	// Update the polydata of the ROI
	ds->updateData(newData);

	// Signal the core that the data set has changed
	this->core()->data()->dataSetChanged(ds);
}


data::DataSet * RoiDrawPlugin::getSelectedRoi()
{
	// Return NULL if no ROI was selected
	if (this->selectedRoi == -1) 
		return NULL;

	// Get the data set of the selected ROI
	data::DataSet * ds = this->roiData.at(this->selectedRoi);

	Q_ASSERT(ds);
	return ds;
}


void RoiDrawPlugin::deleteRoi()
{
	// Get the data set of the selected ROI
	data::DataSet * ds = getSelectedRoi();

	if (!ds) 
		return;

	this->core()->data()->removeDataSet(ds);
}


void RoiDrawPlugin::saveROI()
{
	// Get the index of the current ROI
	int roiIndex = this->ui->roiComboBox->currentIndex();

	if (roiIndex < 0 || roiIndex >= this->roiData.size())
		return;

	// Get the current data directory
	QDir dataDir = this->core()->getDataDirectory();

	// Open a file dialog to get a filename
	QString fileName = QFileDialog::getSaveFileName(NULL, "Write ROI", dataDir.absolutePath(), " Region of Interest (*.pol)");
	
	// Check if the filename is correct
	if (fileName.isEmpty())
		return;

	// Convert the QString to a character array
	QByteArray ba = fileName.toAscii();
	char * fileNameChar = ba.data();

	// Get the roiPolyData object containing the fibers
	vtkPolyData * output = this->roiData.at(roiIndex)->getVtkPolyData();

	// Check if the fibers exist
	if (!output)
		return;

	// Create a roiPolyData writer
	vtkPolyDataWriter * writer = vtkPolyDataWriter::New();

	// Configure the writer
	writer->SetFileName(fileNameChar);
	writer->SetInput(output);
	writer->SetFileTypeToASCII();

	// Write output file
	writer->Write();

	// Delete the writer
	writer->Delete();

	vtkObject * attObject;

	// Check if the ROI contains a transformation matrix
	if (this->roiData.at(roiIndex)->getAttributes()->getAttribute("transformation matrix", attObject))
	{
		std::string err = "";

		// If so, write the matrix to a ".tfm" file
		bool success = TransformationMatrixIO::writeMatrix(std::string(fileNameChar), vtkMatrix4x4::SafeDownCast(attObject), err);

		// Display error messages if necessary
		if (!success)
		{
			this->core()->out()->showMessage(QString(err.c_str()));
		}
	}
}


void RoiDrawPlugin::rename()
{
	data::DataSet * ds = getSelectedRoi();

	if (!ds) 
		return;

	QString newName = this->ui->nameEdit->text();

	// Change the name of the data set
	ds->setName(newName);

	this->core()->data()->dataSetChanged(ds);
}


void RoiDrawPlugin::groupROIs()
{
	// Create a new ROI grouping dialog
	ROIGroupDialog * groupDialog = new ROIGroupDialog;
	groupDialog->setFixedSize(300, 400);

	// Create a list of the names of the available ROIs, and add them to the dialog
	QStringList roiNames;
	for (int roiDSIndex = 0; roiDSIndex < this->roiData.size(); ++roiDSIndex)
	{
		roiNames.append(this->roiData.at(roiDSIndex)->getName());
		groupDialog->addROIName(this->roiData.at(roiDSIndex)->getName());
	}

	// Find the first default name that is not yet in use
	QString iString;
	for (int i = 1; i < 100; ++i)
	{
		// Set the string to "ROI Group #"
		iString.setNum(i);
		QString defaultName = "ROI Group " + iString;

		// If the string is not yet taken, copy it to the dialog
		if (!(roiNames.contains(defaultName)))
		{
			groupDialog->setDefaultName(defaultName);
			break;
		}
	}

	// Execute the dialog. If the user clicks "Cancel", return here.
	if (groupDialog->exec() == 0)
	{
		delete groupDialog;
		return;
	}

	// Get the indices of the ROIs selected by the user
	QList<int> selectedROIs = groupDialog->getSelectedROIs();

	// We should have at least one index
	if (selectedROIs.isEmpty())
	{
		delete groupDialog;
		return;
	}

	// Get the data set pointers of the selected ROIs
	QList<data::DataSet *> selectedROIsDS;
	for (int i = 0; i < selectedROIs.size(); ++i)
	{
		selectedROIsDS.append(this->roiData.at(selectedROIs.at(i)));
	}

	// Create a filter for merging the polydata
	vtkAppendPolyData * appendPD = vtkAppendPolyData::New();

	// Add all selected ROIs as input to the filter
	for (int i = 0; i < selectedROIsDS.size(); ++i)
	{
		vtkPolyData * roiPD = selectedROIsDS.at(i)->getVtkPolyData();

		if (roiPD)
		{
			appendPD->AddInput(roiPD);
		}
	}

	// Update the filter, and get its output
	appendPD->Update();
	vtkPolyData * mergedPD = appendPD->GetOutput();

	// Create a new data set for the merged ROIs
	data::DataSet * mergedDS = new data::DataSet(groupDialog->getGroupName(), "regionOfInterest", mergedPD);

	// Copy the transformation matrix of the first selected ROI to the output
	// ROI. We do not check whether all transformation matrices are the same.
	mergedDS->getAttributes()->copyTransformationMatrix(selectedROIsDS.at(0));

	// Add the data set to the data manager
	this->core()->data()->addDataSet(mergedDS);
	appendPD->Delete();

	// If necessary, delete the input ROIs
	if (groupDialog->getDeleteInputROIs())
	{
		for (int i = 0; i < selectedROIs.size(); ++i)
		{
			this->core()->data()->removeDataSet(selectedROIsDS.at(i));
		}
	}
	
	// Done, delete the grouping dialog
	delete groupDialog;
}


vtkSubCanvas * RoiDrawPlugin::getSubcanvas(int i)
{
	// Return the required 2D subcanvas
	return this->fullCore()->canvas()->GetSubCanvas2D(i);
}


void RoiDrawPlugin::activateSubCanvas(int canvasID)
{
	// We can only switch canvasses if we're currently drawing a ROI
	if (!(this->isDrawing))
	{
		this->currentSubCanvas = canvasID;
		return;
	}

	// Check the index of the selected ROI
	if (this->tracers.size() == 0 || this->selectedRoi < 0 || this->selectedRoi >= this->tracers.size())
		return;

	// Get the current view containing the active tracing widget
	int prevView = this->tracers[this->selectedRoi].currentView;

	// Do nothing if the view has not changed
	if (canvasID == prevView)
		return;

	// Get a pointer to the widget array
	vtkImageTracerWidget2 ** tracerWidgets = this->tracers[this->selectedRoi].widgets;

	// Delete the widget in the current view
	if (prevView != -1)
	{
		if (tracerWidgets[prevView])
		{
			tracerWidgets[prevView]->InteractionOff();
			tracerWidgets[prevView]->Delete();
			tracerWidgets[prevView] = NULL;
		}
	}

	// Create a new widget in the target canvas
	tracerWidgets[canvasID] = vtkImageTracerWidget2::New();

	// High priority ensures correct interaction handling
	tracerWidgets[canvasID]->SetPriority(1.0);

	// Use the interactor of the 2D subcanvasses
	tracerWidgets[canvasID]->SetInteractor(this->fullCore()->canvas()->GetSubCanvas2D(canvasID)->GetInteractor());

	// Set the view prop, which is one of the orthogonal slices
	if (this->slices[canvasID])
		tracerWidgets[canvasID]->SetViewProp(this->slices[canvasID]);

	// Set the renderer
	tracerWidgets[canvasID]->SetDefaultRenderer(this->fullCore()->canvas()->GetSubCanvas2D(canvasID)->GetRenderer());
	tracerWidgets[canvasID]->SetCurrentRenderer(this->fullCore()->canvas()->GetSubCanvas2D(canvasID)->GetRenderer());

	// Configure the behavior and looks of the widget
	tracerWidgets[canvasID]->SetCaptureRadius(1000);
	tracerWidgets[canvasID]->AutoCloseOn();
	tracerWidgets[canvasID]->GetLineProperty()->SetColor(1.0, 1.0, 1.0);
	tracerWidgets[canvasID]->GetLineProperty()->SetLineWidth(2.0);
	tracerWidgets[canvasID]->GetGlyphSource()->SetColor(1, 0, 0);
	tracerWidgets[canvasID]->GetGlyphSource()->SetGlyphTypeToThickCross();
	tracerWidgets[canvasID]->GetGlyphSource()->SetScale(5.0);

	// Do not project the ROI to a plane
	tracerWidgets[canvasID]->ProjectToPlaneOff();

	// Use the normal of the selected plane
	tracerWidgets[canvasID]->SetProjectionNormal(canvasID);
	tracerWidgets[canvasID]->SetProjectionPosition(0);

	// Use the matrix of the selected slice, if available
	if (this->slices[0])
	{
		if (this->slices[0]->GetMatrix())
			tracerWidgets[canvasID]->setMatrix(this->slices[0]->GetMatrix(), this->slices[canvasID]->GetPlaneNormal());
	}

	// Turn the widget on
	tracerWidgets[canvasID]->On();
	tracerWidgets[canvasID]->InteractionOn();

	// Store the current view
	this->tracers[this->selectedRoi].currentView = canvasID;

	// We no longer need to switch the canvas
	this->currentSubCanvas = -1;
}


void RoiDrawPluginCallback::Execute(vtkObject * caller, unsigned long event, void * callData)
{
	// Handle the event of selecting a new subcanvas
	if (event == vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVAS_SELECTED)
	{
		// Index of selected 2D subcanvas, or "-1" if no subcanvas is selected
		int selected2DCanvas = -1;

		// Find the index of the selected subcanvas
		if (this->plugin->getSubcanvas(0) == (vtkSubCanvas *) callData)
			selected2DCanvas = 0;
		else if (this->plugin->getSubcanvas(1) == (vtkSubCanvas *) callData)
			selected2DCanvas = 1;
		else if (this->plugin->getSubcanvas(2) == (vtkSubCanvas *) callData)
			selected2DCanvas = 2;

		// Activate the selected canvas
		if (selected2DCanvas != -1)
			this->plugin->activateSubCanvas(selected2DCanvas);

		this->plugin->currentSubCanvas = selected2DCanvas;
	}
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libRoiDrawPlugin, bmia::RoiDrawPlugin)
