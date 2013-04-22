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
 * PlanesVisPlugin.h
 *
 * 2010-03-11	Tim Peeters
 * - First version
 *
 * 2010-09-17	Tim Peeters
 * - Make it an AdvancedPlugin subclass
 *   so I can add sliced views to the 2D subcanvasses.
 *
 * 2011-03-01	Evert van Aart
 * - Version 1.0.0.
 * - Fixed 2D view cameras for images with a transformation matrix.
 * - Fixed resetting the 2D cameras for transformed images.
 *
 * 2011-04-14	Evert van Aart
 * - Version 1.1.0.
 * - Redesigned UI to make the distinction between LUT-based coloring and RGB-based
 *   coloring more intuitive.
 * - Added comments, redesigned some parts of the code.
 * - Implemented "dataSetChanged" and "dataSetRemoved".
 * - Avoid unnecessary re-renders when switching images.
 * - When applying weighting to RGB colors, weights are not automatically normalized
 *   to the range 0-1.
 *
 * 2011-05-02	Evert van Aart
 * - Version 1.1.1.
 * - Correctly update seed points when a new volume is selected.
 *
 */


/** Includes */

#include "PlanesVisPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\


PlanesVisPlugin::PlanesVisPlugin() : AdvancedPlugin("Planes")
{

	// Create a planes actor, and hide it for now
	this->actor = vtkImageOrthogonalSlicesActor::New();
	this->actor->VisibilityOff();

	// Setup the GUI
	this->qWidget = new QWidget();
	this->ui = new Ui::PlanesVisualizationForm();
	this->ui->setupUi(this->qWidget);

	// Create the MEV coloring filter
	this->MEVColoringFilter = vtkMEVColoringFilter::New();

	// Add default items to combo boxes
	this->ui->dtiWeightCombo->addItem("None");
	this->ui->lutCombo->addItem("Default");
	
	// Disable the DTI controls by default
	this->ui->dtiRadio->setEnabled(false);
	this->ui->dtiVolumeLabel->setEnabled(false);
	this->ui->dtiVolumeCombo->setEnabled(false);
	this->ui->dtiWeightLabel->setEnabled(false);
	this->ui->dtiWeightCombo->setEnabled(false);

	// Create a default Look-Up Table (black to white in the range 0-1)
	this->defaultLUT = vtkLookupTable::New();
	this->defaultLUT->SetValueRange(0.0, 1.0);
	this->defaultLUT->SetSaturationRange(0.0, 0.0);
	this->defaultLUT->SetAlphaRange(1.0, 1.0);

	// Turn actor interpolation on or off, depending on the default GUI settings
	this->actor->SetInterpolate(this->ui->interpolationCheck->isChecked() ? 1 : 0);

	// No callback yet, we create it in the "init" function
	this->callBack = NULL;

	// Connect the controls
	this->connectControls(true);
}


//---------------------------------[ init ]--------------------------------\\

void PlanesVisPlugin::init()
{
	// Create one seed point set for each plane
	this->seedsX = vtkPoints::New();
	this->seedsY = vtkPoints::New();
	this->seedsZ = vtkPoints::New();

	vtkUnstructuredGrid * pointSetX = vtkUnstructuredGrid::New();
	vtkUnstructuredGrid * pointSetY = vtkUnstructuredGrid::New();
	vtkUnstructuredGrid * pointSetZ = vtkUnstructuredGrid::New();

	pointSetX->SetPoints(this->seedsX);
	pointSetY->SetPoints(this->seedsY);
	pointSetZ->SetPoints(this->seedsZ);

	// Create data sets for the seed points
	this->seedDataSets[0] = new data::DataSet("Plane X", "seed points", pointSetX);
	this->seedDataSets[1] = new data::DataSet("Plane Y", "seed points", pointSetY);
	this->seedDataSets[2] = new data::DataSet("Plane Z", "seed points", pointSetZ);

	// Reduce reference count of the point sets to one
	pointSetX->Delete();
	pointSetY->Delete();
	pointSetZ->Delete();

	// Get the canvas
	vtkMedicalCanvas * canvas = this->fullCore()->canvas();

	QString sliceActorNames[3];
	sliceActorNames[0] = "X Plane";
	sliceActorNames[1] = "Y Plane";
	sliceActorNames[2] = "Z Plane";

	// Loop through all three axes
	for(int axis = 0; axis < 3; ++axis)
	{
		// Add the seed point data set to the data manager
		this->core()->data()->addDataSet(this->seedDataSets[axis]);

		// Add the planes to their respective 2D views
		vtkImageSliceActor * sliceActor = this->actor->GetSliceActor(axis);
		canvas->GetSubCanvas2D(axis)->GetRenderer()->AddActor(sliceActor);

		// Add the slice actors to the data manager
		this->sliceActorDataSets[axis] = new data::DataSet(sliceActorNames[axis], "sliceActor", sliceActor);
		this->core()->data()->addDataSet(this->sliceActorDataSets[axis]);
	}

	// Create the callback class
	this->callBack = PlanesVisPluginCallback::New();
	this->callBack->plugin = this;

	// Add the callback to the canvas as an observer
	this->fullCore()->canvas()->AddObserver(vtkCommand::UserEvent + 
			BMIA_USER_EVENT_SUBCANVAS_CAMERA_RESET, this->callBack);
}


//------------------------------[ Destructor ]-----------------------------\\

PlanesVisPlugin::~PlanesVisPlugin()
{
	// Get the canvas
	vtkMedicalCanvas * canvas = this->fullCore()->canvas();

	// Loop through the three axes
	for (int axis = 0; axis < 3; ++axis)
	{
		// For each axis, remove the corresponding plane from the 2D view...
		canvas->GetSubCanvas2D(axis)->GetRenderer()->RemoveActor(this->actor->GetSliceActor(axis));

		// ...and remove the seed point and slice actor data sets
		this->core()->data()->removeDataSet(this->seedDataSets[axis]);
		this->core()->data()->removeDataSet(this->sliceActorDataSets[axis]);
	}

	// Delete the seed points
	this->seedsX->Delete();
	this->seedsY->Delete();
	this->seedsZ->Delete();

	delete this->qWidget;
	this->actor->Delete();
	this->defaultLUT->Delete();
	this->MEVColoringFilter->Delete();

	// Delete the callback
	if (this->callBack)
		this->callBack->Delete();
}


//---------------------------[ connectControls ]---------------------------\\

void PlanesVisPlugin::connectControls(bool doConnect)
{
	// Connect all controls to their respective slot functions
	if (doConnect)
	{
		connect(this->ui->scalarVolumeCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(changeScalarVolume(int))	);
		connect(this->ui->lutCombo,				SIGNAL(currentIndexChanged(int)),	this, SLOT(changeLUT(int))			);
		connect(this->ui->dtiVolumeCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(changeDTIVolume(int))	);
		connect(this->ui->dtiWeightCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(changeWeightVolume(int))	);
		connect(this->ui->scalarVolumeRadio,	SIGNAL(clicked()),					this, SLOT(applyLUTColoring())		);
		connect(this->ui->dtiRadio,				SIGNAL(clicked()),					this, SLOT(applyRGBColoring())		);
		connect(this->ui->xPositionSpin,		SIGNAL(valueChanged(int)),			this, SLOT(setXSlice(int))			);
		connect(this->ui->yPositionSpin,		SIGNAL(valueChanged(int)),			this, SLOT(setYSlice(int))			);
		connect(this->ui->zPositionSpin,		SIGNAL(valueChanged(int)),			this, SLOT(setZSlice(int))			);
		connect(this->ui->interpolationCheck,	SIGNAL(toggled(bool)),				this, SLOT(setInterpolation(bool))	);
		connect(this->ui->xVisibleCheck,		SIGNAL(toggled(bool)),				this, SLOT(setXVisible(bool))		);
		connect(this->ui->yVisibleCheck,		SIGNAL(toggled(bool)),				this, SLOT(setYVisible(bool))		);
		connect(this->ui->zVisibleCheck,		SIGNAL(toggled(bool)),				this, SLOT(setZVisible(bool))		);
	}
	// Disconnect all signals
	else
	{
		disconnect(this->ui->scalarVolumeCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(changeScalarVolume(int))	);
		disconnect(this->ui->lutCombo,			SIGNAL(currentIndexChanged(int)),	this, SLOT(changeLUT(int))			);
		disconnect(this->ui->dtiVolumeCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(changeDTIVolume(int))	);
		disconnect(this->ui->dtiWeightCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(changeWeightVolume(int))	);
		disconnect(this->ui->scalarVolumeRadio,	SIGNAL(clicked()),					this, SLOT(applyLUTColoring())		);
		disconnect(this->ui->dtiRadio,			SIGNAL(clicked()),					this, SLOT(applyRGBColoring())		);
		disconnect(this->ui->xPositionSpin,		SIGNAL(valueChanged(int)),			this, SLOT(setXSlice(int))			);
		disconnect(this->ui->yPositionSpin,		SIGNAL(valueChanged(int)),			this, SLOT(setYSlice(int))			);
		disconnect(this->ui->zPositionSpin,		SIGNAL(valueChanged(int)),			this, SLOT(setZSlice(int))			);
		disconnect(this->ui->interpolationCheck,SIGNAL(toggled(bool)),				this, SLOT(setInterpolation(bool))	);
		disconnect(this->ui->xVisibleCheck,		SIGNAL(toggled(bool)),				this, SLOT(setXVisible(bool))		);
		disconnect(this->ui->yVisibleCheck,		SIGNAL(toggled(bool)),				this, SLOT(setYVisible(bool))		);
		disconnect(this->ui->zVisibleCheck,		SIGNAL(toggled(bool)),				this, SLOT(setZVisible(bool))		);
	}
}


//------------------------------[ getVtkProp ]-----------------------------\\

vtkProp * PlanesVisPlugin::getVtkProp()
{
	return this->actor;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * PlanesVisPlugin::getGUI()
{
	return this->qWidget;
}


//----------------------------[ dataSetAdded ]-----------------------------\\

void PlanesVisPlugin::dataSetAdded(data::DataSet * ds)
{
	// Scalar volume
	if (ds->getKind() == "scalar volume")
	{
		this->connectControls(false);

		// Add the scalar volume to the list
		this->scalarVolumeDataSets.append(ds);
			
		// Add the scalar volume to both relevant combo boxes
		this->ui->scalarVolumeCombo->addItem(ds->getName());
		this->ui->dtiWeightCombo->addItem(ds->getName());

		// If this is the first scalar volume we added, select it now
		if (this->ui->scalarVolumeCombo->count() == 1)
		{
			this->changeScalarVolume(0);

			// Reset the camera of the 3D volume
			if (this->ui->scalarVolumeRadio->isChecked())
				this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();
		}

		// Likewise for weighting volumes, with the addendum that we need to select
		// it manually (since the combo box already contained the "None" item.

		if (this->ui->dtiWeightCombo->count() == 2)
		{
			this->ui->dtiWeightCombo->setCurrentIndex(1);
			this->changeWeightVolume(1);
		}
			//stephen remove
		//ds->getAttributes()->addAttribute("SlicePosX",0);
		//ds->getAttributes()->addAttribute("SlicePosY",0);
		//ds->getAttributes()->addAttribute("SlicePosZ",0);
		this->connectControls(true);

	} // if [scalar volume]

	// Transfer Functions (LUTs)
	else if (ds->getKind() == "transfer function")
	{
		this->connectControls(false);

		// Add data set to the list and to the GUI
		this->lutDataSets.append(ds);
		this->ui->lutCombo->addItem(ds->getName());

		this->connectControls(true);
	} 

	// Eigensystem Data
	else if (ds->getKind() == "eigen")
	{
		this->connectControls(false);

		// Add the eigensystem (DTI) data set to the list
		this->dtiDataSets.append(ds);

		// Enable the radio button for RGB coloring
		this->ui->dtiRadio->setEnabled(true);

		// Add the data set to the combo box
		this->ui->dtiVolumeCombo->addItem(ds->getName());

		// If this is the first DTI set, we switch to RGB coloring
		if (this->ui->dtiVolumeCombo->count() == 1)
		{
			this->ui->dtiRadio->setChecked(true);
			this->applyRGBColoring();

			// Reset the camera of the 3D volume
			this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();
		}

		this->connectControls(true);
	}
}


//---------------------------[ dataSetChanged ]----------------------------\\

void PlanesVisPlugin::dataSetChanged(data::DataSet * ds)
{
	// General behavior: For each of the input data types, it updates the name in
	// the GUI combo boxes, and if the data set is selected in one of these combo
	// boxes, it also calls the corresponding update function ("changeX").

	// Scalar Volumes
	if(ds->getKind() == "scalar volume" && this->scalarVolumeDataSets.contains(ds))
	{
		this->connectControls(false);
		int dsIndex = this->scalarVolumeDataSets.indexOf(ds);

		this->ui->scalarVolumeCombo->setItemText(dsIndex, ds->getName());
		this->ui->dtiWeightCombo->setItemText(dsIndex + 1, ds->getName());

		if (this->ui->scalarVolumeCombo->currentIndex() == dsIndex)
			this->changeScalarVolume(dsIndex);
		
		if (this->ui->dtiWeightCombo->currentIndex() == dsIndex + 1)
			this->changeWeightVolume(dsIndex + 1);
	
		this->connectControls(true);
		//stephen
		//int pos[3];
		//ds->getAttributes()->getAttribute("SlicePosX",pos[0]);
		//ds->getAttributes()->getAttribute("SlicePosY",pos[1]);
		//ds->getAttributes()->getAttribute("SlicePosZ",pos[2]);
		//cout << pos[0] << " " << pos[1] << " " << pos[2] << endl;
		//emit this->setXSlice(pos[0]);
		//emit this->setYSlice(pos[1]);
		//emit this->setZSlice(pos[2]);
	}

	// Transfer Functions
	else if (ds->getKind() == "transfer function" && this->lutDataSets.contains(ds))
	{
		this->connectControls(false);
		int dsIndex = this->lutDataSets.indexOf(ds);

		this->ui->lutCombo->setItemText(dsIndex + 1, ds->getName());

		if (this->ui->lutCombo->currentIndex() == (dsIndex + 1))
			this->changeLUT(dsIndex + 1);

		this->connectControls(true);
	}

	// DTI Eigensystem
	else if (ds->getKind() == "eigen" && this->dtiDataSets.contains(ds))
	{
		this->connectControls(false);
		int dsIndex = this->dtiDataSets.indexOf(ds);

		this->ui->dtiVolumeCombo->setItemText(dsIndex, ds->getName());

		if (this->ui->dtiVolumeCombo->currentIndex() == dsIndex)
			this->changeDTIVolume(dsIndex);

		this->connectControls(true);
	}
}

	
//----------------------------[ dataSetRemoved ]---------------------------\\

void PlanesVisPlugin::dataSetRemoved(data::DataSet * ds)
{
	// General behavior: Remove the item from the GUI combo boxes and from the
	// data set lists. If the removed data set was selected in one of the combo 
	// boxes, reset the index of that combo box to a default value, and update the 
	// planes for this new index.

	// Scalar Volumes
	if (ds->getKind() == "scalar volume" && this->scalarVolumeDataSets.contains(ds))
	{
		this->connectControls(false);
		int dsIndex = this->scalarVolumeDataSets.indexOf(ds);

		bool dataSetWasSelectedScalar = this->ui->scalarVolumeCombo->currentIndex() == dsIndex;
		bool dataSetWasSelectedWeight = this->ui->dtiWeightCombo->currentIndex() == (dsIndex + 1);

		this->ui->scalarVolumeCombo->removeItem(dsIndex);
		this->ui->dtiWeightCombo->removeItem(dsIndex + 1);

		this->scalarVolumeDataSets.removeAt(dsIndex);

		if (dataSetWasSelectedScalar)
		{
			// Reset to the first data set, or (if none present), deselect all data sets
			int newIndex = (this->ui->scalarVolumeCombo->count() > 0) ? 0 : -1;
			this->ui->scalarVolumeCombo->setCurrentIndex(newIndex);
			this->changeScalarVolume(newIndex);
		}

		if (dataSetWasSelectedWeight)
		{
			// Always reset to the first index ("None")
			this->ui->dtiWeightCombo->setCurrentIndex(0);
			this->changeWeightVolume(0);
		}

		this->connectControls(true);
	}

	// Transfer Functions
	else if (ds->getKind() == "transfer function" && this->lutDataSets.contains(ds))
	{
		this->connectControls(false);
		int dsIndex = this->lutDataSets.indexOf(ds);

		bool dataSetWasSelected = this->ui->lutCombo->currentIndex() == (dsIndex + 1);

		this->ui->lutCombo->removeItem(dsIndex + 1);
		this->lutDataSets.removeAt(dsIndex);

		if (dataSetWasSelected)
		{
			// Always reset to the first index ("Default")
			this->ui->lutCombo->setCurrentIndex(0);
			this->changeLUT(0);
		}

		this->connectControls(true);
	}

	// DTI Eigensystem
	else if (ds->getKind() == "eigen" && this->dtiDataSets.contains(ds))
	{
		this->connectControls(false);
		int dsIndex = this->dtiDataSets.indexOf(ds);

		bool dataSetWasSelected = this->ui->dtiVolumeCombo->currentIndex() == dsIndex;

		this->ui->dtiVolumeCombo->removeItem(dsIndex);
		this->dtiDataSets.removeAt(dsIndex);

		if (dataSetWasSelected)
		{
			// Reset to the first data set, or (if none present), deselect all data sets
			int newIndex = (this->ui->dtiVolumeCombo->count() > 0) ? 0 : -1;
			this->ui->dtiVolumeCombo->setCurrentIndex(newIndex);
			this->changeDTIVolume(newIndex);
		}

		this->connectControls(true);
	}
}


//--------------------------[ changeScalarVolume ]-------------------------\\

void PlanesVisPlugin::changeScalarVolume(int index)
{
	// Only do this is we're currently scowing a scalar volume
	if (!(this->ui->scalarVolumeRadio->isChecked()))
		return;

	// If the index is out of range, simply hide the actor
	if (index < 0 || index >= this->scalarVolumeDataSets.size())
	{
		this->actor->VisibilityOff();
		this->actor->SetInput(NULL);
		this->core()->render();
		return;
	}

	// Get the scalar volume data set
	data::DataSet * ds = this->scalarVolumeDataSets.at(index);

	if (!ds)
		return;

	if (!(ds->getVtkImageData()))
		return;

	// No rendering until we're done here
	this->core()->disableRendering();

	// If the memory size of the image is zero, ask it to update. This is done to
	// deal with images that are only computed on request, like the Anisotropy
	// Measure images.

	if (ds->getVtkImageData()->GetActualMemorySize() == 0)
	{
		ds->getVtkImageData()->Update();
		this->core()->data()->dataSetChanged(ds);
	}

	// Use the image as the input for the actor
	this->actor->SetInput(ds->getVtkImageData());
	// qDebug() << ds->getName() << " : " << ds->getKind() << "#ofComponents: " << ds->getVtkImageData()->GetNumberOfScalarComponents() << endl; 
	// Set transformation matrix, reset slices
	this->configureNewImage(ds);

	// Recompute the default LUT, since the scalar range may have changed
	if (this->ui->lutCombo->currentIndex() == 0)
		this->changeLUT(0);

	// Render the scene
	this->actor->VisibilityOn();
	this->actor->UpdateInput();
	this->core()->enableRendering();
	this->core()->render();
}


//------------------------------[ changeLUT ]------------------------------\\

void PlanesVisPlugin::changeLUT(int index)
{
	// Only do this is we're currently scowing a scalar volume
	if (!(this->ui->scalarVolumeRadio->isChecked()))
		return;

	// If the index is out of range, simply hide the actor
	if (index < 0 || index >= this->lutDataSets.size() + 1)
	{
		this->actor->VisibilityOff();
		this->actor->SetInput(NULL);
		this->core()->render();
		return;
	}

	// If we selected the "Default" LUT...
	if (index == 0)
	{
		// Get the scalar volume data set
		data::DataSet * ds = this->scalarVolumeDataSets.at(this->ui->scalarVolumeCombo->currentIndex());

		if (!ds)
			return;

		if (!(ds->getVtkImageData()))
			return;

		// Use the scalar range of the image for the default LUT
		double range[2];
		ds->getVtkImageData()->GetScalarRange(range);
		this->defaultLUT->SetTableRange(range);
		this->defaultLUT->Build();
		this->actor->SetLookupTable(this->defaultLUT);
	}
	else
	{
		// Get the LUT Transfer Function
		data::DataSet * ds = this->lutDataSets.at(index - 1);

		if (!ds)
			return;

		vtkColorTransferFunction * vtkTf = vtkColorTransferFunction::SafeDownCast(ds->getVtkObject());
		
		if (!vtkTf)
			return;

		this->actor->SetLookupTable(vtkTf);
	}

	// Render the scene using the new LUT
	this->actor->MapColorScalarsThroughLookupTableOn();
	this->core()->render();

	// Signal the data manager that the slice actors have been modified
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[0]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[1]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[2]);
}


//---------------------------[ changeDTIVolume ]---------------------------\\

void PlanesVisPlugin::changeDTIVolume(int index)
{
	// Do nothing if we're not using RGB coloring
	if (!this->ui->dtiRadio->isChecked()) 
		return;

	// If the index is out of range, simply hide the actor
	if (index < 0 || index >= this->dtiDataSets.size())
	{
		this->actor->VisibilityOff();
		this->actor->SetInput(NULL);
		this->core()->render();
		return;
	}

	// No rendering while we set up the new image
	this->core()->disableRendering();

	// If the current weighting volume does not match the dimensions of the new
	// DTI volume, set the weighting volume to "None".

	if (!(this->checkWeightVolumeMatch()))
	{
		this->ui->dtiWeightCombo->setCurrentIndex(0);
	}

	// Get the new DTI data set
	data::DataSet * ds = this->dtiDataSets.at(index);

	if (!ds)
	{
		this->core()->enableRendering();
		return;
	}

	vtkImageData * image = ds->getVtkImageData();
	
	if (!image)
	{
		this->core()->enableRendering();
		return;
	}

	// Update the RGB filter
	this->MEVColoringFilter->SetInput(image);
	this->MEVColoringFilter->UpdateWholeExtent();
	this->MEVColoringFilter->Update();

	// Update the actor
	this->actor->SetInput(this->MEVColoringFilter->GetOutput());
	this->actor->MapColorScalarsThroughLookupTableOff();
	this->actor->UpdateInput();

	// Configure planes and transformation matrices for the new image
	this->configureNewImage(ds);

	// Signal the data manager that the slice actors have been modified
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[0]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[1]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[2]);

	// Render the scene
	this->actor->VisibilityOn();
	this->core()->enableRendering();
	this->core()->render();
}


//--------------------------[ changeWeightVolume ]-------------------------\\

void PlanesVisPlugin::changeWeightVolume(int index)
{
	// Do nothing if we're not using RGB coloring
	if (!this->ui->dtiRadio->isChecked()) 
		return;

	// If the weight volume has been set to "None"...
	if (index == 0)
	{
		// ...update the RGB filter accordingly...
		this->MEVColoringFilter->SetWeightingVolume(NULL);
		this->MEVColoringFilter->Modified();
		this->MEVColoringFilter->Update();

		// ...and render the scene
		this->core()->render();

		return;
	}

	// Get the scalar volume
	data::DataSet * ds = this->scalarVolumeDataSets.at(index - 1);

	if (!ds)
		return;

	if (!(ds->getVtkImageData()))
		return;

	// If the memory size of the image is zero, ask it to update. This is done to
	// deal with images that are only computed on request, like the Anisotropy
	// Measure images.

	if (ds->getVtkImageData()->GetActualMemorySize() == 0)
	{
		ds->getVtkImageData()->Update();
		this->core()->data()->dataSetChanged(ds);
	}

	// If this weight volume does not match the DTI volume, report this to the user,
	// and set the weight volume to "None".

	if (!(this->checkWeightVolumeMatch()))
	{
		QMessageBox::warning(this->getGUI(), "Planes Visualization", "Dimension mismatch between DTI volume and weighting volume!");
		this->ui->dtiWeightCombo->setCurrentIndex(0);
		return;
	}


	// Update the RGB filter
	this->MEVColoringFilter->SetWeightingVolume(ds->getVtkImageData());
	this->MEVColoringFilter->Modified();
	this->MEVColoringFilter->Update();

	// Update the actor
	this->actor->SetInput(this->MEVColoringFilter->GetOutput());
	this->actor->MapColorScalarsThroughLookupTableOff();
	this->actor->UpdateInput();

	// Signal the data manager that the slice actors have been modified
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[0]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[1]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[2]);

	// Done, render the scene
	this->core()->render();

}


//--------------------------[ configureNewImage ]--------------------------\\

void PlanesVisPlugin::configureNewImage(data::DataSet * ds)
{
	// Check if the range of the image has changed
	bool resetSlices = false;
	if (this->actor->GetXMin() != this->ui->xPositionSpin->minimum())	resetSlices = true;
	if (this->actor->GetXMax() != this->ui->xPositionSpin->maximum())	resetSlices = true;
	if (this->actor->GetYMin() != this->ui->yPositionSpin->minimum())	resetSlices = true;
	if (this->actor->GetYMax() != this->ui->yPositionSpin->maximum())	resetSlices = true;
	if (this->actor->GetZMin() != this->ui->zPositionSpin->minimum())	resetSlices = true;
	if (this->actor->GetZMax() != this->ui->zPositionSpin->maximum())	resetSlices = true;

	// If so, reset the slices to the center, and update the GUI accordingly
	if (resetSlices)
	{
		this->connectControls(false);

		this->actor->CenterSlices();
		int x = this->actor->GetX();
		int y = this->actor->GetY();
		int z = this->actor->GetZ();

		this->ui->xPositionSpin-> setMinimum(this->actor->GetXMin());
		this->ui->xPositionSlide->setMinimum(this->actor->GetXMin());
		this->ui->xPositionSpin-> setMaximum(this->actor->GetXMax());
		this->ui->xPositionSlide->setMaximum(this->actor->GetXMax());
		this->ui->xPositionSpin-> setValue(x);

		this->ui->yPositionSpin-> setMinimum(this->actor->GetYMin());
		this->ui->yPositionSlide->setMinimum(this->actor->GetYMin());
		this->ui->yPositionSpin-> setMaximum(this->actor->GetYMax());
		this->ui->yPositionSlide->setMaximum(this->actor->GetYMax());
		this->ui->yPositionSpin-> setValue(y);

		this->ui->zPositionSpin-> setMinimum(this->actor->GetZMin());
		this->ui->zPositionSlide->setMinimum(this->actor->GetZMin());
		this->ui->zPositionSpin-> setMaximum(this->actor->GetZMax());
		this->ui->zPositionSlide->setMaximum(this->actor->GetZMax());
		this->ui->zPositionSpin-> setValue(z);

		// Call these functions to make sure that the seeds are updated as well
		this->setXSlice(x, false);
		this->setYSlice(y, false);
		this->setZSlice(z, false);

		this->connectControls(true);
	}

	// Check for a transformation matrix
	vtkObject * obj = NULL;
	if (ds->getAttributes()->getAttribute("transformation matrix", obj))
	{
		// Cast the object to a transformation matrix
		vtkMatrix4x4 * transformationMatrix = vtkMatrix4x4::SafeDownCast(obj);

		// Check if this went okay
		if (!transformationMatrix)
		{
			this->core()->out()->logMessage("Not a valid transformation matrix!");
			return;
		}

		// Loop through all three dimensions
		for (int i = 0; i < 3; ++i)
		{
			// Copy the matrix to a new one, and apply it to the current slice actor
			vtkMatrix4x4 * matrixCopy = vtkMatrix4x4::New();
			matrixCopy->DeepCopy(transformationMatrix);
			this->actor->GetSliceActor(i)->SetUserMatrix(matrixCopy);
			matrixCopy->Delete();
		}
	} 
	// Otherwise, set identity matrices
	else 
	{
		// Loop through all three dimensions
		for (int i = 0; i < 3; ++i)
		{
			// Create an identity matrix and apply it to the current slice actor
			vtkMatrix4x4 * id = vtkMatrix4x4::New();
			id->Identity();
			this->actor->GetSliceActor(i)->SetUserMatrix(id);
			id->Delete();
		}
	}

	vtkMedicalCanvas * canvas = this->fullCore()->canvas();

	// Reset the camera of the 3D volume
	this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();

	// Reset the cameras of the 2D view
	this->reset2DCamera(canvas->GetSubCanvas2D(0)->GetRenderer(), this->actor->GetSliceActor(0), 0);
	this->reset2DCamera(canvas->GetSubCanvas2D(1)->GetRenderer(), this->actor->GetSliceActor(1), 1);
	this->reset2DCamera(canvas->GetSubCanvas2D(2)->GetRenderer(), this->actor->GetSliceActor(2), 2);

	// Signal the data manager that the slice actors have been modified
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[0]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[1]);
	this->core()->data()->dataSetChanged(this->sliceActorDataSets[2]);

	this->core()->data()->dataSetChanged(this->seedDataSets[0]);
	this->core()->data()->dataSetChanged(this->seedDataSets[1]);
	this->core()->data()->dataSetChanged(this->seedDataSets[2]);
}


//------------------------[ checkWeightVolumeMatch ]-----------------------\\

bool PlanesVisPlugin::checkWeightVolumeMatch()
{
	// Check if indices are in the correct range
	if (	this->ui->dtiVolumeCombo->currentIndex() < 0 || 
			this->ui->dtiVolumeCombo->currentIndex() >= this->dtiDataSets.size() ||
			this->ui->dtiWeightCombo->currentIndex() < 0 ||
			this->ui->dtiWeightCombo->currentIndex() - 1 >= this->scalarVolumeDataSets.size())
	{
		return false;
	}

	// If the weighting volume is set to "None", we've always got a match
	if (this->ui->dtiWeightCombo->currentIndex() == 0)
		return true;

	// Get the data sets of the DTI eigensystem image and the weighting image
	data::DataSet * dtiDS = this->dtiDataSets.at(this->ui->dtiVolumeCombo->currentIndex());
	data::DataSet * weightDS = this->scalarVolumeDataSets.at(this->ui->dtiWeightCombo->currentIndex() - 1);

	if (dtiDS == NULL || weightDS == NULL)
		return false;

	vtkImageData * dtiImage = dtiDS->getVtkImageData();
	vtkImageData * weightImage = weightDS->getVtkImageData();

	if (dtiImage == NULL || weightImage == NULL)
		return false;

	int dtiDims[3];
	int weightDims[3];

	// Get the dimensions of the two images
	dtiImage->GetDimensions(dtiDims);
	weightImage->GetDimensions(weightDims);

	// Check if the dimensions match
	for (int i = 0; i < 3; ++i)
	{
		if (dtiDims[i] != weightDims[i])
			return false;
	}

	return true;
}


//----------------------------[ reset2DCamera ]----------------------------\\

void PlanesVisPlugin::reset2DCamera(vtkRenderer * renderer, vtkImageSliceActor * sliceActor, int axis)
{
	if (renderer == NULL || sliceActor == NULL || axis < 0 || axis > 2)
		return;

	// Get the image used to create the actor
	vtkImageData * input = sliceActor->GetInput();
	
	if (!input) 
		return;

	// Create a new parallel projection camera
	vtkCamera * newCamera = vtkCamera::New();
	newCamera->ParallelProjectionOn();

	double bounds[6];
	double center[3];
	double normal[3];
	double viewUp[3];

	// Get the bounds of the input, and the center and normal of the plane
	input->GetBounds(bounds);
	sliceActor->GetPlaneCenter(center);
	sliceActor->GetPlaneNormal(normal);

	// Find the minimum and maximum bounds
	double min = bounds[0];
	if (bounds[2] < min)	min = bounds[2];
	if (bounds[4] < min)	min = bounds[4];

	double max = bounds[1];
	if (bounds[3] > max)	max = bounds[3];
	if (bounds[5] > max)	max = bounds[5];

	// The view direction depends on the axis
	switch (axis)
	{
		case 0:
			viewUp[0] = 0.0;
			viewUp[1] = 0.0;
			viewUp[2] = 1.0;
			break;
		case 1:
			viewUp[0] = 0.0;
			viewUp[1] = 0.0;
			viewUp[2] = 1.0;
			break;

		case 2:
			viewUp[0] = 0.0;
			viewUp[1] = 1.0;
			viewUp[2] = 0.0;
			break;
	}

	// Get the user transformation matrix from the plane actor
	vtkMatrix4x4 * m = sliceActor->GetUserMatrix();

	// Check if the matrix exists
	if (m)
	{
		// Transform the center of the plane (including translation)
		double center4[4] = {center[0], center[1], center[2], 1.0};
		m->MultiplyPoint(center4, center4);
		center[0] = center4[0];
		center[1] = center4[1];
		center[2] = center4[2];

		// Transform the plane normal (excluding translation)
		double normal4[4] = {normal[0], normal[1], normal[2], 0.0};
		m->MultiplyPoint(normal4, normal4);
		normal[0] = normal4[0];
		normal[1] = normal4[1];
		normal[2] = normal4[2];

		// Transform the view vector (excluding translation)
		double viewUp4[4] = {viewUp[0], viewUp[1], viewUp[2], 0.0};
		m->MultiplyPoint(viewUp4, viewUp4);
		viewUp[0] = viewUp4[0];
		viewUp[1] = viewUp4[1];
		viewUp[2] = viewUp4[2];

		// Transform the corner points of the images
		double boundsMin[4] = {bounds[0], bounds[2], bounds[4], 0.0};
		double boundsMax[4] = {bounds[1], bounds[3], bounds[5], 0.0};
		m->MultiplyPoint(boundsMin, boundsMin);
		m->MultiplyPoint(boundsMax, boundsMax);

		// Recompute the minimum and maximum of the bounds
		min = boundsMin[0];
		if (boundsMin[1] < min)		min = boundsMin[1];
		if (boundsMin[2] < min)		min = boundsMin[2];

		max = boundsMax[0];
		if (boundsMax[1] > max)		max = boundsMax[1];
		if (boundsMax[2] > max)		max = boundsMax[2];
	}

	// Normalize the normal
	if (vtkMath::Norm(normal) == 0.0)
	{
		this->core()->out()->showMessage("Plane normal has zero length!");
		return;
	}

	vtkMath::Normalize(normal);

	// Normalize the view vector
	if (vtkMath::Norm(viewUp) == 0.0)
	{
		this->core()->out()->showMessage("Plane 'viewUp' vector has zero length!");
		return;
	}

	vtkMath::Normalize(viewUp);

	// Set the position of the camera. We start at the center of the plane, and move 
	// along its normal to ensure head-on projection for rotated planes. The distance
	// moved along the normal is equal to the image size along the selected axis, to
	// ensure that the camera is placed outside of the volume.

	newCamera->SetPosition(	center[0] - (bounds[1] - bounds[0]) * normal[0], 
							center[1] - (bounds[3] - bounds[2]) * normal[1], 
							center[2] - (bounds[5] - bounds[4]) * normal[2]);

	// Set the view vector for the camera
	newCamera->SetViewUp(viewUp);

	// Set the center of the plane as the camera's focal point
	newCamera->SetFocalPoint(center);
	newCamera->SetParallelScale((max - min) / 2);

	// Done, set the new camera
	renderer->SetActiveCamera(newCamera);
	newCamera->Delete();
}


//------------------------------[ setXSlice ]------------------------------\\

void PlanesVisPlugin::setXSlice(int x, bool updateData)
{
	// Set the slice position
	this->actor->SetX(x);

	// Get the input image of the slice
	vtkImageData * input = this->actor->GetInput();
		
	if (input)
	{
		// Get the bounds and spacing of the input image
		double bounds[6]; 
		input->GetBounds(bounds);

		double spacing[3]; 
		input->GetSpacing(spacing);

		// Limit the bounds of the X-dimensions to the slice location
		bounds[0] = bounds[1] = this->actor->GetSliceActor(0)->GetSliceLocation();

		// Compute the seed points in this plane
		this->updateSeeds(this->seedsX, bounds, spacing);

		// Tell the data manager that the seed points have changed
		if (updateData)
			this->core()->data()->dataSetChanged(this->seedDataSets[0]);
	}

	if (updateData)
	{
		// Tell the data manager that the slice actor has been modified
		this->core()->data()->dataSetChanged(this->sliceActorDataSets[0]);
		this->core()->render();
	}
}


//------------------------------[ setYSlice ]------------------------------\\

void PlanesVisPlugin::setYSlice(int y, bool updateData)
{
	// Like "setXSlice"
	this->actor->SetY(y);
	vtkImageData * input = this->actor->GetInput();
		
	if (input)
	{
		double bounds[6]; 
		input->GetBounds(bounds);
		
		double spacing[3]; 
		input->GetSpacing(spacing);
		
		bounds[2] = bounds[3] = this->actor->GetSliceActor(1)->GetSliceLocation();
		this->updateSeeds(this->seedsY, bounds, spacing);

		if (updateData)
			this->core()->data()->dataSetChanged(this->seedDataSets[1]);
	} 

	if (updateData)
	{
		this->core()->data()->dataSetChanged(this->sliceActorDataSets[1]);
		this->core()->render();
	}
}


//------------------------------[ setZSlice ]------------------------------\\

void PlanesVisPlugin::setZSlice(int z, bool updateData)
{
	// Like "setXSlice"
	this->actor->SetZ(z);
	vtkImageData * input = this->actor->GetInput();

	if (input)
	{
		double bounds[6]; 
		input->GetBounds(bounds);

		double spacing[3]; 
		input->GetSpacing(spacing);
		
		bounds[4] = bounds[5] = this->actor->GetSliceActor(2)->GetSliceLocation();
		this->updateSeeds(this->seedsZ, bounds, spacing);
	
		if (updateData)
			this->core()->data()->dataSetChanged(this->seedDataSets[2]);
	}

	if (updateData)
	{
		this->core()->data()->dataSetChanged(this->sliceActorDataSets[2]);
		this->core()->render();
	}
}


//-----------------------------[ setXVisible ]-----------------------------\\

void PlanesVisPlugin::setXVisible(bool v)
{
	// Hide or show the slice, and render the scene
	this->actor->SetSliceVisible(0, v);
	this->core()->render();
}


//-----------------------------[ setYVisible ]-----------------------------\\

void PlanesVisPlugin::setYVisible(bool v)
{
	this->actor->SetSliceVisible(1, v);
	this->core()->render();
}


//-----------------------------[ setZVisible ]-----------------------------\\

void PlanesVisPlugin::setZVisible(bool v)
{
	this->actor->SetSliceVisible(2, v);
	this->core()->render();
}


//---------------------------[ setInterpolation ]--------------------------\\

void PlanesVisPlugin::setInterpolation(bool i)
{
	// Turn interpolation on or off, and render the scene
	this->actor->SetInterpolate(i ? 1 : 0);
	this->core()->render();
}


//-----------------------------[ updateSeeds ]-----------------------------\\

void PlanesVisPlugin::updateSeeds(vtkPoints * points, double bounds[6], double steps[3])
{
	// Check the bounds
	for (int i = 0; i < 3; ++i)
	{
		Q_ASSERT(bounds[2 * i] <= bounds[2 * i + 1]);
		Q_ASSERT(steps[i] > 0.0);
	}
	// Reset the point set
	points->Reset();

	double x; 
	double y; 
	double z;

	x = bounds[0];

	// Add a seed point on every voxel
	while (x <= bounds[1])
	{
		y = bounds[2];
		
		while (y <= bounds[3])
		{
			z = bounds[4];
			
			while (z <= bounds[5])
			{
				points->InsertNextPoint(x, y, z);
				z += steps[2];
			}

			y += steps[1];
		}

		x += steps[0];
	}
}


//---------------------------[ applyLUTColoring ]--------------------------\\

void PlanesVisPlugin::applyLUTColoring()
{
	// First set the scalar volume, and then set the LUT
	this->changeScalarVolume(this->ui->scalarVolumeCombo->currentIndex());
	this->changeLUT(this->ui->lutCombo->currentIndex());

	// Disable/enable GUI widgets
	this->ui->scalarVolumeLabel->setEnabled(true);
	this->ui->scalarVolumeCombo->setEnabled(true);
	this->ui->lutLabel->setEnabled(true);
	this->ui->lutCombo->setEnabled(true);

	this->ui->dtiVolumeLabel->setEnabled(false);
	this->ui->dtiVolumeCombo->setEnabled(false);
	this->ui->dtiWeightLabel->setEnabled(false);
	this->ui->dtiWeightCombo->setEnabled(false);
}

	
//---------------------------[ applyRGBColoring ]--------------------------\\

void PlanesVisPlugin::applyRGBColoring()
{
	// First set the DTI volume, and then the weighting volume
	this->changeDTIVolume(this->ui->dtiVolumeCombo->currentIndex());
	this->changeWeightVolume(this->ui->dtiWeightCombo->currentIndex());

	this->ui->scalarVolumeLabel->setEnabled(false);
	this->ui->scalarVolumeCombo->setEnabled(false);
	this->ui->lutLabel->setEnabled(false);
	this->ui->lutCombo->setEnabled(false);

	this->ui->dtiVolumeLabel->setEnabled(true);
	this->ui->dtiVolumeCombo->setEnabled(true);
	this->ui->dtiWeightLabel->setEnabled(true);
	this->ui->dtiWeightCombo->setEnabled(true);
}


//-----------------------------[ getSubcanvas ]----------------------------\\

vtkSubCanvas * PlanesVisPlugin::getSubcanvas(int i)
{
	// Return the required 2D subcanvas
	return this->fullCore()->canvas()->GetSubCanvas2D(i);
}


//-------------------------[ resetSubCanvasCamera ]------------------------\\

void PlanesVisPlugin::resetSubCanvasCamera(int i)
{
	// Get the renderer of the selected subcanvas
	vtkRenderer * ren = this->fullCore()->canvas()->GetSubCanvas2D(i)->GetRenderer();

	// Reset the camera for this renderer
	this->reset2DCamera(ren, this->actor->GetSliceActor(i), i);

	// Redraw the screen
	this->fullCore()->render();
}


//-------------------------------[ Execute ]-------------------------------\\

void PlanesVisPluginCallback::Execute(vtkObject * caller, unsigned long event, void * callData)
{
	// Handle the event of selecting a new subcanvas
	if (event == vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVAS_CAMERA_RESET)
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

		// Reset the camera for the selected subcanvas
		if (selected2DCanvas != -1)
		{
			this->plugin->resetSubCanvasCamera(selected2DCanvas);

			// Turn on the abort flag to show that we've handled this event
			this->AbortFlagOn();
		}
	}
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libPlanesVisPlugin, bmia::PlanesVisPlugin)
