
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

/* Crop3DPlugin.cxx
*
* 2013-02-10	Mehmet Yusufoglu
* - Version 1.0.0.
* - First version
*
* 2013-06-30	Mehmet Yusufoglu
* Remove this->fullCore()->canvas()->GetRenderer3D()->ResetCamera(); line to prevent resetting of 
* the camera at each PlanesVisualisation plane position slider movement. Slider triggers datachanged,
* and the line above resets the camera, which is not wanted.
*/

/** Includes */

#include "Crop3DPlugin.h"


namespace bmia {


	//-----------------------------[ Constructor ]-----------------------------\\

	Crop3DPlugin::Crop3DPlugin() : AdvancedPlugin("Crop3D")
	{
		// Create a planes actor, and hide it for now
		this->actor = vtkImageOrthogonalSlicesActor::New();
		this->actor->VisibilityOff();

		// Setup the GUI
		this->qWidget = new QWidget();
		this->ui = new Ui::Crop3DForm();
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

		// be sure roi for croppong is not displayed.
		this->ui->roiBoxVisibleCheckBox->setChecked(false);

		// Turn actor interpolation on or off, depending on the default GUI settings
		this->actor->SetInterpolate(this->ui->interpolationCheck->isChecked() ? 1 : 0);

		// No callback yet, we create it in the "init" function
		this->callBack = NULL;

		// Connect the controls
		this->connectControls(true);
	}


	//---------------------------------[ init ]--------------------------------\\

	void Crop3DPlugin::init()
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
		this->seedDataSets[0] = new data::DataSet("CropPlane X", "seed points", pointSetX);
		this->seedDataSets[1] = new data::DataSet("CropPlane Y", "seed points", pointSetY);
		this->seedDataSets[2] = new data::DataSet("CropPlane Z", "seed points", pointSetZ);

		// Reduce reference count of the point sets to one
		pointSetX->Delete();
		pointSetY->Delete();
		pointSetZ->Delete();

		// Get the canvas
		vtkMedicalCanvas * canvas = this->fullCore()->canvas();

		QString sliceActorNames[3];
		sliceActorNames[0] = "Crop X Plane";
		sliceActorNames[1] = "Crop Y Plane";
		sliceActorNames[2] = "Crop Z Plane";

		// Loop through all three axes
		for(int axis = 0; axis < 3; ++axis)
		{
			// Add the seed point data set to the data manager
			// the seeds can be added just after the crop, otherwise they wil be same with the seed of planes plugin.
			//this->core()->data()->addDataSet(this->seedDataSets[axis]);

			// Add the planes to their respective 2D views
			vtkImageSliceActor * sliceActor = this->actor->GetSliceActor(axis);
			//remove 2d planes for cropped volume so that they will not be displayed on top of planes plugins actors.
			//canvas->GetSubCanvas2D(axis)->GetRenderer()->AddActor(sliceActor);

			// Add the slice actors to the data manager
			this->sliceActorDataSets[axis] = new data::DataSet(sliceActorNames[axis], "sliceActor", sliceActor);
			this->core()->data()->addDataSet(this->sliceActorDataSets[axis]);
		}

		// Create the callback class
		this->callBack = Crop3DPluginCallback::New();
		this->callBack->plugin = this;

		// Add the callback to the canvas as an observer
		this->fullCore()->canvas()->AddObserver(vtkCommand::UserEvent + 
			BMIA_USER_EVENT_SUBCANVAS_CAMERA_RESET, this->callBack);

		this->setZVisible(0);
		this->setXVisible(0);
		this->setYVisible(0);

		//3D ROI box as a polydata
		boxPts =vtkPoints::New();
		polyDataBox =  vtkPolyData::New();

		// 8 points as vertices of 3D box
		boxPts->SetNumberOfPoints(8);
		// mapper for 3d roi 
		mapperPolyDataBox =  vtkPolyDataMapper::New();
		

		// Add the points to the roi polydata 
		this->polyDataBox->Allocate();
		polyDataBox->SetPoints(boxPts);
		vtkIdType connectivity[2];

		//create 3d roi box as a rectangular prism
		// 12 Edges of 3d roi box
		for(int i=0; i<7;i++)
		{
			connectivity[0] = i;
			connectivity[1] = i+1;
			this->polyDataBox->InsertNextCell(VTK_LINE,2,connectivity);
		} 
		connectivity[0] = 7;
		connectivity[1] = 0;
		this->polyDataBox->InsertNextCell(VTK_LINE,2,connectivity);
		connectivity[0] = 0;
		connectivity[1] = 3;
		this->polyDataBox->InsertNextCell(VTK_LINE,2,connectivity);
		connectivity[0] = 2;
		connectivity[1] = 5;
		this->polyDataBox->InsertNextCell(VTK_LINE,2,connectivity);
		connectivity[0] = 4;
		connectivity[1] = 7;
		this->polyDataBox->InsertNextCell(VTK_LINE,2,connectivity);
		connectivity[0] = 1;
		connectivity[1] = 6;
		this->polyDataBox->InsertNextCell(VTK_LINE,2,connectivity);

		// display pipeline
		actorPolyDataBox =  vtkActor::New();
		actorPolyDataBox->SetVisibility(0);
#if VTK_MAJOR_VERSION <= 5
		mapperPolyDataBox->SetInput(polyDataBox);
#else
		mapperPolyDataBox->SetInputData(polyDataBox);
#endif

		actorPolyDataBox->SetMapper(mapperPolyDataBox);

		//add to the actor as a part so that it can be controlled by the visibility button in the main window
		this->actor->AddPart(actorPolyDataBox);


	}


	//------------------------------[ Destructor ]-----------------------------\\

	Crop3DPlugin::~Crop3DPlugin()
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

		boxPts->Delete()  ;
		polyDataBox->Delete() ;
		mapperPolyDataBox->Delete() ;
		actorPolyDataBox->Delete() ;

		// Delete the callback
		if (this->callBack)
			this->callBack->Delete();
	}


	//---------------------------[ connectControls ]---------------------------\\

	void Crop3DPlugin::connectControls(bool doConnect)
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

			// 3D Crop ROI signals
			connect(this->ui->cropButton,		SIGNAL(clicked()),			this , SLOT( cropData() ),	Qt::UniqueConnection		);
			connect(this->ui->roiBoxVisibleCheckBox,	SIGNAL(toggled(bool)),				this, SLOT(setRoiBoxVisible(bool))	);

			// spin -> slider connection 
			connect(this->ui->x0ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderX0 , SLOT(setValue(int) )			);
			connect(this->ui->y0ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderY0 , SLOT(setValue(int) )			);
			connect(this->ui->z0ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderZ0 , SLOT(setValue(int) )			);
			connect(this->ui->x1ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderX1 , SLOT(setValue(int) )			);
			connect(this->ui->y1ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderY1 , SLOT(setValue(int) )			);
			connect(this->ui->z1ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderZ1 , SLOT(setValue(int) )			);
			//slider -> spin connections
			connect(this->ui->horizontalSliderX0,		SIGNAL(valueChanged(int)),			this->ui->x0ROIPositionSpin, SLOT(setValue(int) )			);
			connect(this->ui->horizontalSliderY0,		SIGNAL(valueChanged(int)),			this->ui->y0ROIPositionSpin, SLOT(setValue(int) )			);
			connect(this->ui->horizontalSliderZ0,		SIGNAL(valueChanged(int)),			this->ui->z0ROIPositionSpin, SLOT(setValue(int) )			);
			connect(this->ui->horizontalSliderX1,		SIGNAL(valueChanged(int)),			this->ui->x1ROIPositionSpin, SLOT(setValue(int) )			);
			connect(this->ui->horizontalSliderY1,		SIGNAL(valueChanged(int)),			this->ui->y1ROIPositionSpin, SLOT(setValue(int) )			);
			connect(this->ui->horizontalSliderZ1,		SIGNAL(valueChanged(int)),			this->ui->z1ROIPositionSpin, SLOT(setValue(int) )			);

			// spin to 3d roi box connections !!!
			connect(this->ui->x0ROIPositionSpin,		SIGNAL(valueChanged(int)),  this, SLOT(changeRoiBoundary(int) ));
			connect(this->ui->x1ROIPositionSpin,		SIGNAL(valueChanged(int)),  this, SLOT(changeRoiBoundary(int) ));
			connect(this->ui->y0ROIPositionSpin,		SIGNAL(valueChanged(int)),  this, SLOT(changeRoiBoundary(int) ));
			connect(this->ui->y1ROIPositionSpin,		SIGNAL(valueChanged(int)),  this, SLOT(changeRoiBoundary(int) ));
			connect(this->ui->z0ROIPositionSpin,		SIGNAL(valueChanged(int)),  this, SLOT(changeRoiBoundary(int) ));
			connect(this->ui->z1ROIPositionSpin,		SIGNAL(valueChanged(int)),  this, SLOT(changeRoiBoundary(int) ));

			connect(this->ui->xPositionSlide,SIGNAL(valueChanged(int)), this->ui->xPositionSpin, SLOT(setValue(int)));
			connect(this->ui->yPositionSlide,SIGNAL(valueChanged(int)), this->ui->yPositionSpin, SLOT(setValue(int)));
			connect(this->ui->zPositionSlide,SIGNAL(valueChanged(int)), this->ui->zPositionSpin, SLOT(setValue(int)));
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

			//ROI
			disconnect(this->ui->cropButton,		SIGNAL(clicked()),			this , SLOT( cropData() )			);
			//3D Crop ROI spin -> slider connection 
			disconnect(this->ui->x0ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderX0 , SLOT(setValue(int) )		);
			disconnect(this->ui->y0ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderY0 , SLOT(setValue(int) )		);
			disconnect(this->ui->z0ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderZ0 , SLOT(setValue(int) )		);
			disconnect(this->ui->x1ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderX1 , SLOT(setValue(int) )		);
			disconnect(this->ui->y1ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderY1 , SLOT(setValue(int) )		);
			disconnect(this->ui->z1ROIPositionSpin,		SIGNAL(valueChanged(int)),			this->ui->horizontalSliderZ1 , SLOT(setValue(int) )		);
			//3D Crpo ROI   slider->spin conection 
			disconnect(this->ui->horizontalSliderX0,		SIGNAL(valueChanged(int)),			this->ui->x0ROIPositionSpin, SLOT(setValue(int) )	);
			disconnect(this->ui->horizontalSliderY0,		SIGNAL(valueChanged(int)),			this->ui->y0ROIPositionSpin, SLOT(setValue(int) )	);
			disconnect(this->ui->horizontalSliderZ0,		SIGNAL(valueChanged(int)),			this->ui->z0ROIPositionSpin, SLOT(setValue(int) )	);
			disconnect(this->ui->horizontalSliderX1,		SIGNAL(valueChanged(int)),			this->ui->x1ROIPositionSpin, SLOT(setValue(int) )	);
			disconnect(this->ui->horizontalSliderY1,		SIGNAL(valueChanged(int)),			this->ui->y1ROIPositionSpin, SLOT(setValue(int) )	);
			disconnect(this->ui->horizontalSliderZ1,		SIGNAL(valueChanged(int)),			this->ui->z1ROIPositionSpin, SLOT(setValue(int) )	);

			disconnect(this->ui->xPositionSlide,SIGNAL(valueChanged(int)), this->ui->xPositionSpin, SLOT(setValue(int)));
			disconnect(this->ui->yPositionSlide,SIGNAL(valueChanged(int)), this->ui->yPositionSpin, SLOT(setValue(int)));
			disconnect(this->ui->zPositionSlide,SIGNAL(valueChanged(int)), this->ui->zPositionSpin, SLOT(setValue(int)));

		}
	}


	//------------------------------[ getVtkProp ]-----------------------------\\

	vtkProp * Crop3DPlugin::getVtkProp()
	{
		return this->actor;
	}


	//--------------------------------[ getGUI ]-------------------------------\\

	QWidget * Crop3DPlugin::getGUI()
	{
		return this->qWidget;
	}


	//----------------------------[ dataSetAdded ]-----------------------------\\

	void Crop3DPlugin::dataSetAdded(data::DataSet * ds)
	{
		//cout << "Crop3DPlugin dataSetAdded " << ds->getName().toStdString() << " " << ds->getName().toStdString() << endl;

		int isCropped(0);
		ds->getAttributes()->getAttribute("isSubVolume",isCropped);


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


		//else if (ds->getKind() == "DTI")
		//{
		//	 
		//	this->connectControls(false);

		//	// Add the eigensystem (DTI) data set to the list
		//	this->dtiDataSets.append(ds);

		//	// Enable the radio button for RGB coloring
		//	this->ui->dtiRadio->setEnabled(true);

		//	// Add the data set to the combo box
		//	this->ui->dtiVolumeCombo->addItem(ds->getName());

		//	// If this is the first DTI set, we switch to RGB coloring
		//	if (this->ui->dtiVolumeCombo->count() == 1)
		//	{
		//		this->ui->dtiRadio->setChecked(true);
		//		this->applyRGBColoring();

		//		// Reset the camera of the 3D volume
		//		this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();
		//	}

		//	this->connectControls(true);
		//}

		else if (ds->getKind() == "eigen") // &&  isCropped)
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

		//else if (ds->getKind() == "DTI")
		//{
		//	 
		//	this->connectControls(false);

		//	// Add the eigensystem (DTI) data set to the list
		//	this->dtiDataSets.append(ds);

		//	// Enable the radio button for RGB coloring
		//	this->ui->dtiRadio->setEnabled(true);

		//	// Add the data set to the combo box
		//	this->ui->dtiVolumeCombo->addItem(ds->getName());

		//	// If this is the first DTI set, we switch to RGB coloring
		//	if (this->ui->dtiVolumeCombo->count() == 1)
		//	{
		//		this->ui->dtiRadio->setChecked(true);
		//		this->applyRGBColoring();

		//		// Reset the camera of the 3D volume
		//		this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();
		//	}

		//	this->connectControls(true);
		//}
	}


	//---------------------------[ dataSetChanged ]----------------------------\\

	void Crop3DPlugin::dataSetChanged(data::DataSet * ds)
	{
		// General behavior: For each of the input data types, it updates the name in
		// the GUI combo boxes, and if the data set is selected in one of these combo
		// boxes, it also calls the corresponding update function ("changeX").
		//cout << "dataSetChanged" << ds->getName().toStdString() << endl;
		// Scalar Volumes
		//cin.get();
		int isCropped(0);
		ds->getAttributes()->getAttribute("isSubVolume",isCropped);

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

	void Crop3DPlugin::dataSetRemoved(data::DataSet * ds)
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


	//
	void Crop3DPlugin::setRoiBoxVisible(bool v)
	{
		if(v) 	
			this->changeRoiBoundary(0);// argument not used

		// Take form the slice sliders 
		this->actorPolyDataBox->SetVisibility(v);
		this->core()->render();  

	}


	////
	void Crop3DPlugin::changeRoiBoundary( int notUsed )
	{
		int *bndSlices = new int[6];
		get3DROIBoundaries(bndSlices);

		data::DataSet * ds;
		if ((this->ui->scalarVolumeRadio->isChecked()))

			ds = this->scalarVolumeDataSets.at( this->ui->scalarVolumeCombo->currentIndex());
		else if(this->ui->dtiRadio->isChecked())
		{
			ds = this->dtiDataSets.at(this->ui->dtiVolumeCombo->currentIndex());
		}
		else
		{
			qDebug() << "Neither of the radio buttons is checked!"<< endl;
			return;
		}

		if (!ds)
			return;

		if (!(ds->getVtkImageData()))
			return;

		// Check for a transformation matrix
		vtkObject * obj = NULL;
		vtkMatrix4x4 * transformationMatrix=vtkMatrix4x4::New();
		//There may be no transformation attribute
		if (ds->getAttributes()->getAttribute("transformation matrix", obj))
		{

			// Cast the object to a transformation matrix
			transformationMatrix = vtkMatrix4x4::SafeDownCast(obj);

			// Check if this went okay
			if (!transformationMatrix)
			{
				this->core()->out()->logMessage("Not a valid transformation matrix!");
				//return;
			}
		}

		//Alternative///////////////////
		double xMin, xMax, yMin, yMax, zMin, zMax;
		xMin = bndSlices[0]; xMax = bndSlices[1];
		yMin = bndSlices[2]; yMax = bndSlices[3];
		zMin = bndSlices[4]; zMax = bndSlices[5];

		boxPts->SetPoint(0, xMax, yMin, zMax);
		boxPts->SetPoint(1, xMax, yMin, zMin);
		boxPts->SetPoint(2, xMax, yMax, zMin);
		boxPts->SetPoint(3, xMax, yMax, zMax);
		boxPts->SetPoint(7, xMin, yMin, zMax);
		boxPts->SetPoint(6, xMin, yMin, zMin);
		boxPts->SetPoint(5, xMin, yMax, zMin);
		boxPts->SetPoint(4, xMin, yMax, zMax);


		this->polyDataBox->Update();


		this->actorPolyDataBox->SetUserMatrix(transformationMatrix);// this can be done in the selection of radio buttons.
		this->mapperPolyDataBox->SetImmediateModeRendering(1);
		// Setup actor and mapper

		this->core()->render();

	}


	//--------------------------[ changeScalarVolume ]-------------------------\\

	void Crop3DPlugin::changeScalarVolume(int index)
	{


		// Only do this is we're currently scowing a scalar volume
		if (!(this->ui->scalarVolumeRadio->isChecked()))
			return;

		// If the index is out of range, simply hide the actor
		if (index < 0 || index >= this->scalarVolumeDataSets.size())
		{
			qDebug() << "index is out of range";
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

		//this->actor->SetInput(NULL); // test 1
		//this->actor->UpdateInput(); // test 1.5
		//this->core()->render(); // tes 2
		// Use the image as the input for the actor
		this->actor->SetInput(ds->getVtkImageData());

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

	void Crop3DPlugin::changeLUT(int index)
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

	void Crop3DPlugin::changeDTIVolume(int index)
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

	void Crop3DPlugin::changeWeightVolume(int index)
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
			QMessageBox::warning(this->getGUI(), "Crop3D Visualisation", "Dimension mismatch between DTI volume and weighting volume!");
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

	void Crop3DPlugin::configureNewImage(data::DataSet * ds)
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

			// 3D Crop ROI Part
			this->set3DROISliderLimits();

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

		//roi box
		//this->boxRep->PlaceWidget(this->actor->GetBounds()); // if planes are not visible does not work
		this->changeRoiBoundary(0); // argument not used

	}


	//------------------------[ checkWeightVolumeMatch ]-----------------------\\

	bool Crop3DPlugin::checkWeightVolumeMatch()
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

	void Crop3DPlugin::reset2DCamera(vtkRenderer * renderer, vtkImageSliceActor * sliceActor, int axis)
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

	void Crop3DPlugin::setXSlice(int x, bool updateData)
	{
		// Set the slice position
		//cout << "set slice x" << endl;
		this->actor->SetX(x);
		//stephens plugin
		//data::DataSet * ds = this->scalarVolumeDataSets.at(this->ui->scalarVolumeCombo->currentIndex());
		//if(ds->getAttributes()->hasAttribute("SlicePosX"))
		//{
		//	ds->getAttributes()->addAttribute("SlicePosX",x);
		//   cout << "Added attribute \n";
		//	this->core()->data()->dataSetChanged(ds);
		//}
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

	void Crop3DPlugin::setYSlice(int y, bool updateData)
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

	void Crop3DPlugin::setZSlice(int z, bool updateData)
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

	void Crop3DPlugin::setXVisible(bool v)
	{
		// Hide or show the slice, and render the scene
		this->actor->SetSliceVisible(0, v);
		this->core()->render();
	}


	//-----------------------------[ setYVisible ]-----------------------------\\

	void Crop3DPlugin::setYVisible(bool v)
	{
		this->actor->SetSliceVisible(1, v);
		this->core()->render();
	}


	//-----------------------------[ setZVisible ]-----------------------------\\

	void Crop3DPlugin::setZVisible(bool v)
	{
		this->actor->SetSliceVisible(2, v);
		this->core()->render();
	}


	//---------------------------[ setInterpolation ]--------------------------\\

	void Crop3DPlugin::setInterpolation(bool i)
	{
		// Turn interpolation on or off, and render the scene
		this->actor->SetInterpolate(i ? 1 : 0);
		this->core()->render();
	}


	//-----------------------------[ updateSeeds ]-----------------------------\\

	void Crop3DPlugin::updateSeeds(vtkPoints * points, double bounds[6], double steps[3])
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

	void Crop3DPlugin::applyLUTColoring()
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

	void Crop3DPlugin::applyRGBColoring()
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

	vtkSubCanvas * Crop3DPlugin::getSubcanvas(int i)
	{
		// Return the required 2D subcanvas
		return this->fullCore()->canvas()->GetSubCanvas2D(i);
	}


	//-------------------------[ resetSubCanvasCamera ]------------------------\\

	void Crop3DPlugin::resetSubCanvasCamera(int i)
	{
		// Get the renderer of the selected subcanvas
		vtkRenderer * ren = this->fullCore()->canvas()->GetSubCanvas2D(i)->GetRenderer();

		// Reset the camera for this renderer
		this->reset2DCamera(ren, this->actor->GetSliceActor(i), i);

		// Redraw the screen
		this->fullCore()->render();
	}


	//-------------------------------[ Execute ]-------------------------------\\

	void Crop3DPluginCallback::Execute(vtkObject * caller, unsigned long event, void * callData)
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

	// Set Slider limits to data dimensions
	void Crop3DPlugin::set3DROISliderLimits()
	{
		//slider limits
		this->ui->horizontalSliderX0->setMinimum(this->actor->GetXMin());
		this->ui->horizontalSliderX0->setMaximum(this->actor->GetXMax());
		this->ui->horizontalSliderY0->setMinimum(this->actor->GetYMin());
		this->ui->horizontalSliderY0->setMaximum(this->actor->GetYMax());
		this->ui->horizontalSliderZ0->setMinimum(this->actor->GetZMin());
		this->ui->horizontalSliderZ0->setMaximum(this->actor->GetZMax());

		this->ui->horizontalSliderX1->setMinimum(this->actor->GetXMin());
		this->ui->horizontalSliderX1->setMaximum(this->actor->GetXMax());
		this->ui->horizontalSliderY1->setMinimum(this->actor->GetYMin());
		this->ui->horizontalSliderY1->setMaximum(this->actor->GetYMax());
		this->ui->horizontalSliderZ1->setMinimum(this->actor->GetZMin());
		this->ui->horizontalSliderZ1->setMaximum(this->actor->GetZMax());


		//slider set value
		this->ui->horizontalSliderX0->setValue(this->actor->GetXMin());
		this->ui->horizontalSliderY0->setValue(this->actor->GetYMin());
		this->ui->horizontalSliderZ0->setValue(this->actor->GetZMin());


		this->ui->horizontalSliderX1->setValue(this->actor->GetXMax());
		this->ui->horizontalSliderY1->setValue(this->actor->GetYMax());
		this->ui->horizontalSliderZ1->setValue(this->actor->GetZMax());


		// spins
		this->ui->x0ROIPositionSpin->setMinimum(this->actor->GetXMin());
		this->ui->x0ROIPositionSpin->setMaximum(this->actor->GetXMax());

		this->ui->x1ROIPositionSpin->setMinimum(this->actor->GetXMin());
		this->ui->x1ROIPositionSpin->setMaximum(this->actor->GetXMax());

		this->ui->y0ROIPositionSpin->setMinimum(this->actor->GetYMin());
		this->ui->y0ROIPositionSpin->setMaximum(this->actor->GetYMax());

		this->ui->y1ROIPositionSpin->setMinimum(this->actor->GetYMin());
		this->ui->y1ROIPositionSpin->setMaximum(this->actor->GetYMax());

		this->ui->z0ROIPositionSpin->setMinimum(this->actor->GetZMin());
		this->ui->z0ROIPositionSpin->setMaximum(this->actor->GetZMax());

		this->ui->z1ROIPositionSpin->setMinimum(this->actor->GetZMin());
		this->ui->z1ROIPositionSpin->setMaximum(this->actor->GetZMax());

		// set spins
		this->ui->x0ROIPositionSpin->setValue(this->actor->GetXMin());
		this->ui->x1ROIPositionSpin->setValue(this->actor->GetXMax());

		this->ui->y0ROIPositionSpin->setValue(this->actor->GetYMin());	 
		this->ui->y1ROIPositionSpin->setValue(this->actor->GetYMax());

		this->ui->z0ROIPositionSpin->setValue(this->actor->GetZMin());
		this->ui->z1ROIPositionSpin->setValue(this->actor->GetZMax());

	}

	// Get ROI Boundaries Set by the user
	void Crop3DPlugin::get3DROIBoundaries(int *bnd)
	{ 

		bnd[0] = this->ui->horizontalSliderX0->value(); // * spacing
		bnd[1] = this->ui->horizontalSliderX1->value(); 
		bnd[2] = this->ui->horizontalSliderY0->value();
		bnd[3] = this->ui->horizontalSliderY1->value();
		bnd[4] = this->ui->horizontalSliderZ0->value();
		bnd[5] = this->ui->horizontalSliderZ1->value();
	}

	void Crop3DPlugin::cropData()
	{ 

		if( (this->ui->horizontalSliderX0->value() >  this->ui->horizontalSliderX1->value()) ||
			(this->ui->horizontalSliderY0->value() >  this->ui->horizontalSliderY1->value()) || (this->ui->horizontalSliderZ0->value() >  this->ui->horizontalSliderZ1->value()))
		{
			this->core()->out()->showMessage("Initial border value must be less then the second border value along an axis.", "Boundary Problem");
			return ;  
		}      

		for (int i=0; i< this->scalarVolumeDataSets.size();i++)
			qDebug() << i << " " << this->scalarVolumeDataSets.at(i)->getName() << " " << this->scalarVolumeDataSets.at(i)->getKind() << endl;

		for ( int i=0; i< this->dtiDataSets.size();i++)
			qDebug() << i << " " << this->dtiDataSets.at(i)->getName() << " " << this->dtiDataSets.at(i)->getKind() << endl;


		data::DataSet * dataDS;

		if (this->ui->dtiRadio->isChecked())
			dataDS = this->core()->data()->getDataSet(this->ui->dtiVolumeCombo->currentText(),"DTI");  
		else 
			dataDS = this->scalarVolumeDataSets.at(this->ui->scalarVolumeCombo->currentIndex());

		if(dataDS == NULL)
			qDebug() << "Dataset dataDS == NULL" << endl;

		this->crop3DDataSet(dataDS);

		for(int axis = 0; axis < 3; ++axis)
		{
			// Add the seed point data set to the data manager
			// The line below was at the initialization function, but that creates confision. If there is no volume crapped there also exists a crop seed plane which is same with original uncur plane.
			this->core()->data()->addDataSet(this->seedDataSets[axis]);

		}

	}

	void Crop3DPlugin::crop3DDataSet(data::DataSet * ds)
	{
		// qDebug() << "crop3DDataSet "<< ds->getKind() << ds->getName() << endl;
		vtkImageData * image ; 
		// Scalar volume
		if ((ds->getKind() == "scalar volume" ) || (ds->getKind() == "eigen") || (ds->getKind() == "DTI") )  
		{
			image   = ds->getVtkImageData();
			if(!image) 	
			{
				qDebug() << "Not imagedata " << endl; 
				return; }
		}
		else 
		{
			qDebug() << "Neither of the datasets can be cropped." << endl; 
			return;
		}


		vtkExtractVOI *extractVOI = vtkExtractVOI::New();
		extractVOI->SetInputConnection(image->GetProducerPort());
		int bnd[6];
		this->get3DROIBoundaries(bnd);
		extractVOI->SetVOI(bnd[0],bnd[1],bnd[2],bnd[3],bnd[4],bnd[5]);		
		extractVOI->Update();
		vtkImageData* extracted = extractVOI->GetOutput();

		extracted->Update();	 
		vtkObject *obj = vtkObject::SafeDownCast(extracted);
		QString croppedDataName= "Cropped-" + ds->getName();

		if (obj)
		{

			data::DataSet *croppedDS = new data::DataSet( croppedDataName, ds->getKind(),obj);


			vtkObject * objMatrix;
			if ((ds->getAttributes()->getAttribute("transformation matrix", objMatrix)))

			{

				//Add the transformation matrix to the dataset
				croppedDS->getAttributes()->addAttribute("transformation matrix", objMatrix);

			}
			int isCropped(1);
			croppedDS->getAttributes()->addAttribute("isSubVolume", isCropped);

			this->core()->data()->addDataSet(croppedDS); // to only this plugin or to all ?


			this->core()->render();  
		}
		else
		{
			qDebug() << "casting problem \n";
		}

	}

} // namespace bmia


Q_EXPORT_PLUGIN2(libCrop3DPlugin, bmia::Crop3DPlugin)