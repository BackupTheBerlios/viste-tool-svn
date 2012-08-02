/*
 * FiberVisualizationPlugin.cxx
 *
 * 2010-07-15	Tim Peeters
 * - First version
 * 
 * 2010-09-15	Evert van Aart
 * - Implemented "dataSetRemoved".
 *
 * 2010-10-05	Evert van Aart
 * - Added support for different fiber shapes and coloring methods.
 * - An object of type "FiberVisualizationPipeline" is created for each
 *   fiber set. This object contains all visualization settings and filters
 *   for that fiber set.
 * - Implemented "dataSetChanged".
 * 
 * 2010-10-22	Evert van Aart
 * - Added support for coloring using the "CellData" array.
 *
 * 2010-10-25	Evert van Aart
 * - Merged coloring option "None" into option "Single Color". When "Single
 *   Color" is selected, user can use buttons to color the fibers white, use
 *   an automatically generated color (using "vtkQtColorChart"), or select
 *   a custom color. For now, automatic coloring is the default for new fibers.
 * 
 * 2011-01-12	Evert van Aart
 * - Added support for Look-Up Tables (transfer functions).
 * - Added a combo box for the eigensystem images.
 * - Before switching to a new coloring method or fiber shape, the class now checks
 *   whether the required data is available (i.e., eigensystem image when using 
 *   MEV coloring), and it will display a message box if this check fails.
 * - Implemented "dataSetChanged" and "dataSetRemoved" for all input data set types.
 *
 * 2011-01-20	Evert van Aart
 * - Added support for transformation matrices.
 * - Write ".tfm" file when saving fibers.
 * 
 * 2011-02-01	Evert van Aart
 * - Added support for bypassing the simplification filter.
 * - Added a "Delete Fibers" button.
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.0.
 * - When saving fibers, the plugin now automatically selects the 
 *   data directory defined in the default profile. 
 *
 * 2011-04-18	Evert van Aart
 * - Moved the simplifcation filter to the "Helpers" library.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.2.
 * - Improved attribute handling.
 *
 * 2011-07-12	Evert van Aart
 * - Version 1.0.3.
 * - "isVisible" attribute is now also checked for new data sets, not just changed ones.
 *
 */


/** Includes */

#include "FiberVisualizationPlugin.h"
#include "FiberVisualizationPipeline.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberVisualizationPlugin::FiberVisualizationPlugin() : Plugin("Fiber Visualization")
{
	// No data selected
    this->selectedData = -1;

	// No eigensystem image data
	this->currentEigenData = NULL;

	// Create a new assembly
    this->assembly = vtkPropAssembly::New();

	// Setup the GUI
    this->widget = new QWidget();
    this->ui = new Ui::FiberVisualizationForm();
    this->ui->setupUi(this->widget);

    // Link events in the GUI to function calls
	this->connectAll();

	// Enable or disable DUI controls
	this->setGUIEnable();
}


//------------------------------[ connectAll ]-----------------------------\\

void FiberVisualizationPlugin::connectAll()
{
    connect(this->ui->visibleCheckBox, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->coloringTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeColoringMethod()));
    connect(this->ui->lightingEnableCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->lightingAmbientSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->lightingDiffuseSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->lightingSpecularSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->lightingShineSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->shadowsEnableCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->shadowsAmbientSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->shadowsDiffuseSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    connect(this->ui->shadowsThicknessSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->shapeSimplifyEnable, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->shapeCombo, SIGNAL(currentIndexChanged(QString)), this, SLOT(changeShape()));
	connect(this->ui->shapeTubeUpdateButton, SIGNAL(clicked(bool)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->coloringTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->colorShiftValuesCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->colorUseAIWeightingCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->colorAICombo, SIGNAL(currentIndexChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->lutCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	connect(this->ui->dataList, SIGNAL(currentRowChanged(int)), this, SLOT(selectData(int)));
    connect(this->ui->colorSingleCustomButton, SIGNAL(clicked()), this, SLOT(changeSingleColor()));
    connect(this->ui->colorSingleWhiteButton, SIGNAL(clicked()), this, SLOT(changeSingleColorToWhite()));
    connect(this->ui->colorSingleAutoButton, SIGNAL(clicked()), this, SLOT(changeSingleColorToAuto()));
	connect(this->ui->applyToAllButton, SIGNAL(clicked()), this, SLOT(applySettingsToAll()));
	connect(this->ui->saveButton, SIGNAL(clicked()), this, SLOT(writeFibersToFile()));
	connect(this->ui->deleteButton, SIGNAL(clicked()), this, SLOT(deleteFibers()));
}


//----------------------------[ disconnectAll ]----------------------------\\

void FiberVisualizationPlugin::disconnectAll()
{
    disconnect(this->ui->visibleCheckBox, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->coloringTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeColoringMethod()));
    disconnect(this->ui->lightingEnableCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->lightingAmbientSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->lightingDiffuseSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->lightingSpecularSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->lightingShineSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->shadowsEnableCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->shadowsAmbientSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->shadowsDiffuseSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
    disconnect(this->ui->shadowsThicknessSpin, SIGNAL(valueChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->shapeSimplifyEnable, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->shapeCombo, SIGNAL(currentIndexChanged(QString)), this, SLOT(changeShape()));
	disconnect(this->ui->shapeTubeUpdateButton, SIGNAL(clicked(bool)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->coloringTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->colorShiftValuesCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->colorUseAIWeightingCheck, SIGNAL(toggled(bool)), this, SLOT(settingsFromGUIToPipeline()));
 	disconnect(this->ui->colorAICombo, SIGNAL(currentIndexChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->lutCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(settingsFromGUIToPipeline()));
	disconnect(this->ui->dataList, SIGNAL(currentRowChanged(int)), this, SLOT(selectData(int)));
    disconnect(this->ui->colorSingleCustomButton, SIGNAL(clicked()), this, SLOT(changeSingleColor()));
    disconnect(this->ui->colorSingleWhiteButton, SIGNAL(clicked()), this, SLOT(changeSingleColorToWhite()));
    disconnect(this->ui->colorSingleAutoButton, SIGNAL(clicked()), this, SLOT(changeSingleColorToAuto()));
	disconnect(this->ui->applyToAllButton, SIGNAL(clicked()), this, SLOT(applySettingsToAll()));
	disconnect(this->ui->saveButton, SIGNAL(clicked()), this, SLOT(writeFibersToFile()));
	disconnect(this->ui->deleteButton, SIGNAL(clicked()), this, SLOT(deleteFibers()));
}


//------------------------------[ Destructor ]-----------------------------\\

FiberVisualizationPlugin::~FiberVisualizationPlugin()
{
	// Remove the GUI
    delete this->widget; 
	this->widget = NULL;

	// Delete the assembly
    this->assembly->Delete();
}


//------------------------------[ getVtkProp ]-----------------------------\\

vtkProp * FiberVisualizationPlugin::getVtkProp()
{
    return this->assembly;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * FiberVisualizationPlugin::getGUI()
{
    return this->widget;
}


//-----------------------------[ setGUIEnable ]----------------------------\\

void FiberVisualizationPlugin::setGUIEnable()
{
	// If no fibers exist, disable all pages
	if (this->selectedData < 0)
	{
		this->ui->colorPage->setEnabled(false);
		this->ui->shadowsPage->setEnabled(false);
		this->ui->lightingPage->setEnabled(false);
		this->ui->shapePage->setEnabled(false);
		this->ui->visibleCheckBox->setEnabled(false);
		return;
	}

	// Enable all pages
	this->ui->colorPage->setEnabled(true);
	this->ui->shadowsPage->setEnabled(true);
	this->ui->lightingPage->setEnabled(true);
	this->ui->shapePage->setEnabled(true);
	this->ui->visibleCheckBox->setEnabled(true);

	// Coloring options
	bool enableColorDialog = (this->ui->coloringTypeComboBox->currentIndex() == this->FC_SingleColor);

	this->ui->colorSingleCustomButton->setEnabled(enableColorDialog);
	this->ui->colorSingleAutoButton->setEnabled(enableColorDialog);
	this->ui->colorSingleWhiteButton->setEnabled(enableColorDialog);
	this->ui->colorSingleGroup->setEnabled(enableColorDialog);

	bool enableColorVectorOptions =		this->ui->coloringTypeComboBox->currentIndex() == this->FC_MEV 
									 || this->ui->coloringTypeComboBox->currentIndex() == this->FC_Direction;
	this->ui->colorShiftValuesCheck->setEnabled(enableColorVectorOptions);
	this->ui->colorUseAIWeightingCheck->setEnabled(enableColorVectorOptions);
	bool enableAICombo =   (enableColorVectorOptions && this->ui->colorUseAIWeightingCheck->isChecked()) 
						 || this->ui->coloringTypeComboBox->currentIndex() == this->FC_AI;
	this->ui->colorAILabel->setEnabled(enableAICombo);
	this->ui->colorAICombo->setEnabled(enableAICombo);

	bool enableLUT =	this->ui->coloringTypeComboBox->currentIndex() == this->FC_AI || 
						this->ui->coloringTypeComboBox->currentIndex() == this->FC_FiberData;
	this->ui->lutLabel->setEnabled(enableLUT);
	this->ui->lutCombo->setEnabled(enableLUT);

	// GPU-related options
	bool enableGPUOptions =   (this->ui->shapeCombo->currentIndex() == FS_Streamlines); 	
	this->ui->lightingPage->setEnabled(enableGPUOptions);
	this->ui->shadowsPage->setEnabled(enableGPUOptions);

	// Lighting options
	bool enableLightingOptions = this->ui->lightingEnableCheck->isChecked() && enableGPUOptions;
	this->ui->lightingAmbientLabel->setEnabled(		enableLightingOptions		);
	this->ui->lightingDiffuseLabel->setEnabled(		enableLightingOptions		);
	this->ui->lightingSpecularLabel->setEnabled(	enableLightingOptions		);
	this->ui->lightingShineLabel->setEnabled(		enableLightingOptions		);
	this->ui->lightingAmbientSpin->setEnabled(		enableLightingOptions		);
	this->ui->lightingDiffuseSpin->setEnabled(		enableLightingOptions		);
	this->ui->lightingSpecularSpin->setEnabled(		enableLightingOptions		);
	this->ui->lightingShineSpin->setEnabled(		enableLightingOptions		);
	this->ui->lightingAmbientSlider->setEnabled(	enableLightingOptions		);
	this->ui->lightingDiffuseSlider->setEnabled(	enableLightingOptions		);
	this->ui->lightingSpecularSlider->setEnabled(	enableLightingOptions		);
	this->ui->lightingShineSlider->setEnabled(		enableLightingOptions		);

	// Shadows options
	bool enableShadowsOptions = this->ui->shadowsEnableCheck->isChecked() && enableLightingOptions;
	this->ui->shadowsAmbientLabel->setEnabled(		enableLightingOptions		);
	this->ui->shadowsDiffuseLabel->setEnabled(		enableLightingOptions		);
	this->ui->shadowsThicknessLabel->setEnabled(	enableLightingOptions		);
	this->ui->shadowsAmbientSpin->setEnabled(		enableLightingOptions		);
	this->ui->shadowsDiffuseSpin->setEnabled(		enableLightingOptions		);
	this->ui->shadowsThicknessSpin->setEnabled(		enableLightingOptions		);
	this->ui->shadowsAmbientSlider->setEnabled(		enableLightingOptions		);
	this->ui->shadowsDiffuseSlider->setEnabled(		enableLightingOptions		);
	this->ui->shadowsThicknessSlider->setEnabled(	enableLightingOptions		);

	// Shape options
	this->ui->shapePLLengthLabel->setEnabled(this->ui->shapeSimplifyEnable->isChecked());
	this->ui->shapePLLengthSpin->setEnabled(this->ui->shapeSimplifyEnable->isChecked());

	int i = this->ui->shapeCombo->currentIndex();
	this->ui->shapeTubeRadiusLabel->setEnabled(	i == FS_Streamtubes			);
	this->ui->shapeTubeRadiusSpin->setEnabled(	i == FS_Streamtubes			);
	this->ui->shapeTubeHyperLabel->setEnabled(	i == FS_Hyperstreamtubes	|| 
												i == FS_Hyperstreamprisms	|| 
												i == FS_Streamribbons		);
	this->ui->shapeTubeHyperSpin->setEnabled(	i == FS_Hyperstreamtubes	|| 
												i == FS_Hyperstreamprisms	|| 
												i == FS_Streamribbons		);
	this->ui->shapeTubeSidesLabel->setEnabled(	i == FS_Streamtubes			|| 
												i == FS_Hyperstreamtubes	);
	this->ui->shapeTubeSidesSpin->setEnabled(	i == FS_Streamtubes			|| 
												i == FS_Hyperstreamtubes	);
	this->ui->eigensystemLabel->setEnabled(		i == FS_Hyperstreamprisms	|| 
												i == FS_Streamribbons		||
												i == FS_Hyperstreamtubes	);
	this->ui->eigensystemCombo->setEnabled(		i == FS_Hyperstreamprisms	|| 
												i == FS_Streamribbons		||
												i == FS_Hyperstreamtubes	);
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void FiberVisualizationPlugin::dataSetAdded(data::DataSet * ds)
{
	// Check if the data set exists
	if (!ds)
		return;

	// Add eigensystem image data
	if (ds->getKind() == "eigen")
	{
		this->addEigenDataSet(ds);
	}

	// Add Anisotropy Index image data 
	else if (ds->getKind() == "scalar volume")
	{
		this->addAIDataSet(ds);
	}

	// Add color look-up table
	else if (ds->getKind() == "transfer function")
	{
		this->addLUTDataSet(ds);
	}

	// Add fibers
	else if (ds->getKind() == "fibers") 
	{
		this->addFiberDataSet(ds);
	}
}


//---------------------------[ addFiberDataSet ]---------------------------\\

bool FiberVisualizationPlugin::addFiberDataSet(bmia::data::DataSet *ds)
{
	// Get the polydata from the data set
    vtkPolyData * polydata = ds->getVtkPolyData();

	// Check if the polygon data exists
    if (!polydata)
		return false;

    // Add the new data set to the list of currently available fiber sets
    this->fiberSets.append(ds);

	// Create a new actor for the fiber set
	vtkActor * actor = vtkActor::New();

	// New fiber set is visible by default
	actor->SetVisibility(true);

	// Create a visualization pipeline for the actor
	FiberVisualizationPipeline * newPipeline = new FiberVisualizationPipeline(this->core()->out());

	// Store new pointers in an "actorInfo" struct
	actorInfo newActorInfo;
	newActorInfo.actor = actor;
	newActorInfo.actorPipeline = newPipeline;

	// Add the transformation matrix, if available
	vtkObject * m = NULL;

	if (ds->getAttributes()->getAttribute("transformation matrix", m))
	{
		actor->SetUserMatrix(vtkMatrix4x4::SafeDownCast(m));
	}

	// Get the visibility attribute
	double attribute;

	if (ds->getAttributes()->getAttribute("isVisible", attribute))
	{
		// Hide the fibers if necessary
		if (attribute < 0.0)
			actor->SetVisibility(0);
	}

    // Add the actor to the assembly to be rendered:
    this->assembly->AddPart(actor);

    // Add the actor information to the list
	this->actors.append(newActorInfo);

    // Add the new data set to the list of data sets in the GUI
    this->ui->dataList->addItem(ds->getName());

    // Select the newly added dataset. Since the new actor is appended to the 
	// list of actors, its index is one less than the size of the list.

	this->selectData(this->actors.size() - 1);

	// Automatically color the fibers
	this->changeSingleColorToAuto();

	// Copy GUI settings to pipeline to setup the pipeline
	this->settingsFromGUIToPipeline();

	return true;
}


//----------------------------[ addAIDataSet ]-----------------------------\\

bool FiberVisualizationPlugin::addAIDataSet(data::DataSet * ds)
{
	// Check if the data set contains image data
	if (!ds->getVtkImageData())
		return false;

	// Check if the image data contains point data
	if (!ds->getVtkImageData()->GetPointData())
		return false;

	// Add the data set to the list of data sets
	this->aiSets.append(ds);

	// Add the name of the data set to the combo box
	this->ui->colorAICombo->addItem(ds->getName());

	// Select the first image
	if (this->ui->colorAICombo->count() == 1)
		this->ui->colorAICombo->setCurrentIndex(0);

	return true;
}


//---------------------------[ addEigenDataSet ]---------------------------\\

bool FiberVisualizationPlugin::addEigenDataSet(data::DataSet * ds)
{
	// Check if the data set contains image data
	if (!ds->getVtkImageData())
		return false;

	// Check if the image data contains point data
	if (!ds->getVtkImageData()->GetPointData())
		return false;

	// Add the data set to the list of data sets
	this->eigenSets.append(ds);

	// Add the name of the data set to the combo box
	this->ui->eigensystemCombo->addItem(ds->getName());

	return true;
}

//----------------------------[ addLUTDataSet ]-----------------------------\\

bool FiberVisualizationPlugin::addLUTDataSet(data::DataSet * ds)
{
	// Check if the data set contains a VTK object
	if (!ds->getVtkObject())
		return false;

	// Check if the VTK object can be cast to a LUT
	if (!(vtkScalarsToColors::SafeDownCast(ds->getVtkObject())))
		return false;

	// Add the data set to the list of data sets
	this->lutSets.append(ds);

	// Add the name of the data set to the combo box
	this->ui->lutCombo->addItem(ds->getName());

	// Select the first item
	if (this->ui->lutCombo->currentIndex() < 0)
	{
		this->ui->lutCombo->setCurrentIndex(0);
	}

	return true;
}


//----------------------------[ dataSetChanged ]---------------------------\\

void FiberVisualizationPlugin::dataSetChanged(data::DataSet * ds)
{
	// Check if the data set exists
	if (!ds)
		return;

	// Anisotropy Index images
	if (ds->getKind() == "scalar volume" && this->aiSets.contains(ds))
	{
		// Change the name of the data set
		int index = this->aiSets.indexOf(ds);
		this->ui->colorAICombo->setItemText(index, ds->getName());

		// If we're using this AI image for coloring, redraw the fibers
		if (	this->ui->colorAICombo->currentIndex() == index &&
				(	this->ui->coloringTypeComboBox->currentIndex() == this->FC_AI ||
					(	this->ui->coloringTypeComboBox->currentIndex() == this->FC_Direction && this->ui->colorUseAIWeightingCheck->isChecked()) ||
					(	this->ui->coloringTypeComboBox->currentIndex() == this->FC_MEV       && this->ui->colorUseAIWeightingCheck->isChecked()) ) )
		{
			this->changeColoringMethod();
		}

		return;
	}

	// Eigensystem images
	if (ds->getKind() == "eigen" && this->eigenSets.contains(ds))
	{
		// Change the name of the data set
		int index = this->eigenSets.indexOf(ds);
		this->ui->eigensystemCombo->setItemText(index, ds->getName());

		// If we're using this image for the fiber shape or the coloring, redraw the fiber
		if ( this->ui->eigensystemCombo->currentIndex() == index &&
				(	this->ui->shapeCombo->currentIndex() == this->FS_Hyperstreamtubes  ||
					this->ui->shapeCombo->currentIndex() == this->FS_Hyperstreamprisms ||
					this->ui->shapeCombo->currentIndex() == this->FS_Streamribbons     ) )
		{
			this->changeShape();
		}

		if (this->ui->eigensystemCombo->currentIndex() == index &&
				this->ui->coloringTypeComboBox->currentIndex() == this->FC_MEV)
		{
			this->changeColoringMethod();
		}

		return;
	}

	// Look-Up Tables
	if (ds->getKind() == "transfer function" && this->lutSets.contains(ds))
	{
		// Change the name of the data set
		int index = this->lutSets.indexOf(ds);
		this->ui->lutCombo->setItemText(index + 1, ds->getName());

		// If we're currently using this LUT, redraw the fibers
		if (	(this->ui->lutCombo->currentIndex() - 1) == index &&
				(	this->ui->coloringTypeComboBox->currentIndex() == this->FC_AI			||
					this->ui->coloringTypeComboBox->currentIndex() == this->FC_FiberData    ) )
		{
			this->settingsFromGUIToPipeline();
		}
	}

	// Check if the data is of type "fibers", and if it in the list of 
	// data sets that have been added to this plugin.

	if (ds->getKind() != "fibers" || !this->fiberSets.contains(ds))
	{
		return;
	}
	
	// Attribute value of the data set
	double attribute;

	// Check if the data set contains polydata
	if (!ds->getVtkPolyData())
		return;

	// Get the index of the data set in the list of data sets
	int dsIndex = this->fiberSets.indexOf(ds);

	// Get the actor of the current data set
	vtkActor * currentActor = this->actors.at(dsIndex).actor;

	// Check if the actor exists
	if (!currentActor)
		return;

	// Add the transformation matrix, if available
	vtkObject * m = NULL;

	if (ds->getAttributes()->getAttribute("transformation matrix", m))
	{
		currentActor->SetUserMatrix(vtkMatrix4x4::SafeDownCast(m));
	}
	else
	{
		// If no transformation is available, set the identity matrix
		vtkMatrix4x4 * id = vtkMatrix4x4::New();
		id->Identity();
		currentActor->SetUserMatrix(id);
	}

	// Set the visibility of the data set
	if (ds->getAttributes()->getAttribute("isVisible", attribute))
	{
		// 1.0 means that the data set should be visible...
		if (attribute == 1.0)
			currentActor->SetVisibility(1);
		// ... and -1.0 means that it should be invisible.
		else if (attribute == -1.0)
			currentActor->SetVisibility(0);

		// (all other values means that the visibility shouldn't change)

		// Reset the attribute to zero
		ds->getAttributes()->addAttribute("isVisible", 0.0);

		// Copy new settings to the GUI
		this->settingsFromPipelineToGUI();
	}

	// Only continue if the "updatePipeline" attribute has value 1.0
	if (ds->getAttributes()->getAttribute("updatePipeline", attribute))
	{
		if (attribute != 1.0)
		{
			this->core()->render();
			return;
		}
	}

	// If the attribute did not yet exist, create it now
	else
	{
		ds->getAttributes()->addAttribute("updatePipeline", 1.0);
	}

	// Get the pipeline of the current actor
	FiberVisualizationPipeline * currentPipeline = this->actors.at(dsIndex).actorPipeline;

	// Check if the current pipeline exists
	if (!currentPipeline)
		return;

	// Rebuild pipeline using new data
	currentPipeline->rebuildPipeline = true;
	currentPipeline->setupPipeline(ds->getVtkPolyData(), currentActor);

	// Tell the pipeline that the input data has changed
	currentPipeline->modifiedInput();

	// Get the list item of the fiber set
	QListWidgetItem * currentListItem = this->ui->dataList->item(dsIndex);

	// Update the fiber set name
	if (currentListItem)
		currentListItem->setText(ds->getName());

	// Update the selected fiber set name
	if (dsIndex == this->selectedData)
		this->ui->dataSetName->setText(ds->getName());

	// Re-render the fibers
	this->core()->render();

	// We're done, so set the "updatePipeline" attribute to 0.0
	ds->getAttributes()->addAttribute("updatePipeline", 0.0);
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void FiberVisualizationPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Check if the data set exists
	if (!ds)
		return;

	// Anisotropy Index images
	if (ds->getKind() == "scalar volume" && this->aiSets.contains(ds))
	{
		// Remove the data set form the list and the GUI
		int index = this->aiSets.indexOf(ds);
		this->ui->colorAICombo->removeItem(index);
		this->aiSets.removeAt(index);

		// Redraw the fibers if necessary
		if (	this->ui->coloringTypeComboBox->currentIndex() == this->FC_AI ||
			(	this->ui->coloringTypeComboBox->currentIndex() == this->FC_Direction && this->ui->colorUseAIWeightingCheck->isChecked()) ||
			(	this->ui->coloringTypeComboBox->currentIndex() == this->FC_MEV       && this->ui->colorUseAIWeightingCheck->isChecked()) )
		{
			this->changeColoringMethod();
		}

		return;
	}

	// Eigensystem images
	if (ds->getKind() == "eigen" && this->eigenSets.contains(ds))
	{
		// Remove the data set form the list and the GUI
		int index = this->eigenSets.indexOf(ds);
		this->ui->eigensystemCombo->removeItem(index);
		this->eigenSets.removeAt(index);

		// Redraw the fibers if necessary
		if ( this->ui->shapeCombo->currentIndex() == this->FS_Hyperstreamtubes  ||
			this->ui->shapeCombo->currentIndex() == this->FS_Hyperstreamprisms ||
			this->ui->shapeCombo->currentIndex() == this->FS_Streamribbons     )
		{
			this->changeShape();
		}

		if (this->ui->coloringTypeComboBox->currentIndex() == this->FC_MEV)
		{
			this->changeColoringMethod();
		}

		return;
	}

	// Look-Up Tables
	if (ds->getKind() == "transfer function" && this->lutSets.contains(ds))
	{
		// Remove the data set form the list and the GUI
		int index = this->lutSets.indexOf(ds);
		this->ui->lutCombo->removeItem(index);
		this->lutSets.removeAt(index);

		// Redraw the fibers if necessary
		if ( this->ui->coloringTypeComboBox->currentIndex() == this->FC_AI			||
			this->ui->coloringTypeComboBox->currentIndex() == this->FC_FiberData   )
		{
			this->settingsFromGUIToPipeline();
		}
	}

	// Check if the data is of type "fibers", and if it in the list of 
	// data sets that have been added to this plugin.

	if (ds->getKind() != "fibers" || !this->fiberSets.contains(ds))
	{
		return;
	}

	// Get the index of the data set in the list of data sets
	int dsIndex = this->fiberSets.indexOf(ds);

	// Remove the data set pointer from the list
	this->fiberSets.removeAt(dsIndex);

	// Disconnect the signal for now
	disconnect(this->ui->dataList, SIGNAL(currentRowChanged(int)), this, SLOT(selectData(int)));

	// If this is the only item in the list...
	if (this->fiberSets.isEmpty())
	{
		// ...remove all entries and selections in the list view...
		this->ui->dataList->clear();

		// ...and clear the selection parameters.
		this->selectedData = -1;
		this->selectData(-1);
	}
	else
	{
		// Remove the entry from the list view
		this->ui->dataList->takeItem(dsIndex);

		// If the item we just deleted was the selected item, reset the
		// selected data variable to -1 to clear all selections.

		if (this->selectedData == dsIndex)
		{
			this->selectedData = -1;
		}

		// If the selected item was below the deleted item in the list,
		// we need to decrement the selected data variable, since it
		// has just moved up one row.

		else if (this->selectedData > dsIndex)
		{
			this->selectedData--;
		}

		// Set or reset the selected data
		this->selectData(this->selectedData);
	}

	// Reconnect the signal
	connect(this->ui->dataList, SIGNAL(currentRowChanged(int)), this, SLOT(selectData(int)));

	// Get the actor belonging to the data set
	actorInfo selectedActor = this->actors.at(dsIndex);

	// Remove the actor from the list of actors
	this->actors.removeAt(dsIndex);

	// Remove the actor from the assembly
	this->assembly->RemovePart(selectedActor.actor);

	// Delete the actor pipeline
	delete selectedActor.actorPipeline;

	// Delete the actor. This also deletes this actor's mapper.
	selectedActor.actor->Delete();

	// Render now to ensure that deleted fibers immediately leave the 
	// screen (as opposed to the first time the user moves the camera)

	this->core()->render();

}


//------------------------------[ selectData ]-----------------------------\\

void FiberVisualizationPlugin::selectData(int row)
{
	// Store the new index
    this->selectedData = row;

	// Deselect data sets
	if (row < 0)
	{
		this->ui->dataList->setCurrentRow(-1);
		this->ui->dataSetName->setText("No fibers selected.");
		return;
	}

	// Set the selected data set text
    this->ui->dataSetName->setText(this->fiberSets.at(this->selectedData)->getName());

	// Update the GUI with the settings of the current fiber set
	this->settingsFromPipelineToGUI();
}


//----------------------[ settingsFromGUIToPipeline ]----------------------\\

void FiberVisualizationPlugin::settingsFromGUIToPipeline()
{
	// Check if the "selectedData" index is within range
	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return;

	// Get the selected actor
	vtkActor * currentActor = this->actors.at(this->selectedData).actor;

	// Check if the selected actor exists
	if (!currentActor)
		return;

	// Set the visibility of the fiber set
	currentActor->SetVisibility((int) this->ui->visibleCheckBox->isChecked());

	// Get pipeline of the selected actor
	FiberVisualizationPipeline * currentPipeline = this->actors.at(this->selectedData).actorPipeline;

	// Check if the current pipeline exists
	if (!currentPipeline)
		return;

	// Get the current fiber data set
	data::DataSet * ds = this->fiberSets.at(this->selectedData);

	// Setup simplification filter
	currentPipeline->setupSimplifyFilter((float) this->ui->shapePLLengthSpin->value(), this->ui->shapeSimplifyEnable->isChecked());

	// Set the eigensystem data image
	if (this->eigenSets.empty())
	{
		currentPipeline->eigenImageData = NULL;
		currentPipeline->eigenImageIndex = -1;
	}
	else
	{
		int currentEigenData = this->ui->eigensystemCombo->currentIndex();

		if (currentEigenData >= 0)
			currentPipeline->eigenImageData = eigenSets.at(currentEigenData)->getVtkImageData();

		currentPipeline->eigenImageIndex = currentEigenData;
	}

	// Set the AI data image
	if (this->aiSets.empty())
	{
		currentPipeline->aiImageData = NULL;
		currentPipeline->aiImageIndex = -1;
	}
	else
	{
		int currentAIData = this->ui->colorAICombo->currentIndex();

		if (currentAIData >= 0)
		{
			data::DataSet * currentAIDS = this->aiSets.at(currentAIData);

			if (currentAIDS->getVtkImageData()->GetActualMemorySize() == 0)
			{
				currentAIDS->getVtkImageData()->Update();
				this->core()->data()->dataSetChanged(currentAIDS);
			}

			currentPipeline->aiImageData = currentAIDS->getVtkImageData();
		}

		currentPipeline->aiImageIndex = currentAIData;
	}

	// Setup the coloring filter
	currentPipeline->setupColorFilter( (FiberColor)	this->ui->coloringTypeComboBox->currentIndex(), 
													this->ui->colorShiftValuesCheck->isChecked(), 
													this->ui->colorUseAIWeightingCheck->isChecked() );

	// Setup the shape filter
	currentPipeline->setupShapeFilter( (FiberShape) this->ui->shapeCombo->currentIndex(), 
													this->ui->shapeTubeSidesSpin->value(),
													(float) this->ui->shapeTubeRadiusSpin->value(),
													(float) this->ui->shapeTubeHyperSpin->value()	);

	// Set the Look-Up Table
	if (this->ui->lutCombo->currentIndex() == 0)
	{
		currentPipeline->lut = NULL;
		currentPipeline->lutIndex = -1;
	}
	else
	{
		int lutIndex = this->ui->lutCombo->currentIndex() - 1;
		data::DataSet * lutDS = this->lutSets.at(lutIndex);
		vtkObject * lutObject = lutDS->getVtkObject();
		currentPipeline->lut = vtkScalarsToColors::SafeDownCast(lutObject);
		currentPipeline->lutIndex = lutIndex;
	}

	// Setup the mapper
	currentPipeline->setupMapper();

	// Set the lighting options
	currentPipeline->setupLighting(	this->ui->lightingEnableCheck->isChecked(),
									0.01f * (float) this->ui->lightingAmbientSpin->value(),
									0.01f * (float) this->ui->lightingDiffuseSpin->value(),
									0.01f * (float) this->ui->lightingSpecularSpin->value(),
									(float) this->ui->lightingShineSpin->value() );

	// Set the shadows options
	currentPipeline->setupShadows(	this->ui->shadowsEnableCheck->isChecked(),
									0.01f * (float) this->ui->shadowsAmbientSpin->value(),
									0.01f * (float) this->ui->shadowsDiffuseSpin->value(),
									(float) this->ui->shadowsThicknessSpin->value() );

	// Rebuild the pipeline if necessary
	currentPipeline->setupPipeline(ds->getVtkPolyData(), currentActor);

	// Enable or disable controls based on current settings
	this->setGUIEnable();

	// Re-render the fibers
	this->core()->render();
}


//--------------------------[ changeSingleColor ]--------------------------\\

void FiberVisualizationPlugin::changeSingleColor()
{
	// Check if the "selectedData" index is within range
	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return;

	// Get the selected actor
	vtkActor * currentActor = this->actors.at(this->selectedData).actor;

	// Check if the selected actor exists
	if (!currentActor)
		return;

	// Get the actor properties
	vtkProperty * actorProperty = currentActor->GetProperty();

	// Check if the properties exist
    if (!actorProperty)
		return;

	// Get the current fiber color
    double oldColorRGB[3];
    QColor oldColor;
    actorProperty->GetColor(oldColorRGB);
    oldColor.setRgbF(oldColorRGB[0], oldColorRGB[1], oldColorRGB[2]);    

	// Use a color dialog to get the new color
    QColor newColor = QColorDialog::getColor(oldColor, 0);

	// If the new color is valid...
    if (newColor.isValid())
	{
		// ...set the new color...
		actorProperty->SetColor(newColor.redF(), newColor.greenF(), newColor.blueF());

		// ...and re-render the fibers.
		this->core()->render();
	}
}


//-----------------------[ changeSingleColorToWhite ]----------------------\\

void FiberVisualizationPlugin::changeSingleColorToWhite()
{
	// Get actor properties (see "changeSingleColor")

	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return;

	vtkActor * currentActor = this->actors.at(this->selectedData).actor;

	if (!currentActor)
		return;

	vtkProperty * actorProperty = currentActor->GetProperty();

	// Set the color to white
	actorProperty->SetColor(1.0, 1.0, 1.0);

	// Re-render the screen
	this->core()->render();
}


//-----------------------[ changeSingleColorToAuto ]----------------------\\

void FiberVisualizationPlugin::changeSingleColorToAuto()
{
	// Get actor properties (see "changeSingleColor")

	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return;

	vtkActor * currentActor = this->actors.at(this->selectedData).actor;

	if (!currentActor)
		return;

	vtkProperty * actorProperty = currentActor->GetProperty();

	// Create a color chart containing (by default) seven distinct colors
	vtkQtChartColors colorChart;

	// Get the color corresponding to the index of the current fiber set. Since
	// the first color of the color is black, we skip it by taking the
	// modulo six of the index and adding one.

	QColor autoColor = colorChart.getColor((this->selectedData % 6) + 1);

	// Set the color to the automatically generated color
	actorProperty->SetColor(autoColor.redF(), autoColor.greenF(), autoColor.blueF());

	// Re-render the screen
	this->core()->render();
}


//----------------------[ settingsFromPipelineToGUI ]----------------------\\

void FiberVisualizationPlugin::settingsFromPipelineToGUI()
{
	// Check if the "selectedData" index is within range
	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return;

	// Get the selected actor
	vtkActor * currentActor = this->actors.at(this->selectedData).actor;

	// Check if the selected actor exists
	if (!currentActor)
		return;

	// Get pipeline of the selected actor
	FiberVisualizationPipeline * currentPipeline = this->actors.at(this->selectedData).actorPipeline;

	// Check if the pipeline exists
	if (!currentPipeline)
		return;

	// Disconnect all controls to prevent their signals from triggering the "SLOT" functions
    this->disconnectAll();

	// Set visibility checkbox
	this->ui->visibleCheckBox->setChecked((bool) currentActor->GetVisibility());

	// Set options of shape filter
	this->ui->shapeCombo->setCurrentIndex(currentPipeline->ShapeType);
	this->ui->shapePLLengthSpin->setValue((double) currentPipeline->SimplifyStepSize);
	this->ui->shapeTubeHyperSpin->setValue((double) currentPipeline->ShapeHyperScale);
	this->ui->shapeTubeRadiusSpin->setValue((double) currentPipeline->ShapeRadius);
	this->ui->shapeTubeSidesSpin->setValue(currentPipeline->ShapeNumberOfSides);
	this->ui->eigensystemCombo->setCurrentIndex(currentPipeline->eigenImageIndex);

	// Set options of the coloring filter
	this->ui->colorAICombo->setCurrentIndex(currentPipeline->aiImageIndex);
	this->ui->coloringTypeComboBox->setCurrentIndex(currentPipeline->ColorType);
	this->ui->colorShiftValuesCheck->setChecked(currentPipeline->ColorShiftValues);
	this->ui->colorUseAIWeightingCheck->setChecked(currentPipeline->ColorUseAIWeighting);
	this->ui->lutCombo->setCurrentIndex(currentPipeline->lutIndex + 1);

	// Set the lighting options
	this->ui->lightingEnableCheck->setChecked(currentPipeline->LightingEnable);
	this->ui->lightingAmbientSpin->setValue((int) (100.0f * currentPipeline->LightingAmbient)	);
	this->ui->lightingDiffuseSpin->setValue((int) (100.0f * currentPipeline->LightingDiffuse)	);
	this->ui->lightingSpecularSpin->setValue((int)(100.0f * currentPipeline->LightingSpecular)	);
	this->ui->lightingShineSpin->setValue((int) currentPipeline->LightingSpecularPower	);

	// Set the shadows options
	this->ui->shadowsEnableCheck->setChecked(currentPipeline->ShadowsEnable);
	this->ui->shadowsAmbientSpin->setValue((int) (100.0f * currentPipeline->ShadowsAmbient)	);
	this->ui->shadowsDiffuseSpin->setValue((int) (100.0f * currentPipeline->ShadowsDiffuse)	);
	this->ui->shadowsThicknessSpin->setValue(currentPipeline->ShadowsWidth);
    
	// Reconnect all controls to their respective "SLOT" function
    this->connectAll();

	// Enable or disable controls based on current settings
	this->setGUIEnable();
}


//--------------------------[ applySettingsToAll ]-------------------------\\

void FiberVisualizationPlugin::applySettingsToAll()
{
	// Check if the "selectedData" index is within range
	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return;

	// Copy settings from selected pipeline to the GUI
	this->settingsFromPipelineToGUI();

	// Make a copy of the current selection
	int originalSelectedData = this->selectedData;

	// Loop through all fiber sets
	for (int d = 0; d < this->fiberSets.size(); ++d)
	{
		// Set the "selectedData" index to the current fiber set
		this->selectedData = d;

		// Copy the settings from the GUI to the newly selected pipeline
		this->settingsFromGUIToPipeline();
	}

	// Restore the original "selectedData" index
	this->selectedData = originalSelectedData;
}


//--------------------------[ writeFibersToFile ]--------------------------\\

void FiberVisualizationPlugin::writeFibersToFile()
{
	// Get the current data directory
	QDir dataDir = this->core()->getDataDirectory();

	// Open a file dialog to get a filename
	QString fileName = QFileDialog::getSaveFileName(NULL, "Write Fibers", dataDir.absolutePath(), " Fibers (*.fbs)");

	// Check if the filename is correct
	if (fileName.isEmpty())
		return;

	// Convert the QString to a character array
	QByteArray ba = fileName.toAscii();
	char * fileNameChar = ba.data();

	// Get the polydata object containing the fibers
	vtkPolyData * output = this->getSelectedFibers();

	// Check if the fibers exist
	if (!output)
		return;
	
	// Create a polydata writer
	vtkPolyDataWriter * writer = vtkPolyDataWriter::New();

	// Configure the writer
	writer->SetFileName(fileNameChar);
	writer->SetInput(output);
	writer->SetFileTypeToASCII();
	
	// Enable progress bar for the writer
	this->core()->out()->createProgressBarForAlgorithm(writer, "Fiber Visualization", "Writing fibers to file...");

	// Write output file
	writer->Write();

	// Disable progress bar for the writer
	this->core()->out()->deleteProgressBarForAlgorithm(writer);

	// Delete the writer
	writer->Delete();

	// Get the current fiber data set
	data::DataSet * fiberDS = this->fiberSets.at(this->selectedData);
	
	vtkObject * attObject;

	// Check if the fiber data set contains a transformation matrix
	if (fiberDS->getAttributes()->getAttribute("transformation matrix", attObject))
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


//------------------------[ changeColoringMethod ]-------------------------\\

void FiberVisualizationPlugin::changeColoringMethod()
{
	// Get the fibers
	vtkPolyData * currentFibers = this->getSelectedFibers();

	if (!currentFibers)
		return;

	bool valid = false;

	// Fiber Data using LUTs
	if (this->ui->coloringTypeComboBox->currentIndex() == this->FC_FiberData)
	{
		// Check if the fibers contain point data
		if (currentFibers->GetPointData())
		{
			// Check fi the point data contains a scalar array
			if (vtkDataArray * scalars = currentFibers->GetPointData()->GetScalars())
			{
				// Scalar array should have as many points as the input fiber set, and at least one component
				if (scalars->GetNumberOfTuples() == currentFibers->GetNumberOfPoints() && scalars->GetNumberOfComponents() > 0)
				{
					valid = true;
				}
			}
		}

		// Invalid data set
		if (!valid)
		{
			QMessageBox::warning(this->getGUI(), "Fiber Visualization", "Selected fiber data set does not contain a suitable scalar array!", QMessageBox::Ok, QMessageBox::Ok);
			this->ui->coloringTypeComboBox->setCurrentIndex(this->FC_SingleColor);
		}
	}

	// Cell Data using RGB
	else if (this->ui->coloringTypeComboBox->currentIndex() == this->FC_CellData)
	{
		// Check if the fibers contain cell data
		if (vtkCellData * celldata = currentFibers->GetCellData())
		{
			// Cell data should have as many tuples as the number of input fibers, and three components
			if (celldata->GetNumberOfTuples() == currentFibers->GetNumberOfLines() &&
				celldata->GetNumberOfComponents() == 3)
			{
					valid = true;
			}
		}

		// Invalid data set
		if (!valid)
		{
			QMessageBox::warning(this->getGUI(), "Fiber Visualization", "Selected fiber data set does not contain suitable RGB cell data!", QMessageBox::Ok, QMessageBox::Ok);
			this->ui->coloringTypeComboBox->setCurrentIndex(this->FC_SingleColor);
		}
	}

	// Main Eigenvector
	else if (this->ui->coloringTypeComboBox->currentIndex() == this->FC_MEV)
	{
		// There should be at least one eigensystem image
		if (this->eigenSets.size() > 0)
		{
			valid = true;

			// If no eigensystem image has been selected, select the first one
			if (this->ui->eigensystemCombo->currentIndex() < 0)
			{
				this->ui->eigensystemCombo->setCurrentIndex(0);
			}
		}

		// Invalid data set
		if (!valid)
		{
			QMessageBox::warning(this->getGUI(), "Fiber Visualization", "No eigensystem data available!", QMessageBox::Ok, QMessageBox::Ok);
			this->ui->coloringTypeComboBox->setCurrentIndex(this->FC_SingleColor);
		}
	}

	// Anisotropy Index
	else if (this->ui->coloringTypeComboBox->currentIndex() == this->FC_AI)
	{
		// There should be at least one AI image
		if (this->aiSets.size() > 0)
		{
			valid = true;

			// If no AI image has been selected, select the first one
			if (this->ui->colorAICombo->currentIndex() < 0)
			{
				this->ui->colorAICombo->setCurrentIndex(0);
			}
		}

		// Invalid data set
		if (!valid)
		{
			QMessageBox::warning(this->getGUI(), "Fiber Visualization", "No anisotropy index data available!", QMessageBox::Ok, QMessageBox::Ok);
			this->ui->coloringTypeComboBox->setCurrentIndex(this->FC_SingleColor);
		}
	}

	// Redraw the fibers
	this->settingsFromGUIToPipeline();
}


//-----------------------------[ changeShape ]-----------------------------\\

void FiberVisualizationPlugin::changeShape()
{
	// Get the current fibers
	vtkPolyData * currentFibers = this->getSelectedFibers();

	if (!currentFibers)
		return;

	bool valid = false;

	// These three shape types need eigensystem data
	if (	this->ui->shapeCombo->currentIndex() == FS_Hyperstreamtubes		|| 
			this->ui->shapeCombo->currentIndex() == FS_Hyperstreamprisms	|| 
			this->ui->shapeCombo->currentIndex() == FS_Streamribbons		)
	{
		// Check if the plugin contains an eigensystem data set
		if (this->eigenSets.size() > 0)
		{
			valid = true;

			// Select the first image
			if (this->ui->eigensystemCombo->currentIndex() < 0)
			{
				this->ui->eigensystemCombo->setCurrentIndex(0);
			}
		}

		// Invalid data set
		if (!valid)
		{
			QMessageBox::warning(this->getGUI(), "Fiber Visualization", "No eigensystem data available!", QMessageBox::Ok, QMessageBox::Ok);
			this->ui->shapeCombo->setCurrentIndex(this->FS_Streamlines);
		}
	}

	// Redraw the fibers
	this->settingsFromGUIToPipeline();
}


//--------------------------[ getSelectedFibers ]--------------------------\\

vtkPolyData * FiberVisualizationPlugin::getSelectedFibers()
{
	// Check if the "selectedData" index is within range
	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return NULL;

	// Get the current fiber data set
	data::DataSet * ds = this->fiberSets.at(this->selectedData);

	// Check if the data set exists
	if (!ds)
		return NULL;

	// Get the polydata object containing the fibers
	return (ds->getVtkPolyData());
}


//----------------------------[ deleteFibers ]-----------------------------\\

void FiberVisualizationPlugin::deleteFibers()
{
	// Check if the "selectedData" index is within range
	if (this->selectedData < 0 || this->selectedData >= this->fiberSets.size())
		return;

	// Get the current fiber data set
	data::DataSet * ds = this->fiberSets.at(this->selectedData);

	// Check if the data set exists
	if (!ds)
		return;

	// Delete this data set
	this->core()->data()->removeDataSet(ds);
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libFiberVisualizationPlugin, bmia::FiberVisualizationPlugin)
