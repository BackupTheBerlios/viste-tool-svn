/*
 * HARDIFiberTrackingPlugin.cxx
 *
 * 2011-10-14	Anna Vilanova
 * - Version 1.0.0.
 * - First version
 *
 * 2011-11-03 Bart van Knippenberg
 * - Disabled autamatic range setting for the scalar values in the ui
 *
 * 2013-25-03 Mehmet Yusufoglu, Bart Van Knippenberg
 * - dataSetAdded function also accepts "discrete sphere" as input to the list. 
 * - UpdateFibers function calls HARDIFiberTrackingFilter with a data type parameter(sphericalHarmonics) anymore, 
 * parameter is either 1 or 0 depending on the data type read.
 *
 * 2013-29-05 Mehmet Yusufoglu, Bart Van Knippenberg
 * - scalar volumes should not be added to the HARDI data combo box. Removed.
 *
 */




/** Includes */

#include "HARDIFiberTrackingPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

HARDIFiberTrackingPlugin::HARDIFiberTrackingPlugin() : Plugin("HARDI Fiber Tracking")
{
	// Change the name of the plugin if necessary
	this->changePluginName();

	// Create a new Qt widget
    this->widget = new QWidget();

	// Create a new GUI form
    this->ui = new Ui::HARDIFiberTrackingForm();

	// Setup the GUI
    this->ui->setupUi(this->widget);

    // Link the "Update" button to the "updateFibers" function
	connect(this->ui->updateButton, SIGNAL(clicked()), this, SLOT(updateFibers()));

	// Connect GUI elements
	connect(this->ui->AIDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(updateScalarRange()));
	connect(this->ui->trackingMethodCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(changeMethod(int)));

	// Clear the toolbox to get rid of the dummy page
	this->clearToolbox();
}


//-----------------------------[ Destructor ]------------------------------\\

HARDIFiberTrackingPlugin::~HARDIFiberTrackingPlugin()
{
	// Remove the Qt widget
	delete this->widget; 
	this->widget = NULL;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * HARDIFiberTrackingPlugin::getGUI()
{
	// Return the Qt widget
    return this->widget;
}


//--------------------------[ updateScalarRange ]--------------------------\\

void HARDIFiberTrackingPlugin::updateScalarRange()
{
	// Get the index of the selected scalar data set
	int scalarImageID = this->ui->AIDataCombo->currentIndex();

	if (scalarImageID < 0 || scalarImageID >= this->aiDataList.size())
		return;

	// Get the scalar volume data set
	data::DataSet * ds = this->aiDataList.at(scalarImageID);

	if (!ds)
		return;

	vtkImageData * scalarImage = ds->getVtkImageData();

	if (!scalarImage)
		return;

	// Update the image
	if (scalarImage->GetActualMemorySize() == 0)
	{
		scalarImage->Update();
		this->core()->data()->dataSetChanged(ds);
	}

	double range[2];

	// Get the range of the scalars
	scalarImage->GetScalarRange(range);

	// Copy the range to the scalar threshold spinboxes
	//this->ui->parametersMinAISpin->setMinimum(range[0]);
	//this->ui->parametersMinAISpin->setMaximum(range[1]);
	//this->ui->parametersMaxAISpin->setMinimum(range[0]);
	//this->ui->parametersMaxAISpin->setMaximum(range[1]);
}


//----------------------------[ dataSetAdded ]-----------------------------\\

void HARDIFiberTrackingPlugin::dataSetAdded(data::DataSet * ds)
{
	// Check if the data set exists
    Q_ASSERT(ds);
	
	// Add Spherical harmonics dataset image - For the moment we just accept spherical harmonics from HARDI
//	if (ds->getKind() == "spherical harmonics")
		if (ds->getKind() == "discrete sphere" || ds->getKind() == "spherical harmonics")
	{
		qDebug() <<  "added to hardi data set:" << ds->getKind() << endl;
		this->addHARDIDataSet(ds);
		return;
	}
	
	// AI Scalar image
	if (ds->getKind() == "scalar volume")
	{
		this->addAIDataSet(ds);
		//scalar volume not to be added to hardi combo
		//this->addHARDIDataSet(ds);
		return;
	}
	
	// Seed point set
	if (ds->getKind() == "seed points")
	{
		this->addSeedPoints(ds);
		return;
	}
}


//----------------------------[ addHARDIDataSet ]----------------------------\\

bool HARDIFiberTrackingPlugin::addHARDIDataSet(data::DataSet * ds)
{
	// Check if the data set contains an image data
	vtkImageData * image = ds->getVtkImageData();

	if (!image)
	{
		QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", "New HARDI data is NULL.", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("HARDI Fiber Tracking Plugin: New HARDI data is NULL.");
		return false;
	}

	// Check if the image contains point data
	vtkPointData * imagePD = image->GetPointData();
	if (!imagePD)
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "No PointData set for new HARDI data.", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: No PointData set for new DTI tensor data.");
		return false;
	}


	// Add the data set to the list of data sets
	this->HARDIDataList.append(ds);

	// Add the name of the data set to the combo box
	this->ui->HARDIDataCombo->addItem(ds->getName());

	return true;
}


//----------------------------[ addAIDataSet ]-----------------------------\\

bool HARDIFiberTrackingPlugin::addAIDataSet(data::DataSet * ds)
{
	// Check if the data set contains image data
	if (!ds->getVtkImageData())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "New AI volume data is NULL.", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: New AI volume data is NULL.");
		return false;
	}

	// Check if the image data contains point data
	if (!ds->getVtkImageData()->GetPointData())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "No PointData set for new AI volume.", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: No PointData set for new AI volume.");
		return false;
	}

	// Add the data set to the list of data sets
	this->aiDataList.append(ds);

	// Add the name of the data set to the combo box
	this->ui->AIDataCombo->addItem(ds->getName());

	return true;
}


//----------------------------[ addSeedPoints ]----------------------------\\

bool HARDIFiberTrackingPlugin::addSeedPoints(data::DataSet * ds)
{
	// Check if the data exists
	if (!ds->getVtkObject())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "No data for new seed point set.", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: No data for new seed point set.");
		return false;		
	}

	// Add the data set to the list of data sets
	this->seedList.append(ds);

	// Add the name of the data set to the GUI list
	this->ui->seedList->addItem(ds->getName());

	return true;
}


//----------------------------[ dataSetChanged ]---------------------------\\

void HARDIFiberTrackingPlugin::dataSetChanged(data::DataSet * ds)
{
	// Call the correct function depending on the data type
	if (ds->getKind() == "spherical harmonics")
	{
		this->changeHARDIDataSet(ds);
	}
	else if (ds->getKind() == "scalar volume")
	{
		this->changeAIDataSet(ds);
	}
	else if (ds->getKind() == "seed points")
	{
		this->changeSeedPoints(ds);
	}
}


//---------------------------[ changeHARDIDataSet ]--------------------------\\

void HARDIFiberTrackingPlugin::changeHARDIDataSet(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->HARDIDataList.contains(ds)))
	{
		return;
	}

	// Change the name of the data set in the GUI
	int index = this->HARDIDataList.indexOf(ds);
	this->ui->HARDIDataCombo->setItemText(index, ds->getName());
}


//---------------------------[ changeAIDataSet ]---------------------------\\

void HARDIFiberTrackingPlugin::changeAIDataSet(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->aiDataList.contains(ds)))
	{
		return;
	}

	// Change the name of the data set in the GUI
	int index = this->aiDataList.indexOf(ds);
	this->ui->AIDataCombo->setItemText(index, ds->getName());
}


//---------------------------[ changeSeedPoints ]--------------------------\\

void HARDIFiberTrackingPlugin::changeSeedPoints(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->seedList.contains(ds)))
	{
		return;
	}

	// Change the name of the data set in the GUI
	int index = this->seedList.indexOf(ds);
	this->ui->seedList->item(index)->setText(ds->getName());

	// For all output data sets that use the changed seed point data set as input, 
	// update the output. This will re-run the fiber tracking algorithm.

	for (QList<outputInfo>::iterator i = this->outputInfoList.begin(); i != this->outputInfoList.end(); ++i)
	{
		if ((*i).seed == ds)
		{
			if ((*i).output)
			{
				if ((*i).output->getVtkPolyData())
				{
					(*i).output->getVtkPolyData()->Update();
					this->core()->data()->dataSetChanged((*i).output);
				}
			}
		}
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void HARDIFiberTrackingPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Call the correct function depending on the data type
	if (ds->getKind() == "spherical harmonics" || ds->getKind() == "discrete sphere"  )
	{
		this->removeHARDIDataSet(ds);
	}
	else if (ds->getKind() == "scalar volume")
	{
		this->removeAIDataSet(ds);
	}
	else if (ds->getKind() == "seed points")
	{
		this->removeSeedPoints(ds);
	}
}


//---------------------------[ removeHARDIDataSet ]--------------------------\\

void HARDIFiberTrackingPlugin::removeHARDIDataSet(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->HARDIDataList.contains(ds)))
	{
		return;
	}

	// Delete item from the GUI and the data set list
	int index = this->HARDIDataList.indexOf(ds);
	this->ui->HARDIDataCombo->removeItem(index);
	this->HARDIDataList.removeAt(index);
}


//---------------------------[ removeAIDataSet ]---------------------------\\

void HARDIFiberTrackingPlugin::removeAIDataSet(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->aiDataList.contains(ds)))
	{
		return;
	}

	// Delete item from the GUI and the data set list
	int index = this->aiDataList.indexOf(ds);
	this->ui->AIDataCombo->removeItem(index);
	this->aiDataList.removeAt(index);
}


//---------------------------[ removeSeedPoints ]--------------------------\\

void HARDIFiberTrackingPlugin::removeSeedPoints(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->seedList.contains(ds)))
	{
		return;
	}

	// Delete item from the GUI and the data set list
	int index = this->seedList.indexOf(ds);
	this->ui->seedList->takeItem(index);
	this->seedList.removeAt(index);
}


//-----------------------------[ updateFibers ]----------------------------\\

void HARDIFiberTrackingPlugin::updateFibers()
{
	// Get the index of the selected HARDI data set
	int selectedHARDIData = this->ui->HARDIDataCombo->currentIndex();

	// Check if the index is correct
	if (selectedHARDIData < 0 || selectedHARDIData >= HARDIDataList.size())
	{
		QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", "HARDI data index out of range!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("HARDI Fiber Tracking Plugin: HARDI data index out of range!");
		return;
	}

	// Get the index of the selected AI data set
	int selectedAIData = this->ui->AIDataCombo->currentIndex();

	// Check if the index is correct
	if (selectedAIData < 0 || selectedAIData >= aiDataList.size())
	{
		QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", "AI data index out of range!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("HARDI Fiber Tracking Plugin: AI data index out of range!");
		return;
	}

	// Get pointers to the "vtkImageData" objects of the DTI tensors and AI scalars
	vtkImageData *	HARDIimageData = HARDIDataList.at(selectedHARDIData)->getVtkImageData();
	vtkImageData *  aiImageData =  aiDataList.at( selectedAIData)->getVtkImageData();

	// Update the AI data
	if (aiImageData->GetActualMemorySize() == 0)
	{
		aiImageData->Update();
		this->core()->data()->dataSetChanged(aiDataList.at( selectedAIData));
	}

	int HARDIDims[3];			// Dimensions of HARDI image
	int  aiDims[3];				// Dimensions of AI scalar image

	// Get the dimensions of the selected images
	HARDIimageData->GetDimensions(HARDIDims);
	 aiImageData->GetDimensions( aiDims);

	// Check if the dimensions are the same
	if (HARDIDims[0] != aiDims[0] || HARDIDims[1] != aiDims[1] || HARDIDims[2] != aiDims[2])
	{
		QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", "Dimensions of HARDI data and AI data do not match!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("HARDI Fiber Tracking Plugin: Dimensions of HARDI data and AI data do not match!");
		return;
	}

	double HARDISpacing[3];	// Spacing of HARDI image
	double  aiSpacing[3];	// Spacing of AI scalar image

	// Get the spacing of the selected images
	HARDIimageData->GetSpacing(HARDISpacing);
	aiImageData->GetSpacing( aiSpacing);

	// Check if the spacing values are the same
	if (HARDISpacing[0] != aiSpacing[0] || HARDISpacing[1] != aiSpacing[1] || HARDISpacing[2] != aiSpacing[2])
	{
		QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", "Spacing of DTI data and AI data do not match!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("HARDI Fiber Tracking Plugin: Spacing of DTI data and AI data do not match!");
		return;
	}
	
	switch ((TrackingMethod) this->ui->trackingMethodCombo->currentIndex())
	{
		case TM_Deterministic:
			if(HARDIDataList.at(selectedHARDIData)->getKind()=="discrete sphere")
			this->doDeterministicFiberTracking(HARDIimageData, aiImageData, 0);
			else if(HARDIDataList.at(selectedHARDIData)->getKind()=="spherical harmonics")
			this->doDeterministicFiberTracking(HARDIimageData, aiImageData, 1);
			break;

		default:
			break;
	}
}


//------------------------[ addFibersToDataManager ]-----------------------\\

void HARDIFiberTrackingPlugin::addFibersToDataManager(vtkPolyData * fibers, QString fiberName, TrackingMethod method, data::DataSet * seed)
{
	// Create a new output information object
	outputInfo newInfo;

	// Set available information
	newInfo.hardi    = HARDIDataList.at(this->ui->HARDIDataCombo->currentIndex());
	newInfo.method = method;
	newInfo.seed   = seed;

	// Index of matching data set in output list
	int outputIndex;

	// Try to get a matching output data set for overwriting
	bool overwrite = this->overwriteDataSet(&newInfo, &outputIndex);

	// If the output data does not contain any fibers (usually because the stopping
	// criteria are too strict), we remove existing data sets with the same name.

	if (fibers->GetLines()->GetNumberOfCells() == 0)
	{
		// Print log message
		this->core()->out()->logMessage("HARDI Fiber Tracking Plugin: No fibers in fiber set '" + fiberName + "'.");

		// If the data set should be overwritten, we delete it at this point
		if (overwrite)
		{
			// Get the existing data set pointer
			data::DataSet * oldDs = newInfo.output;

			// Remove the data set from the data manager
			this->core()->data()->removeDataSet(oldDs);

			// Remove the output information from the list
			this->outputInfoList.removeAt(outputIndex);
		}

		return;
	}

	// If "overwrite" is false, we create a new data set
	if (!overwrite)
	{
		// Create a new data set for the fibers
		data::DataSet * ds = new data::DataSet(fiberName, "fibers", (vtkObject *) fibers);

		// Copy the transformation matrix of the DTI image to the fibers
		ds->getAttributes()->copyTransformationMatrix(newInfo.hardi);

		// Add the data set to the data manager
		this->core()->data()->addDataSet(ds);

		// Store the output pointer and default name
		newInfo.output  = ds;
		newInfo.oldName = fiberName;

		// Add the output information to the list
		this->outputInfoList.append(newInfo);
	}
	// If "overwrite" is true, we modify an existing data set
	else
	{
		// Get the existing data set pointer
		data::DataSet * oldDs = newInfo.output;

		// Change the fibers stored in the data set
		oldDs->updateData((vtkObject *) fibers);

		// This condition fails if another plugin has changed the name of the data set,
		// or the user has done so manually. The idea here is that, for the user, this 
		// new name is likely preferable over the old one (which is why he changed it),
		// so we do not change the name back to the default one contained in "fiberName".

		if (oldDs->getName() == newInfo.oldName)
		{
			// If the name hasn't been changed externally, we now update it to the new
			// default name (which may be the same as the old data name).

			oldDs->setName(fiberName);

			// Store the new name in the output information
			newInfo.oldName = fiberName;
		}

		// update output information in the list
		this->outputInfoList.replace(outputIndex, newInfo);

		// Set "updatePipeline" to 1.0, to signal the visualization plugin that it should re-execute its pipeline.
		oldDs->getAttributes()->addAttribute("updatePipeline", 1.0);

		// Set "isVisible" to 1.0
		oldDs->getAttributes()->addAttribute("isVisible", 1.0);

		// Copy the transformation matrix of the DTI image to the fibers
		oldDs->getAttributes()->copyTransformationMatrix(newInfo.hardi);

		// Tell the data manager that the old data set has changed.
		this->core()->data()->dataSetChanged(oldDs);
	}
}


//---------------------------[ overwriteDataSet ]--------------------------\\

bool HARDIFiberTrackingPlugin::overwriteDataSet(outputInfo * newFiberInfo, int * outputIndex)
{
	// Never overwrite if the GUI checkbox isn't checked
	if (!this->ui->overwriteCheck->isChecked())
	{
		return false;
	}

	// Does the list of output information structures contain a match?
	bool matchFound = false;

	// Index of the matching information (if found)
	int index = 0;

	// Create an iterator for the list of output information
	QList<outputInfo>::iterator outputInfoIter;

	// Loop through all output data sets
	for ( outputInfoIter  = this->outputInfoList.begin();
		  outputInfoIter != this->outputInfoList.end();
		  ++outputInfoIter, ++index							)
	{
		// If HARDI image, seeding ROI and method (ROI or WVS) match, we have found a match!
		if ( ( (*outputInfoIter).hardi   == newFiberInfo->hardi)    &&
			 ( (*outputInfoIter).seed    == newFiberInfo->seed)   &&
			 ( (*outputInfoIter).method  == newFiberInfo->method) )
		{
			matchFound = true;
			break;
		}
	}

	// If we have found a match...
	if (matchFound)
	{
		// ...check if this data set (still) exists in the data manager.
		if (this->core()->data()->listAllDataSets().contains((*outputInfoIter).output))
		{
			// If so, store the data set pointer and the index
			newFiberInfo->output = (*outputInfoIter).output;
			newFiberInfo->oldName = (*outputInfoIter).oldName;
			(*outputIndex) = index;
		}
		// If the data set no longer exists...
		else
		{
			// ... we cannot overwrite anything...
			matchFound = false;

			// ...and the matching list item should be deleted
			this->outputInfoList.removeAt(index);
		}
	}

	return matchFound;
}


//----------------------------[ changeMethod ]-----------------------------\\

void HARDIFiberTrackingPlugin::changeMethod(int index)
{
	switch((TrackingMethod) index)
	{
		// Deterministic
		case TM_Deterministic:
			this->setupGUIForHARDIDeterministic();
			break;

		// Whole Volume Seeding
	/*	case TM_WVS:
			this->setupGUIForWVS();
			break;

		// Geodesic Fiber Tracking
		case TM_Geodesic:
			this->setupGUIForGeodesics();
			break;
*/
		default:
			this->core()->out()->showMessage("Unknown HARDI fiber tracking method!", "HARDI Fiber Tracking");
			this->changeMethod(0);
			break;
	}
}


//-----------------------------[ clearToolbox ]----------------------------\\

void HARDIFiberTrackingPlugin::clearToolbox()
{
	// Starting at the last page, loop through all toolbox pages except for the
	// first one. The first page, "Tracking Parameters", is relevant for all methods,
	// so will never be deleted.

	for (int i = this->ui->fiberTrackingToolbox->count() - 1; i > 0; --i)
	{
		// Delete the widget of the page, and remove the page itself
		delete this->ui->fiberTrackingToolbox->widget(i);
		this->ui->fiberTrackingToolbox->removeItem(i);
	}

	// Delete the GUI information for WVS
/*	if (this->wvsGUI)
	{
		delete this->wvsGUI;
		this->wvsGUI = NULL;
	}

	// Delete the GUI information for geodesic fiber-tracking
	if (this->geodesicGUI)
	{
		delete this->geodesicGUI;
		this->geodesicGUI = NULL;
	}*/
}


//--------------------------[ enableAllControls ]--------------------------\\

void HARDIFiberTrackingPlugin::enableAllControls()
{
	this->ui->seedList->setEnabled(true);
}

//---------------------------[ changePluginName ]--------------------------\\

void HARDIFiberTrackingPlugin::changePluginName()
{
	// Do nothing here, the default name is okay
}



Q_EXPORT_PLUGIN2(libbmia_HARDIFiberTrackingPlugin, bmia::HARDIFiberTrackingPlugin)

} // namespace bmia

