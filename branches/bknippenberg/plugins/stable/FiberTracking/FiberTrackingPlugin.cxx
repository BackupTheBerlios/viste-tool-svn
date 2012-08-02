/*
 * FiberTrackingPlugin.cxx
 *
 * 2010-09-13	Evert van Aart
 * - First Version.
 *
 * 2010-09-15	Evert van Aart
 * - Added error message boxes.
 * - Added additional checks for input data.
 * - Improved adding output to data manager.
 *
 * 2010-09-30	Evert van Aart
 * - When recomputing a fiber set, the plugin will now change the existing 
 *   data set, instead of deleting and re-adding it. This way, the visualization
 *   options of the data set (such as shape, color, etcetera) will not be reset.
 *
 * 2010-11-09	Evert van Aart
 * - Added support for showing/hiding fibers.
 *
 * 2010-11-23	Evert van Aart
 * - Added "Overwrite existing fibers" checkbox to the GUI.
 * - If this box is checked, the plugin checks if a fiber set with the same DTI image,
 *   ROI, and tracking method (ROI or WVS) has been added to the data manager at some 
 *   point. If so, and if this data set still exists, the fibers in that data set are
 *   overwritten. If not, a new data set is generated. 
 * - This should work better with the new data management approach, which no longer
 *   requires unique data set names.
 *
 * 2010-12-10	Evert van Aart
 * - Implemented "dataSetChanged" and "dataSetRemoved".
 *
 * 2011-01-24	Evert van Aart
 * - Added support for transformation matrices.
 *
 * 2011-02-09	Evert van Aart
 * - Version 1.0.0.
 * - Added support for maximum scalar threshold values.
 * - In the GUI, changed references to "AI" values to "scalar" values. The idea
 *   is that any type of scalar data can be used as stopping criterium, not just
 *   AI values. This essentially enables the use of masking volumes.
 *
 * 2011-03-14	Evert van Aart
 * - Version 1.0.1.
 * - Fixed a bug in which it was not always detected when a fiber moved to a new
 *   voxel. Because of this, the fiber tracking process kept using the data of the
 *   old cell, resulting in fibers that kept going in areas of low anisotropy.
 *
 * 2011-03-16	Evert van Aart
 * - Version 1.0.2.
 * - Fixed a bug that could cause crashes if a fiber left the volume. 
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.3.
 * - Maximum AI value no longer automatically snaps to the scalar range maximum
 *   when changing the AI image used.
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.0.4.
 * - Properly update the fibers when their seed points change.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.5.
 * - Improved progress reporting.
 * - Slightly improved speed for Whole Volume Seeding.
 * - GUI now enables/disables the WVS controls when WVS is checked/unchecked.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.6.
 * - Improved attribute handling.
 *
 * 2011-06-01	Evert van Aart
 * - Version 1.1.0.
 * - Added geodesic fiber-tracking.
 * - GUI now only show controls for the selected fiber tracking method.
 * - Moved some functions to separate files to avoid one huge file.
 *
 * 2011-06-06	Evert van Aart
 * - Version 1.1.1.
 * - Fixed a bug in WVS that allowed fibers shorter than the minimum fiber length
 *   to still be added to the output.
 *
 * 2011-07-07	Evert van Aart
 * - Version 1.2.0.
 * - Added CUDA support.
 *
 */



/** Includes */

#include "FiberTrackingPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberTrackingPlugin::FiberTrackingPlugin() : Plugin("Fiber Tracking")
{
	// Warn the user if CUDA is not installed. We do not interrupt initialization
	// of the plugin; if the user tries to perform geodesic fiber-tracking, we simply
	// do nothing (and display another warning).

	if (this->isCUDASupported() == false)
	{
		QMessageBox::warning(NULL, "Fiber Tracking", "Could not find CUDA. Please use the regular version of this plugin.");
	}

	// Change the name of the plugin if necessary
	this->changePluginName();

	// Create a new Qt widget
    this->widget = new QWidget();

	// Create a new GUI form
    this->ui = new Ui::FiberTrackingForm();

	// Setup the GUI
    this->ui->setupUi(this->widget);

    // Link the "Update" button to the "updateFibers" function
	connect(this->ui->updateButton, SIGNAL(clicked()), this, SLOT(updateFibers()));

	// Connect GUI elements
	connect(this->ui->AIDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(updateScalarRange()));
	connect(this->ui->trackingMethodCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(changeMethod(int)));

	// Initialize optional GUI components to NULL
	this->wvsGUI		= NULL;
	this->geodesicGUI	= NULL;

	// Clear the toolbox to get rid of the dummy page
	this->clearToolbox();
}


//-----------------------------[ Destructor ]------------------------------\\

FiberTrackingPlugin::~FiberTrackingPlugin()
{
	// Remove the Qt widget
	delete this->widget; 
	this->widget = NULL;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * FiberTrackingPlugin::getGUI()
{
	// Return the Qt widget
    return this->widget;
}


//--------------------------[ updateScalarRange ]--------------------------\\

void FiberTrackingPlugin::updateScalarRange()
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
	this->ui->parametersMinAISpin->setMinimum(range[0]);
	this->ui->parametersMinAISpin->setMaximum(range[1]);
	this->ui->parametersMaxAISpin->setMinimum(range[0]);
	this->ui->parametersMaxAISpin->setMaximum(range[1]);
}


//----------------------------[ dataSetAdded ]-----------------------------\\

void FiberTrackingPlugin::dataSetAdded(data::DataSet * ds)
{
	// Check if the data set exists
    Q_ASSERT(ds);
	
	// DTI Tensor image
	if (ds->getKind() == "DTI")
	{
		this->addDTIDataSet(ds);
		return;
	}
	
	// AI Scalar image
	if (ds->getKind() == "scalar volume")
	{
		this->addAIDataSet(ds);
		return;
	}
	
	// Seed point set
	if (ds->getKind() == "seed points")
	{
		this->addSeedPoints(ds);
		return;
	}
}


//----------------------------[ addDTIDataSet ]----------------------------\\

bool FiberTrackingPlugin::addDTIDataSet(data::DataSet * ds)
{
	// Check if the data set contains image data
	if (!ds->getVtkImageData())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "New DTI tensor data is NULL.", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: New DTI tensor data is NULL.");
		return false;
	}

	// Check if the image data contains point data
	if (!ds->getVtkImageData()->GetPointData())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "No PointData set for new DTI tensor data.", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: No PointData set for new DTI tensor data.");
		return false;
	}

	// Add the data set to the list of data sets
	this->dtiDataList.append(ds);

	// Add the name of the data set to the combo box
	this->ui->DTIDataCombo->addItem(ds->getName());

	return true;
}


//----------------------------[ addAIDataSet ]-----------------------------\\

bool FiberTrackingPlugin::addAIDataSet(data::DataSet * ds)
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

bool FiberTrackingPlugin::addSeedPoints(data::DataSet * ds)
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

void FiberTrackingPlugin::dataSetChanged(data::DataSet * ds)
{
	// Call the correct function depending on the data type
	if (ds->getKind() == "DTI")
	{
		this->changeDTIDataSet(ds);
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


//---------------------------[ changeDTIDataSet ]--------------------------\\

void FiberTrackingPlugin::changeDTIDataSet(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->dtiDataList.contains(ds)))
	{
		return;
	}

	// Change the name of the data set in the GUI
	int index = this->dtiDataList.indexOf(ds);
	this->ui->DTIDataCombo->setItemText(index, ds->getName());
}


//---------------------------[ changeAIDataSet ]---------------------------\\

void FiberTrackingPlugin::changeAIDataSet(data::DataSet * ds)
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

void FiberTrackingPlugin::changeSeedPoints(data::DataSet * ds)
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

void FiberTrackingPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Call the correct function depending on the data type
	if (ds->getKind() == "DTI")
	{
		this->removeDTIDataSet(ds);
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


//---------------------------[ removeDTIDataSet ]--------------------------\\

void FiberTrackingPlugin::removeDTIDataSet(data::DataSet * ds)
{
	// Check if the data set has been added to this plugin
	if (!(this->dtiDataList.contains(ds)))
	{
		return;
	}

	// Delete item from the GUI and the data set list
	int index = this->dtiDataList.indexOf(ds);
	this->ui->DTIDataCombo->removeItem(index);
	this->dtiDataList.removeAt(index);
}


//---------------------------[ removeAIDataSet ]---------------------------\\

void FiberTrackingPlugin::removeAIDataSet(data::DataSet * ds)
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

void FiberTrackingPlugin::removeSeedPoints(data::DataSet * ds)
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

void FiberTrackingPlugin::updateFibers()
{
	// Get the index of the selected DTI data set
	int selectedDTIData = this->ui->DTIDataCombo->currentIndex();

	// Check if the index is correct
	if (selectedDTIData < 0 || selectedDTIData >= dtiDataList.size())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "DTI data index out of range!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: DTI data index out of range!");
		return;
	}

	// Get the index of the selected AI data set
	int selectedAIData = this->ui->AIDataCombo->currentIndex();

	// Check if the index is correct
	if (selectedAIData < 0 || selectedAIData >= aiDataList.size())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "AI data index out of range!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: AI data index out of range!");
		return;
	}

	// Get pointers to the "vtkImageData" objects of the DTI tensors and AI scalars
	vtkImageData * dtiImageData = dtiDataList.at(selectedDTIData)->getVtkImageData();
	vtkImageData *  aiImageData =  aiDataList.at( selectedAIData)->getVtkImageData();

	// Update the AI data
	if (aiImageData->GetActualMemorySize() == 0)
	{
		aiImageData->Update();
		this->core()->data()->dataSetChanged(aiDataList.at( selectedAIData));
	}

	int dtiDims[3];			// Dimensions of DTI tensor image
	int  aiDims[3];			// Dimensions of AI scalar image

	// Get the dimensions of the selected images
	dtiImageData->GetDimensions(dtiDims);
	 aiImageData->GetDimensions( aiDims);

	// Check if the dimensions are the same
	if (dtiDims[0] != aiDims[0] || dtiDims[1] != aiDims[1] || dtiDims[2] != aiDims[2])
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "Dimensions of DTI data and AI data do not match!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: Dimensions of DTI data and AI data do not match!");
		return;
	}

	double dtiSpacing[3];	// Spacing of DTI tensor image
	double  aiSpacing[3];	// Spacing of AI scalar image

	// Get the spacing of the selected images
	dtiImageData->GetSpacing(dtiSpacing);
	 aiImageData->GetSpacing( aiSpacing);

	// Check if the spacing values are the same
	if (dtiSpacing[0] != aiSpacing[0] || dtiSpacing[1] != aiSpacing[1] || dtiSpacing[2] != aiSpacing[2])
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "Spacing of DTI data and AI data do not match!", 
								QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: Spacing of DTI data and AI data do not match!");
		return;
	}

	switch ((TrackingMethod) this->ui->trackingMethodCombo->currentIndex())
	{
		case TM_Streamlines:
			this->doStreamlineFiberTracking(dtiImageData, aiImageData);
			break;

		case TM_WVS:
			this->doWVSFiberTracking(dtiImageData, aiImageData);
			break;

		case TM_Geodesic:
			this->doGeodesicFiberTracking(dtiImageData, aiImageData);
			break;

		default:
			break;
	}
}


//------------------------[ addFibersToDataManager ]-----------------------\\

void FiberTrackingPlugin::addFibersToDataManager(vtkPolyData * fibers, QString fiberName, TrackingMethod method, data::DataSet * seed)
{
	// Create a new output information object
	outputInfo newInfo;

	// Set available information
	newInfo.dti    = dtiDataList.at(this->ui->DTIDataCombo->currentIndex());
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
		this->core()->out()->logMessage("Fiber Tracking Plugin: No fibers in fiber set '" + fiberName + "'.");

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
		ds->getAttributes()->copyTransformationMatrix(newInfo.dti);

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
		oldDs->getAttributes()->copyTransformationMatrix(newInfo.dti);

		// Tell the data manager that the old data set has changed.
		this->core()->data()->dataSetChanged(oldDs);
	}
}


//---------------------------[ overwriteDataSet ]--------------------------\\

bool FiberTrackingPlugin::overwriteDataSet(outputInfo * newFiberInfo, int * outputIndex)
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
		// If DTI image, seeding ROI and method (ROI or WVS) match, we have found a match!
		if ( ( (*outputInfoIter).dti     == newFiberInfo->dti)    &&
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

void FiberTrackingPlugin::changeMethod(int index)
{
	switch((TrackingMethod) index)
	{
		// Streamlines
		case TM_Streamlines:
			this->setupGUIForStreamlines();
			break;

		// Whole Volume Seeding
		case TM_WVS:
			this->setupGUIForWVS();
			break;

		// Geodesic Fiber Tracking
		case TM_Geodesic:
			this->setupGUIForGeodesics();
			break;

		default:
			this->core()->out()->showMessage("Unknown fiber tracking method!", "Fiber Tracking");
			this->changeMethod(0);
			break;
	}
}


//-----------------------------[ clearToolbox ]----------------------------\\

void FiberTrackingPlugin::clearToolbox()
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
	if (this->wvsGUI)
	{
		delete this->wvsGUI;
		this->wvsGUI = NULL;
	}

	// Delete the GUI information for geodesic fiber-tracking
	if (this->geodesicGUI)
	{
		delete this->geodesicGUI;
		this->geodesicGUI = NULL;
	}
}


//--------------------------[ enableAllControls ]--------------------------\\

void FiberTrackingPlugin::enableAllControls()
{
	// Enable the seed list (disabled for WVS)
	this->ui->seedList->setEnabled(true);
}


} // namespace bmia

