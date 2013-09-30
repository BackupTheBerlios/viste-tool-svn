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
 * SeedingPlugin.cxx
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.0.0.
 * - First version. Currently, the GUI controls include support for 2D ROIs, 
 *   scalar volumes, and fibers, but volume seeding has not yet been implemented.
 *
 * 2011-05-10	Evert van Aart
 * - Version 1.1.0.
 * - Added support for volume seeding.
 *
 */


/** Includes */

#include "SeedingPlugin.h"
#include "vtk2DRoiToSeedFilter.h"
#include "vtkPolyDataToSeedPoints.h"
#include "vtkScalarVolumeToSeedPoints.h"
#include "Helpers/vtkStreamlineToSimplifiedStreamline.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

SeedingPlugin::SeedingPlugin() : Plugin("Seeding")
{
	// Create the GUI
	this->widget = new QWidget();
	this->ui = new Ui::SeedingForm();
	this->ui->setupUi(this->widget);

	// Setup different components of the GUI for the initial situation (no data)
	this->setupGUIForROIs();
	this->setupGUIForFibers();
	this->setupGUIForVolumes();

	// Connect GUI controls
	this->connectControlsForROIs(true);
	this->connectControlsForFibers(true);
	this->connectControlsForVolumes(true);
}


//---------------------------------[ init ]--------------------------------\\

void SeedingPlugin::init()
{

}


//------------------------------[ Destructor ]-----------------------------\\

SeedingPlugin::~SeedingPlugin()
{
	// Disconnect the controls
	this->connectControlsForFibers(false);
	this->connectControlsForROIs(false);
	this->connectControlsForVolumes(false);

	// Remove the Qt widget
	delete this->widget; 
	this->widget = NULL;

	// Remove filters and output data sets for all ROIs
	for (QList<roiSeedInfo>::iterator i = this->roiInfoList.begin(); i != this->roiInfoList.end(); ++i)
	{
		if ((*i).filter)
			(*i).filter->Delete();
		
		if ((*i).outDS)
			this->core()->data()->removeDataSet((*i).outDS);
	}

	// Clear the list of ROIs and voxel volumes
	this->roiInfoList.clear();
	this->roiVoxelImages.clear();

	// Remove filters and output data sets for all fibers
	for (QList<fiberSeedInfo>::iterator i = this->fiberInfoList.begin(); i != this->fiberInfoList.end(); ++i)
	{
		if ((*i).simplifyFilter)
			(*i).simplifyFilter->Delete();

		if ((*i).seedFilter)
			(*i).seedFilter->Delete();

		if ((*i).outDS)
			this->core()->data()->removeDataSet((*i).outDS);
	}

	// Clear the fiber list
	this->fiberInfoList.clear();

	// Remove filters and output data sets for all volumes
	for (QList<volumeSeedInfo>::iterator i = this->volumeInfoList.begin(); i != this->volumeInfoList.end(); ++i)
	{

			if ((*i).seedFilter)
				(*i).seedFilter->Delete();

		if ((*i).outDS)
			this->core()->data()->removeDataSet((*i).outDS);
	}

	// Clear the fiber list
	this->volumeInfoList.clear();
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * SeedingPlugin::getGUI()
{
	return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void SeedingPlugin::dataSetAdded(data::DataSet * ds)
{
	// Regions of Interest
	if(ds->getKind() == "regionOfInterest")
	{
		// check if the data set contains polydata
		if (!(ds->getVtkPolyData()))
			return;

		// Create a new ROI information structure with default settings
		roiSeedInfo newInfo;
		newInfo.distance	= 1.0;
		newInfo.imageID		= (this->ui->roiVoxelsCombo->count() > 0) ? 0 : -1;
		newInfo.inDS		= ds;
		newInfo.outDS		= NULL;
		newInfo.seedType	= RST_Distance;
		newInfo.filter		= vtk2DRoiToSeedFilter::New();

		// Add the ROI to the list and to the GUI
		this->roiInfoList.append(newInfo);
		this->ui->roiDataCombo->addItem(ds->getName());

		// Create seed points for these default settings
		this->updateROISeeds(this->roiInfoList[this->roiInfoList.size() - 1]);

		return;
	}

	// VTK images
	if (ds->getVtkImageData())
	{
		// Add the data set to the list and to the GUI
		this->ui->roiVoxelsCombo->addItem(ds->getName());
		this->roiVoxelImages.append(ds->getVtkImageData());

		// If this is the first image...
		if (this->ui->roiVoxelsCombo->count() == 1)
		{
			// ...select it for all existing ROIs. We do not need to update the 
			// seed points, because by definition, none of the ROIs can have voxel
			// seeding at this time (since there were no voxel images until now).

			for (QList<roiSeedInfo>::iterator i = this->roiInfoList.begin(); i != this->roiInfoList.end(); ++i)
			{
				(*i).imageID = 0;
			}

			// Update the GUI for the selected ROI
			this->setupGUIForROIs();
		}

		// Note: No return here, since this data set can also be used for volume seeding
	}

	// Fibers
	if(ds->getKind() == "fibers")
	{
		// check if the data set contains polydata
		if (!(ds->getVtkPolyData()))
			return;

		// Create a new ROI information structure with default settings
		fiberSeedInfo newInfo;
		newInfo.distance		= 1.0;
		newInfo.enable			= false;
		newInfo.doFixedDistance = true;
		newInfo.inDS			= ds;
		newInfo.outDS			= NULL;
		newInfo.simplifyFilter	= vtkStreamlineToSimplifiedStreamline::New();
		newInfo.seedFilter		= vtkPolyDataToSeedPoints::New();

		// Add the fibers to the list and to the GUI
		this->fiberInfoList.append(newInfo);
		this->ui->fiberDataCombo->addItem(ds->getName());

		// Create seed points for these default settings
		this->updateFiberSeeds(this->fiberInfoList[this->fiberInfoList.size() - 1]);

		return;
	}

	// Scalar volumes
	if(ds->getKind() == "scalar volume")
	{
		// Check if the data contains a scalar array
		if (!(ds->getVtkImageData()))
			return;

		// Create a new ROI information structure with default settings
		volumeSeedInfo newInfo;
		newInfo.enable			= false;
		newInfo.minValue		= 0.0;
		newInfo.maxValue		= 1.0;
		newInfo.inDS			= ds;
		newInfo.outDS			= NULL;
		newInfo.initializedRange = false;
		newInfo.seedFilter		= vtkScalarVolumeToSeedPoints::New();

		newInfo.seedFilter->setMinThreshold(0.0);
		newInfo.seedFilter->setMaxThreshold(1.0);

		// Add the volumes to the list and to the GUI
		this->volumeInfoList.append(newInfo);
		this->ui->volumeDataCombo->addItem(ds->getName());

		// Create seed points for these default settings
		this->updateVolumeSeeds(this->volumeInfoList[this->volumeInfoList.size() - 1]);

		return;
	}
	

}


//----------------------------[ dataSetChanged ]---------------------------\\

void SeedingPlugin::dataSetChanged(data::DataSet * ds)
{
	// Regions of Interest
	if(ds->getKind() == "regionOfInterest")
	{
		// If the data set no longer contains polydata, it is dead to us
		if (!(ds->getVtkPolyData()))
			this->dataSetRemoved(ds);

		// Find the ROI information object(s) that use the changed data set, update
		// its seed points, and change its name in the GUI.

		int pos = 0;
		for (QList<roiSeedInfo>::iterator i = this->roiInfoList.begin(); i != this->roiInfoList.end(); ++i, ++pos)
		{
			if ((*i).inDS == ds)
			{
				this->updateROISeeds(*i);
				this->ui->roiDataCombo->setItemText(pos, ds->getName());
			}
		}

		return;
	}

	// VTK images
	if (ds->getVtkImageData() && this->roiVoxelImages.contains(ds->getVtkImageData()))
	{
		// Update the data set name in the GUI
		int imageID = this->roiVoxelImages.indexOf(ds->getVtkImageData());
		this->ui->roiVoxelsCombo->setItemText(imageID, ds->getName());

		// Update all seed point sets that use this image
		for (QList<roiSeedInfo>::iterator i = this->roiInfoList.begin(); i != this->roiInfoList.end(); ++i)
		{
			if ((*i).imageID == imageID && (*i).seedType == RST_Voxel)
				this->updateROISeeds(*i);
		}
	}

	// Fibers
	if(ds->getKind() == "fibers")
	{
		if (!(ds->getVtkPolyData()))
			this->dataSetRemoved(ds);

		// Find the fiber seeds that use this input data set, and update it

		int pos = 0;
		for (QList<fiberSeedInfo>::iterator i = this->fiberInfoList.begin(); i != this->fiberInfoList.end(); ++i, ++pos)
		{
			if ((*i).inDS == ds)
			{
				this->updateFiberSeeds(*i);
				this->ui->fiberDataCombo->setItemText(pos, ds->getName());
			}
		}

		return;
	}

	// Scalar volume
	if(ds->getKind() == "scalar volume")
	{
		if (!(ds->getVtkImageData()))
			this->dataSetRemoved(ds);

		// Find the volume seeds that use this input data set, and update it

		int pos = 0;
		for (QList<volumeSeedInfo>::iterator i = this->volumeInfoList.begin(); i != this->volumeInfoList.end(); ++i, ++pos)
		{
			if ((*i).inDS == ds)
			{
				this->updateVolumeSeeds(*i);
				this->ui->volumeDataCombo->setItemText(pos, ds->getName());
			}
		}

		return;
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void SeedingPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Regions of Interest
	if(ds->getKind() == "regionOfInterest")
	{
		// Find seed point sets that use this ROI data set
		int pos = 0;
		for (QList<roiSeedInfo>::iterator i = this->roiInfoList.begin(); i != this->roiInfoList.end(); ++i, ++pos)
		{
			if ((*i).inDS == ds)
			{
				// Delete the output seed point data set
				if ((*i).outDS)
				{
					this->core()->data()->removeDataSet((*i).outDS);
				}

				// Delete the filter
				if ((*i).filter)
				{
					(*i).filter->Delete();
				}

				// Remove information from the GUI and from the list
				this->roiInfoList.removeAt(pos);
				this->ui->roiDataCombo->removeItem(pos);
				return;
			}
		}

		return;
	}

	// VTK images
	if (ds->getVtkImageData() && this->roiVoxelImages.contains(ds->getVtkImageData()))
	{
		// Get the index of the image
		int dsIndex = this->roiVoxelImages.indexOf(ds->getVtkImageData());

		// Find seed point sets that use this image
		int pos = 0;
		for (QList<roiSeedInfo>::iterator i = this->roiInfoList.begin(); i != this->roiInfoList.end(); ++i, ++pos)
		{
			// If the seed points uses the removed image...
			if ((*i).imageID == dsIndex)
			{
				// ...deselect all images...
				(*i).imageID = -1;

				// ...and switch to distance seeding
				 if ((*i).seedType == RST_Voxel)
				 {
					 (*i).seedType = RST_NoSeeding;
					 this->setupGUIForROIs();
					 this->updateROISeeds(*i);
				 }
			}

			// If the image index is larger than the index of the removed image,
			// decrement the index.

			else if ((*i).imageID > dsIndex)
			{
				(*i).imageID--;
			}

			// Remove the image from the list and from the GUI
			this->roiVoxelImages.removeAt(dsIndex);
			this->ui->roiVoxelsCombo->removeItem(dsIndex);
		}
	}

	// Fibers
	if(ds->getKind() == "fibers")
	{
		// Find seed point sets that use this fiber data set
		int pos = 0;
		for (QList<fiberSeedInfo>::iterator i = this->fiberInfoList.begin(); i != this->fiberInfoList.end(); ++i, ++pos)
		{
			if ((*i).inDS == ds)
			{
				// Delete the output seed point data set
				if ((*i).outDS)
					this->core()->data()->removeDataSet((*i).outDS);

				// Delete the simplification filter
				if ((*i).simplifyFilter)
					(*i).simplifyFilter->Delete();

				// Delete the seed filter
				if ((*i).seedFilter)
					(*i).seedFilter->Delete();

				// Remove information from the GUI and from the list
				this->fiberInfoList.removeAt(pos);
				this->ui->fiberDataCombo->removeItem(pos);
				return;
			}
		}

		return;
	}

	// Scalar volumes
	if(ds->getKind() == "scalar volume")
	{
		// Find seed point sets that use this volume data set
		int pos = 0;
		for (QList<volumeSeedInfo>::iterator i = this->volumeInfoList.begin(); i != this->volumeInfoList.end(); ++i, ++pos)
		{
			if ((*i).inDS == ds)
			{
				// Delete the output seed point data set
				if ((*i).outDS)
					this->core()->data()->removeDataSet((*i).outDS);

				// Delete the seed filter
				if ((*i).seedFilter)
					(*i).seedFilter->Delete();

				// Remove information from the GUI and from the list
				this->volumeInfoList.removeAt(pos);
				this->ui->volumeDataCombo->removeItem(pos);
				return;
			}
		}

		return;
	}
}


//---------------------------[ setupGUIForROIs ]---------------------------\\

void SeedingPlugin::setupGUIForROIs()
{
	// Check if we can enable the ROI seeding group
	bool enable = !(this->ui->roiDataCombo->count() <= 0 || 
					this->ui->roiDataCombo->currentIndex() < 0 ||
					this->ui->roiDataCombo->currentIndex() >= this->roiInfoList.size() );
	
	// Enable or disable the controls
	this->ui->roiNoSeedingRadio->setEnabled(enable);
	this->ui->roiDistanceRadio->setEnabled(enable);
	this->ui->roiDistanceSpin->setEnabled(enable);
	this->ui->roiVoxelsRadio->setEnabled(enable);
	this->ui->roiVoxelsCombo->setEnabled(enable);
	
	if (!enable)
		return;

	// Disconnect all relevant controls for now
	this->connectControlsForROIs(false);

	// Get the information of the current ROI
	roiSeedInfo currentROIInfo = this->roiInfoList[this->ui->roiDataCombo->currentIndex()];

	// Disable the "Voxels" radio button if there are no images
	if (this->ui->roiVoxelsCombo->count() <= 0)
		this->ui->roiVoxelsRadio->setEnabled(false);

	// Set the distance and select the voxel image
	this->ui->roiDistanceSpin->setValue(currentROIInfo.distance);
	this->ui->roiVoxelsCombo->setCurrentIndex(currentROIInfo.imageID);

	// Check the right radio button, disable/enable controls
	switch (currentROIInfo.seedType)
	{
		case RST_NoSeeding:
			this->ui->roiNoSeedingRadio->setChecked(true);
			this->ui->roiDistanceSpin->setEnabled(false);
			this->ui->roiVoxelsCombo->setEnabled(false);
			break;

		case RST_Distance:
			this->ui->roiDistanceRadio->setChecked(true);
			this->ui->roiDistanceSpin->setEnabled(true);
			this->ui->roiVoxelsCombo->setEnabled(false);
			break;

		case RST_Voxel:
			this->ui->roiVoxelsRadio->setChecked(true);
			this->ui->roiDistanceSpin->setEnabled(false);
			this->ui->roiVoxelsCombo->setEnabled(true);
			break;

		default:
			Q_ASSERT(false);
	}

	// Reconnect the controls
	this->connectControlsForROIs(true);
}


//---------------------------[ setupGUIForFibers ]-------------------------\\

void SeedingPlugin::setupGUIForFibers()
{
	// Check if we can enable the fiber seeding group
	bool enable = !(this->ui->fiberDataCombo->count() <= 0 || 
					this->ui->fiberDataCombo->currentIndex() < 0 ||
					this->ui->fiberDataCombo->currentIndex() >= this->fiberInfoList.size() );

	// Enable or disable the controls
	this->ui->fiberNoSeedingRadio->setEnabled(enable);
	this->ui->fiberSeedingRadio->setEnabled(enable);
	this->ui->fiberSimplifyCheck->setEnabled(enable);
	this->ui->fiberSimplifySpin->setEnabled(enable);

	if (!enable)
		return;

	// Disconnect all relevant controls for now
	this->connectControlsForFibers(false);

	// Get the information of the current fiber
	fiberSeedInfo currentFiberInfo = this->fiberInfoList[this->ui->fiberDataCombo->currentIndex()];

	// Set the seed distance and check/uncheck the fixed distance checkbox
	this->ui->fiberSimplifySpin->setValue(currentFiberInfo.distance);
	this->ui->fiberSimplifyCheck->setChecked(currentFiberInfo.doFixedDistance);

	// Select the right radio button
	this->ui->fiberNoSeedingRadio->setChecked(!(currentFiberInfo.enable));
	this->ui->fiberSeedingRadio->setChecked(currentFiberInfo.enable);

	// Enable or disable controls
	enable = this->ui->fiberSeedingRadio->isChecked();
	this->ui->fiberSimplifyCheck->setEnabled(enable);
	enable &= currentFiberInfo.doFixedDistance;
	this->ui->fiberSimplifySpin->setEnabled(enable);

	// Reconnect the controls
	this->connectControlsForFibers(true);
}


//--------------------------[ setupGUIForVolumes ]-------------------------\\

void SeedingPlugin::setupGUIForVolumes()
{
	// Check if we can enable the volume seeding group
	bool enable = !(this->ui->volumeDataCombo->count() <= 0 || 
					this->ui->volumeDataCombo->currentIndex() < 0 ||
					this->ui->volumeDataCombo->currentIndex() >= this->volumeInfoList.size() );

	// Enable or disable the controls
	this->ui->volumeNoSeedingRadio->setEnabled(enable);
	this->ui->volumeSeedingRadio->setEnabled(enable);
	this->ui->volumeRangeLabelA->setEnabled(enable);
	this->ui->volumeRangeLabelB->setEnabled(enable);
	this->ui->volumeRangeMinSpin->setEnabled(enable);
	this->ui->volumeRangeMaxSpin->setEnabled(enable);

	if (!enable)
		return;

	// Disconnect all relevant controls for now
	this->connectControlsForVolumes(false);

	// Get the information of the current volume
	volumeSeedInfo currentVolumeInfo = this->volumeInfoList[this->ui->volumeDataCombo->currentIndex()];

	// Select the right radio button
	this->ui->volumeNoSeedingRadio->setChecked(!(currentVolumeInfo.enable));
	this->ui->volumeSeedingRadio->setChecked(currentVolumeInfo.enable);

	// Set the minimum and maximum
	this->ui->volumeRangeMinSpin->setValue(currentVolumeInfo.minValue);
	this->ui->volumeRangeMaxSpin->setValue(currentVolumeInfo.maxValue);

	// Enable or disable controls
	enable = this->ui->volumeSeedingRadio->isChecked();
	this->ui->volumeRangeMinSpin->setEnabled(enable);
	this->ui->volumeRangeMaxSpin->setEnabled(enable);

	// Reconnect the controls
	this->connectControlsForVolumes(true);
}



//--------------------------[ ROISettingsChanged ]-------------------------\\

void SeedingPlugin::ROISettingsChanged()
{
	// Get the index of the current ROI
	int roiIndex = this->ui->roiDataCombo->currentIndex();

	if (roiIndex < 0 || roiIndex >= this->roiInfoList.size())
		return;

	// Store the distance and the voxel image index
	this->roiInfoList[roiIndex].distance = this->ui->roiDistanceSpin->value();
	this->roiInfoList[roiIndex].imageID = this->ui->roiVoxelsCombo->currentIndex();
	
	// Set the seeding type
	if (this->ui->roiNoSeedingRadio->isChecked())	this->roiInfoList[roiIndex].seedType = RST_NoSeeding;
	if (this->ui->roiDistanceRadio->isChecked())	this->roiInfoList[roiIndex].seedType = RST_Distance;
	if (this->ui->roiVoxelsRadio->isChecked())		this->roiInfoList[roiIndex].seedType = RST_Voxel;

	// Update the GUI to disable/enable controls based on the new setting
	this->setupGUIForROIs();

	// Update the seed points for this ROI
	this->updateROISeeds(this->roiInfoList[roiIndex]);
}


//-------------------------[ FiberSettingsChanged ]------------------------\\

void SeedingPlugin::FiberSettingsChanged()
{
	// Get the index of the current fibers
	int fiberIndex = this->ui->fiberDataCombo->currentIndex();

	if (fiberIndex < 0 || fiberIndex >= this->fiberInfoList.size())
		return;

	// Store the distance and whether or not to use a fixed distance
	this->fiberInfoList[fiberIndex].distance = this->ui->fiberSimplifySpin->value();
	this->fiberInfoList[fiberIndex].doFixedDistance = this->ui->fiberSimplifyCheck->isChecked();

	// Enable or disable seeding
	if (this->ui->fiberNoSeedingRadio->isChecked())	this->fiberInfoList[fiberIndex].enable = false;
	if (this->ui->fiberSeedingRadio->isChecked())	this->fiberInfoList[fiberIndex].enable = true;

	// Update the GUI to disable/enable controls based on the new setting
	this->setupGUIForFibers();

	// Update the seed points for this fiber set
	this->updateFiberSeeds(this->fiberInfoList[fiberIndex]);
}


//------------------------[ VolumeSettingsChanged ]------------------------\\

void SeedingPlugin::VolumeSettingsChanged()
{
	// Get the index of the current volume
	int volumeIndex = this->ui->volumeDataCombo->currentIndex();

	if (volumeIndex < 0 || volumeIndex >= this->volumeInfoList.size())
		return;

	// Get the minimum and maximum value
	double val = this->ui->volumeRangeMinSpin->value();
	this->volumeInfoList[volumeIndex].minValue = this->ui->volumeRangeMinSpin->value();
	val = this->volumeInfoList[volumeIndex].minValue;
	this->volumeInfoList[volumeIndex].maxValue = this->ui->volumeRangeMaxSpin->value();

	// Enable or disable seeding
	if (this->ui->volumeNoSeedingRadio->isChecked())	this->volumeInfoList[volumeIndex].enable = false;
	if (this->ui->volumeSeedingRadio->isChecked())		this->volumeInfoList[volumeIndex].enable = true;

	// Get the scalar volume image
	vtkImageData * image = this->volumeInfoList[volumeIndex].inDS->getVtkImageData();

	// Update the scalar volume if necessary (since some are computed only by request)
	if (this->volumeInfoList[volumeIndex].enable && image->GetActualMemorySize() == 0.0)
	{
		image->Update();
		this->core()->data()->dataSetChanged(this->volumeInfoList[volumeIndex].inDS);
	}

	// If this is the first time we turn on seeding for this volume, set the spin boxes to
	// the scalar range of the image.

	if (this->volumeInfoList[volumeIndex].initializedRange == false)
	{
		double range[2] = {0.0, 1.0};

		if (image->GetPointData() && image->GetPointData()->GetScalars())
		{
			image->GetPointData()->GetScalars()->GetRange(range);
		}
		this->volumeInfoList[volumeIndex].minValue = range[0];
		this->volumeInfoList[volumeIndex].maxValue = range[1];
		this->volumeInfoList[volumeIndex].initializedRange = true;
	}

	// Update the GUI to disable/enable controls based on the new setting
	this->setupGUIForVolumes();

	// Update the seed points for this fiber set
	this->updateVolumeSeeds(this->volumeInfoList[volumeIndex]);
}



//----------------------------[ updateROISeeds ]---------------------------\\

void SeedingPlugin::updateROISeeds(roiSeedInfo &info)
{
	// If we don't want any seeds, throw away the output data set
	if (info.seedType == RST_NoSeeding)
	{
		if (info.outDS)
			this->core()->data()->removeDataSet(info.outDS);

		info.outDS = NULL;
		return;
	}

	// Check if the selected voxel image index is valid
	bool voxelIndexIsValid =	this->ui->roiVoxelsCombo->currentIndex() >= 0 && 
								this->ui->roiVoxelsCombo->currentIndex() < this->roiVoxelImages.size();

	// Configure the filter. If the voxel image index was not valid, we always use distance seeding.
	info.filter->setSeedVoxels(voxelIndexIsValid ? this->roiVoxelImages[this->ui->roiVoxelsCombo->currentIndex()] : NULL);
	info.filter->setSeedMethod(voxelIndexIsValid ? info.seedType : RST_Distance);
	info.filter->setSeedDistance(info.distance);
	info.filter->SetInput(info.inDS->getVtkPolyData());

	// Update the filter
	info.filter->Modified();
	info.filter->Update();

	// If we've got an existing output data set...
	if (info.outDS)
	{
		// ...either update it, if we've got at least one seed point...
		if (info.filter->GetOutput()->GetNumberOfPoints() > 0)
		{
			info.outDS->setName(info.inDS->getName() + " (Seeds)");
			info.outDS->updateData(vtkObject::SafeDownCast(info.filter->GetOutput()));
			this->core()->data()->dataSetChanged(info.outDS);
		}
		// ...or remove it, if there are no seed points
		else
		{
			this->core()->data()->removeDataSet(info.outDS);
			info.outDS = NULL;
		}
	}
	// If we didn't have an output data set yet, and we've got at least one seed
	// point, we create the output data set now.

	else if (info.filter->GetOutput()->GetNumberOfPoints() > 0)
	{
		info.outDS = new data::DataSet(info.inDS->getName() + " (Seeds)", "seed points", vtkObject::SafeDownCast(info.filter->GetOutput()));
		this->core()->data()->addDataSet(info.outDS);
	}
}


//---------------------------[ updateFiberSeeds ]--------------------------\\

void SeedingPlugin::updateFiberSeeds(fiberSeedInfo &info)
{
	// If we don't want any seeds, throw away the output data set
	if (!(info.enable))
	{
		if (info.outDS)
			this->core()->data()->removeDataSet(info.outDS);

		info.outDS = NULL;
		return;
	}

	// Either run the input through the simplification filter...
	if (info.doFixedDistance)
	{
		info.simplifyFilter->SetStepLength(info.distance);
		info.simplifyFilter->SetInput(info.inDS->getVtkPolyData());
		info.simplifyFilter->Modified();
		info.simplifyFilter->Update();
		info.seedFilter->SetInput(vtkDataSet::SafeDownCast(info.simplifyFilter->GetOutput()));
	}
	// ...or add it directly the seeding filter
	else
	{
		info.seedFilter->SetInput(vtkDataSet::SafeDownCast(info.inDS->getVtkPolyData()));
	}

	info.seedFilter->Update();

	// If we've got an existing output data set...
	if (info.outDS)
	{
		// ...either update it, if we've got at least one seed point...
		if (info.seedFilter->GetOutput()->GetNumberOfPoints() > 0)
		{
			info.outDS->setName(info.inDS->getName() + " (Seeds)");
			info.outDS->updateData(vtkObject::SafeDownCast(info.seedFilter->GetOutput()));
			this->core()->data()->dataSetChanged(info.outDS);
		}
		// ...or remove it, if there are no seed points
		else
		{
			this->core()->data()->removeDataSet(info.outDS);
			info.outDS = NULL;
		}
	}

	// If we didn't have an output data set yet, and we've got at least one seed
	// point, we create the output data set now.

	else if (info.seedFilter->GetOutput()->GetNumberOfPoints() > 0)
	{
		info.outDS = new data::DataSet(info.inDS->getName() + " (Seeds)", "seed points", vtkObject::SafeDownCast(info.seedFilter->GetOutput()));
		this->core()->data()->addDataSet(info.outDS);
	}
}


//--------------------------[ updateVolumeSeeds ]--------------------------\\

void SeedingPlugin::updateVolumeSeeds(volumeSeedInfo &info)
{
	// If we don't want any seeds, throw away the output data set
	if (!(info.enable))
	{
		if (info.outDS)
			this->core()->data()->removeDataSet(info.outDS);

		info.outDS = NULL;
		return;
	}

	// Configure the filter
	info.seedFilter->SetInput(info.inDS->getVtkImageData());
	info.seedFilter->setMinThreshold(info.minValue);
	info.seedFilter->setMaxThreshold(info.maxValue);
	info.seedFilter->Modified();
	info.seedFilter->Update();

	// If we've got an existing output data set...
	if (info.outDS)
	{
		// ...either update it, if we've got at least one seed point...
		if (info.seedFilter->GetOutput()->GetNumberOfPoints() > 0)
		{
			info.outDS->setName(info.inDS->getName() + " (Seeds)");
			info.outDS->updateData(vtkObject::SafeDownCast(info.seedFilter->GetOutput()));
			this->core()->data()->dataSetChanged(info.outDS);
		}
		// ...or remove it, if there are no seed points
		else
		{
			this->core()->data()->removeDataSet(info.outDS);
			info.outDS = NULL;
		}
	}

	// If we didn't have an output data set yet, and we've got at least one seed
	// point, we create the output data set now.

	else if (info.seedFilter->GetOutput()->GetNumberOfPoints() > 0)
	{
		info.outDS = new data::DataSet(info.inDS->getName() + " (Seeds)", "seed points", vtkObject::SafeDownCast(info.seedFilter->GetOutput()));
		this->core()->data()->addDataSet(info.outDS);
	}
}


//------------------------[ connectControlsForROIs ]-----------------------\\

void SeedingPlugin::connectControlsForROIs(bool doConnect)
{
	// Connect or disconnect controls for the ROI group
	if (doConnect)
	{
		connect(this->ui->roiDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setupGUIForROIs()));
		connect(this->ui->roiNoSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(ROISettingsChanged()));
		connect(this->ui->roiDistanceRadio, SIGNAL(clicked(bool)), this, SLOT(ROISettingsChanged()));
		connect(this->ui->roiVoxelsRadio, SIGNAL(clicked(bool)), this, SLOT(ROISettingsChanged()));
		connect(this->ui->roiDistanceSpin, SIGNAL(valueChanged(double)), this, SLOT(ROISettingsChanged()));
		connect(this->ui->roiVoxelsCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(ROISettingsChanged()));
	}
	else
	{
		disconnect(this->ui->roiDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setupGUIForROIs()));
		disconnect(this->ui->roiNoSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(ROISettingsChanged()));
		disconnect(this->ui->roiDistanceRadio, SIGNAL(clicked(bool)), this, SLOT(ROISettingsChanged()));
		disconnect(this->ui->roiVoxelsRadio, SIGNAL(clicked(bool)), this, SLOT(ROISettingsChanged()));
		disconnect(this->ui->roiDistanceSpin, SIGNAL(valueChanged(double)), this, SLOT(ROISettingsChanged()));
		disconnect(this->ui->roiVoxelsCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(ROISettingsChanged()));
	}
}


//-----------------------[ connectControlsForFibers ]----------------------\\

void SeedingPlugin::connectControlsForFibers(bool doConnect)
{
	// Connect or disconnect controls for the fibers group
	if (doConnect)
	{
		connect(this->ui->fiberDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setupGUIForFibers()));
		connect(this->ui->fiberNoSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(FiberSettingsChanged()));
		connect(this->ui->fiberSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(FiberSettingsChanged()));
		connect(this->ui->fiberSimplifyCheck, SIGNAL(toggled(bool)), this, SLOT(FiberSettingsChanged()));
		connect(this->ui->fiberSimplifySpin, SIGNAL(valueChanged(double)), this, SLOT(FiberSettingsChanged()));
	}
	else
	{
		disconnect(this->ui->fiberDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setupGUIForFibers()));
		disconnect(this->ui->fiberNoSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(FiberSettingsChanged()));
		disconnect(this->ui->fiberSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(FiberSettingsChanged()));
		disconnect(this->ui->fiberSimplifyCheck, SIGNAL(toggled(bool)), this, SLOT(FiberSettingsChanged()));
		disconnect(this->ui->fiberSimplifySpin, SIGNAL(valueChanged(double)), this, SLOT(FiberSettingsChanged()));
	}
}


//----------------------[ connectControlsForVolumes ]----------------------\\

void SeedingPlugin::connectControlsForVolumes(bool doConnect)
{
	// Connect or disconnect controls for the volume group
	if (doConnect)
	{
		connect(this->ui->volumeDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setupGUIForVolumes()));
		connect(this->ui->volumeNoSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(VolumeSettingsChanged()));
		connect(this->ui->volumeSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(VolumeSettingsChanged()));
		connect(this->ui->volumeRangeMinSpin, SIGNAL(valueChanged(double)), this, SLOT(VolumeSettingsChanged()));
		connect(this->ui->volumeRangeMaxSpin, SIGNAL(valueChanged(double)), this, SLOT(VolumeSettingsChanged()));
	}
	else
	{
		disconnect(this->ui->volumeDataCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setupGUIForVolumes()));
		disconnect(this->ui->volumeNoSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(VolumeSettingsChanged()));
		disconnect(this->ui->volumeSeedingRadio, SIGNAL(clicked(bool)), this, SLOT(VolumeSettingsChanged()));
		disconnect(this->ui->volumeRangeMinSpin, SIGNAL(valueChanged(double)), this, SLOT(VolumeSettingsChanged()));
		disconnect(this->ui->volumeRangeMaxSpin, SIGNAL(valueChanged(double)), this, SLOT(VolumeSettingsChanged()));
	}
}



} // namespace bmia


Q_EXPORT_PLUGIN2(libSeedingPlugin, bmia::SeedingPlugin)
