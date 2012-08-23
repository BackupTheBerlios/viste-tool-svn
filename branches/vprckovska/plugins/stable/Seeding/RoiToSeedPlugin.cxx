/*
 * RoiToSeedPlugin.cxx
 *
 * 2010-10-29	Evert van Aart
 * - First Version.
 *
 * 2010-12-15	Evert van Aart
 * - Added support for voxel seeding.
 * - Seed distance is now read from data set attributes.
 *
 * 2011-03-16	Evert van Aart
 * - Version 1.0.0.
 * - Removed the need to compute the normal for primary planes, making the seeding
 *   more robust for elongated ROIs.
 * - Increased stability for voxel seeding when a ROI is touching the edges of
 *   an image. 
 *
 */

 
/** Includes */

#include "RoiToSeedPlugin.h"
#include "vtk2DRoiToSeedFilter.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

RoiToSeedPlugin::RoiToSeedPlugin() : Plugin("ROI to Seed")
{
	
}


//------------------------------[ Destructor ]-----------------------------\\

RoiToSeedPlugin::~RoiToSeedPlugin()
{
	// Iterator for the list of ROIs
	QList<ROIInfo>::iterator ROIIter;

	// Delete all filters in the list of ROIs
	for (ROIIter = this->ROIInfoList.begin(); ROIIter != this->ROIInfoList.end(); ++ROIIter)
	{
		if ((*ROIIter).ROIFilter)
			(*ROIIter).ROIFilter->Delete();
	}
}


//----------------------------[ dataSetAdded ]-----------------------------\\

void RoiToSeedPlugin::dataSetAdded(bmia::data::DataSet * ds)
{
	// Check if the dataset exists
	if (!ds)
		return;

	// We're only interested in ROIs
	if (ds->getKind() != "regionOfInterest")
		return;

	// Create a new ROI information structure
	ROIInfo newROIInfo;

	// Set the input data
	newROIInfo.inputDS   = ds;
	newROIInfo.inputName = ds->getName();
	newROIInfo.inputPD   = ds->getVtkPolyData();

	// Check if the polydata exists
	if (!(newROIInfo.inputPD))
		return;

	// Generate the name of the output
	newROIInfo.outputName = newROIInfo.inputName + " (Seeds)";

	// Create a new filter, store it in the information structure
	vtk2DRoiToSeedFilter * newFilter = (vtk2DRoiToSeedFilter *) vtk2DRoiToSeedFilter::New();
	newROIInfo.ROIFilter = newFilter;

	// Initialize the output data set to NULL
	newROIInfo.outputDS = NULL;

	// Append the ROI information to the list
	this->ROIInfoList.append(newROIInfo);

	// Seed distance and image used for voxel seeding
	double			seedDistance;
	vtkObject *		seedVoxels;
	vtkImageData *	seedVoxelImage = NULL;

	// Seeding method. We use distance seeding by default.
	SeedMethod seedMethod = SM_Distance;

	// Try to get the seed distance
	if (!ds->getAttributes()->getAttribute("Seed distance", seedDistance))
	{
		// If this fails, try to get the voxel seeding image
		if (!ds->getAttributes()->getAttribute("Seed voxels", seedVoxels))
		{
			// If neither attribute has been set, we assume that the "No Seeding"
			// option has been selected, and we do nothing

			return;
		}
		else
		{
			// If we did successfully get the image, we change the seeding
			// method to voxel seeding.

			seedMethod = SM_Voxels;

			// Cast the "vtkObject" to a "vtkImageData"
			seedVoxelImage = vtkImageData::SafeDownCast(seedVoxels);
		}
	}

	// Set the seed distance and input of the filter
	newFilter->setSeedDistance(seedDistance);
	newFilter->setSeedMethod(seedMethod);
	newFilter->setSeedVoxels(seedVoxelImage);
	newFilter->SetInput(newROIInfo.inputPD);

	// Store the filter pointer in the ROIInfo struct
	newROIInfo.ROIFilter = (vtkDataSetToUnstructuredGridFilter *) newFilter;
	
	// Update the filter
	newFilter->Update();

	// Get the seed points
	vtkUnstructuredGrid * filterOutput = newFilter->GetOutput();

	// Create a new data set containing the seed points
	data::DataSet * newDS = new data::DataSet(newROIInfo.outputName, "seed points", (vtkObject *) filterOutput);

	// Store the data set pointer in the ROIInfo struct
	newROIInfo.outputDS = newDS;

	// Add the seed points to the data manager
	this->core()->data()->addDataSet(newDS);

	// Add the information of the ROI to the list for future reference
	this->ROIInfoList.replace(this->ROIInfoList.size() - 1, newROIInfo);
}


//----------------------------[ dataSetChanged ]---------------------------\\

void RoiToSeedPlugin::dataSetChanged(bmia::data::DataSet * ds)
{
	// Check if the dataset exists
	if (!ds)
		return;

	// We're only interested in ROIs
	if (ds->getKind() != "regionOfInterest")
		return;

	// Iterator for the list of ROIs
	QList<ROIInfo>::iterator ROIIter;

	// Does the input data set exist in the list of ROIs?
	bool roiExists = false;

	// Index of the current ROI
	int roiIndex = 0;

	// Check if the input data set exists
	for (ROIIter = this->ROIInfoList.begin(); ROIIter != this->ROIInfoList.end(); ++ROIIter, ++roiIndex)
	{
		if ((*ROIIter).inputDS == ds)
		{
			roiExists = true;
			break;
		}
	}

	// We don't do anything with data sets that have not yet been added
	if (!roiExists)
		return;

	// Get current ROI information structure
	ROIInfo currentROIInfo = (*ROIIter);

	// Get the ROI filter
	vtk2DRoiToSeedFilter * currentFilter = (vtk2DRoiToSeedFilter *) currentROIInfo.ROIFilter;

	// Update the ROI information object
	currentROIInfo.inputName  = ds->getName();
	currentROIInfo.outputName = ds->getName();
	currentROIInfo.inputPD    = ds->getVtkPolyData();
	currentROIInfo.outputName = currentROIInfo.inputName + " (Seeds)";

	// Check if the polydata exists
	if (!(currentROIInfo.inputPD))
		return;

	// Seed distance and image used for voxel seeding
	double			seedDistance;
	vtkObject *		seedVoxels;
	vtkImageData *	seedVoxelImage = NULL;

	// Seeding method. We use distance seeding by default.
	SeedMethod seedMethod = SM_Distance;

	// Try to get the seed distance
	if (!ds->getAttributes()->getAttribute("Seed distance", seedDistance))
	{
		// If this fails, try to get the voxel seeding image
		if (!ds->getAttributes()->getAttribute("Seed voxels", seedVoxels))
		{
			// Remove the existing output data set
			if (currentROIInfo.outputDS)
			{
				this->core()->data()->removeDataSet(currentROIInfo.outputDS);
			}

			// Set the data set pointer to NULL
			currentROIInfo.outputDS = NULL;

			// Replace the ROI information in the list
			this->ROIInfoList.replace(roiIndex, currentROIInfo);

			return;
		}
		else
		{
			// If we did successfully get the image, we change the seeding
			// method to voxel seeding.

			seedMethod = SM_Voxels;

			// Cast the "vtkObject" to a "vtkImageData"
			seedVoxelImage = vtkImageData::SafeDownCast(seedVoxels);
		}
	}

	// Update the seed distance
	currentFilter->setSeedDistance(seedDistance);
	currentFilter->setSeedMethod(seedMethod);
	currentFilter->setSeedVoxels(seedVoxelImage);

	// Set the input to the new polydata object
	currentFilter->SetInput(ds->getVtkPolyData());

	// Force the filter to update
	currentFilter->Modified();
	currentFilter->Update();

	// Update the output data set...
	if (currentROIInfo.outputDS)
	{
		currentROIInfo.outputDS->updateData((vtkObject *) currentFilter->GetOutput());
		currentROIInfo.outputDS->setName(currentROIInfo.outputName);
		this->core()->data()->dataSetChanged(currentROIInfo.outputDS);
	}
	// ...or create a new one
	else
	{
		data::DataSet * newDS = new data::DataSet(currentROIInfo.outputName, "seed points", (vtkObject *) currentFilter->GetOutput());
		currentROIInfo.outputDS = newDS;
		this->core()->data()->addDataSet(newDS);
	}

	// Replace the information object in the list of ROIs
	this->ROIInfoList.replace(roiIndex, currentROIInfo);
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void RoiToSeedPlugin::dataSetRemoved(bmia::data::DataSet * ds)
{
	// Check if the dataset exists
	if (!ds)
		return;

	// We're only interested in ROIs
	if (ds->getKind() != "regionOfInterest")
		return;

	// Iterator for the list of ROIs
	QList<ROIInfo>::iterator ROIIter;

	// Does the input data set exist in the list of ROIs?
	bool roiExists = false;

	// Index of the deleted ROI
	int roiIndex = 0;

	// Check if the input data set exists
	for (ROIIter = this->ROIInfoList.begin(); ROIIter != this->ROIInfoList.end(); ++ROIIter, ++roiIndex)
	{
		if ((*ROIIter).inputDS == ds)
		{
			roiExists = true;
			break;
		}
	}

	// We don't do anything with data sets that have not yet been added
	if (!roiExists)
		return;

	// Get current ROI information structure
	ROIInfo currentROIInfo = (*ROIIter);

	// Delete the ROI filter
	if (currentROIInfo.ROIFilter)
		currentROIInfo.ROIFilter->Delete();

	// Remove the output data set from the data manager
	if (currentROIInfo.outputDS)
		this->core()->data()->removeDataSet(currentROIInfo.outputDS);

	// Remove the ROI information object
	this->ROIInfoList.removeAt(roiIndex);
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libRoiToSeedPlugin, bmia::RoiToSeedPlugin)
