/*
 * HARDIMeasuresPlugin.cxx
 *
 * 2011-04-29	Evert van Aart
 * - Version 1.0.0.
 * - First version.
 *
 * 2011-05-04	Evert van Aart
 * - Version 1.0.1.
 * - Added the volume measure.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.2.
 * - Improved attribute handling.
 *
 * 2011-08-05	Evert van Aart
 * - Version 1.0.3.
 * - Fixed an error in the computation of the unit vectors.
 *
 */


/** Includes */

#include "HARDIMeasuresPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

HARDIMeasuresPlugin::HARDIMeasuresPlugin() : plugin::Plugin("HARDI Measures")
{

}


//------------------------------[ Destructor ]-----------------------------\\

HARDIMeasuresPlugin::~HARDIMeasuresPlugin()
{
	// Loop through all filters
	for (int i = 0; i < this->discreteSphereFilters.size(); ++i)
	{
		// Delete the progress bar for the current filter
		this->core()->out()->deleteProgressBarForAlgorithm(this->discreteSphereFilters[i]);
	}

	// Clear the list of filters
	this->discreteSphereFilters.clear();

	// Loop through all input images
	for (int imageID = 0; imageID < this->images.size(); ++imageID)
	{
		// Get the output information for the current input image
		OutputInformation currentImage = this->images.at(imageID);

		// Copy the list of scalar measure outputs
		QList<data::DataSet *> outputListCopy = currentImage.outputs;

		// Delete all output images for this input image
		for (int outID = 0; outID < outputListCopy.size(); ++outID)
		{
			data::DataSet * currentOutImage = outputListCopy.at(outID);

			if (currentOutImage)
			{
				this->core()->data()->removeDataSet(currentOutImage);
			}
		}
	}

	// Clear the list of images
	this->images.clear();
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void HARDIMeasuresPlugin::dataSetAdded(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions
	if (ds->getKind() == "discrete sphere" && this->findInputImage(ds) == -1)
	{
		// Check if the data set contains an image
		vtkImageData * image = ds->getVtkImageData();

		if (!image)
			return;

		// Check if the image contains point data
		vtkPointData * imagePD = image->GetPointData();

		if (!imagePD)
			return;

		// Check if the point data contains a spherical directions array
		vtkDoubleArray * anglesArray = vtkDoubleArray::SafeDownCast(imagePD->GetArray("Spherical Directions"));

		if (!anglesArray)
			return;

		// Check if the point data contains a triangles array
		vtkIntArray * trianglesArray = vtkIntArray::SafeDownCast(imagePD->GetArray("Triangles"));

		// If not, compute one now, using the Sphere Triangulator
		if (!trianglesArray)
		{
			trianglesArray = vtkIntArray::New();
			SphereTriangulator * triangulator = new SphereTriangulator;
			triangulator->triangulateFromAnglesArray(anglesArray, trianglesArray);
			trianglesArray->SetName("Triangles");
		}

		// Create an output information object
		OutputInformation newInfo;
		newInfo.input = ds;

		// Loop through all supported measures
		for (int i = 0; i < (int) vtkDiscreteSphereToScalarVolumeFilter::DSPHM_NumberOfMeasures; ++i)
		{
			// Create a filter for the current measure
			vtkDiscreteSphereToScalarVolumeFilter * filter = vtkDiscreteSphereToScalarVolumeFilter::New();
			filter->SetInput(image);
			filter->setTrianglesArray(trianglesArray);
			filter->setCurrentMeasure(i);

			// Create a progress bar for this filter
			this->core()->out()->createProgressBarForAlgorithm(filter, "HARDI Measures");
			this->discreteSphereFilters.append(filter);

			// Create an output data set for the filter output
			data::DataSet * outDS = new data::DataSet(ds->getName() + " [" + 
				filter->getShortMeasureName(i) + "]", "scalar volume", filter->GetOutput());

			// If available, copy the input transformation matrix to the output
			outDS->getAttributes()->copyTransformationMatrix(ds);

			// Add the output data set to the data manager...
			this->core()->data()->addDataSet(outDS);

			// ...and to the list of outputs
			newInfo.outputs.append(outDS);
		}

		// Add the output information object to the list
		this->images.append(newInfo);
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void HARDIMeasuresPlugin::dataSetChanged(data::DataSet * ds)
{
	// When an image is externally modified, we simply remove and re-add it
	if (ds->getKind() == "discrete sphere" && this->findInputImage(ds) != -1)
	{
		this->dataSetRemoved(ds);
		this->dataSetAdded(ds);
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void HARDIMeasuresPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Discrete Sphere Function
	if (ds->getKind() == "discrete sphere")
	{
		// Get the index of the input image
		int imageID = this->findInputImage(ds);

		// Do nothing if this image is not part of the "images" list
		if (imageID == -1)
			return;
	
		// Get the output information object 
		OutputInformation info = this->images.at(imageID);

		// Copy the list of scalar measure outputs
		QList<data::DataSet *> outputListCopy = info.outputs;

		// Remove all output images
		for (int outID = 0; outID < outputListCopy.size(); ++outID)
		{
			data::DataSet * currentOutImage = outputListCopy.at(outID);

			if (currentOutImage)
			{
				this->core()->data()->removeDataSet(currentOutImage);
			}
		}

		// Remove the output information object
		this->images.removeAt(imageID);
	}

	// Scalar volumes
	else if (ds->getKind() == "scalar volume")
	{
		int imageID = -1;
		int outID = -1;

		// Do nothing if the output image is not part of the "images" list
		if (!(this->findOutputImage(ds, imageID, outID)))
			return;

		// Remove the output image from the corresponding output information object
		OutputInformation info = this->images[imageID];
		info.outputs.removeAt(outID);
		this->images[imageID] = info;
	}
}


//----------------------------[ findInputImage ]---------------------------\\

int HARDIMeasuresPlugin::findInputImage(bmia::data::DataSet * ds)
{
	// Find the input data set pointer
	for (int i = 0; i < this->images.size(); ++i)
	{
		if (this->images[i].input == ds)
			return i;
	}

	return -1;
}


//---------------------------[ findOutputImage ]---------------------------\\

bool HARDIMeasuresPlugin::findOutputImage(bmia::data::DataSet * ds, int & imageID, int & outID)
{
	// Loop through all output information objects
	for (int i = 0; i < this->images.size(); ++i)
	{
		// Loop through all outputs of one input image
		for (int j = 0; j < this->images[i].outputs.size(); ++j)
		{
			// If we've found the target image, set the indices and return true
			if (this->images[i].outputs[j] == ds)
			{
				imageID = i;
				outID = j;
				return true;
			}
		}
	}

	// If we could not find the target, return false
	return false;
}

} // namespace bmia


Q_EXPORT_PLUGIN2(libHARDIMeasuresPlugin, bmia::HARDIMeasuresPlugin)
