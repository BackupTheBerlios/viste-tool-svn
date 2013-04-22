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
 * TransformationPlugin.cxx
 *
 * 2011-04-27	Evert van Aart
 * - Version 1.0.0.
 * - First version.
 *
 * 2011-08-22	Evert van Aart
 * - Version 1.0.1.
 * - Improved stability.
 * - Added more comments.
 * - Removed the "Cancel" button from the progress bar.
 *
 */


/** Includes */

#include "TransformationPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

TransformationPlugin::TransformationPlugin() : Plugin("Transformation")
{
	// Create the GUI of the widget
	this->widget = new QWidget();
	this->ui = new Ui::TransformationForm();
	this->ui->setupUi(this->widget);

	// Connect the GUI controls
	connect(this->ui->flipXButton, SIGNAL(clicked()), this, SLOT(flipImageX()));
	connect(this->ui->flipYButton, SIGNAL(clicked()), this, SLOT(flipImageY()));
	connect(this->ui->flipZButton, SIGNAL(clicked()), this, SLOT(flipImageZ()));
}


//---------------------------------[ init ]--------------------------------\\

void TransformationPlugin::init()
{

}


//------------------------------[ Destructor ]-----------------------------\\

TransformationPlugin::~TransformationPlugin()
{
	delete this->widget;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * TransformationPlugin::getGUI()
{
	// Return the GUI widget
	return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void TransformationPlugin::dataSetAdded(data::DataSet * ds)
{
	// Add supported data sets to the GUI
	if (ds->getKind() == "DTI" || ds->getKind() == "scalar volume")
	{
		if (!(ds->getVtkImageData()))
			return;
		
		QString prefix = "Unknown Image Type - ";

		if (ds->getKind() == "DTI")
			prefix = "DTI - ";
		else if (ds->getKind() == "scalar volume")
			prefix = "Scalars - ";

		this->ui->imageCombo->addItem(prefix + ds->getName());
		this->imageList.append(ds);
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void TransformationPlugin::dataSetChanged(data::DataSet * ds)
{
	// Rename existing data sets in the GUI
	if ((ds->getKind() == "DTI" || ds->getKind() == "scalar volume") && this->imageList.contains(ds)) 
	{
		int dsIndex = this->imageList.indexOf(ds);

		QString prefix = "Unknown Image Type - ";

		if (ds->getKind() == "DTI")
			prefix = "DTI - ";
		else if (ds->getKind() == "scalar volume")
			prefix = "Scalars - ";

		this->ui->imageCombo->setItemText(dsIndex, prefix + ds->getName());
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void TransformationPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Remove data set from the GUI and the list
	if ((ds->getKind() == "DTI" || ds->getKind() == "scalar volume") && this->imageList.contains(ds)) 
	{
		int dsIndex = this->imageList.indexOf(ds);

		this->imageList.removeAt(dsIndex);

		this->ui->imageCombo->removeItem(dsIndex);
	}
}


//------------------------------[ flipImage ]------------------------------\\

void TransformationPlugin::flipImage(int axis)
{
	// Make sure we've selected an image
	if (this->ui->imageCombo->currentIndex() < 0 || this->ui->imageCombo->currentIndex() >= this->imageList.size())
		return;

	// Get the data set and its image
	data::DataSet * ds = this->imageList[this->ui->imageCombo->currentIndex()];

	vtkImageData * image = ds->getVtkImageData();

	if (!image)
		return;

	// If necessary, compute the image now (for images that are computed on-demand)
	if (image->GetActualMemorySize() == 0)
	{
		image->Update();
		this->core()->data()->dataSetChanged(ds);
	}

	// Get the point data of the image
	vtkPointData * pointData = image->GetPointData();

	if (!pointData)
		return;

	// Create a progress bar
	QProgressDialog progress;
	progress.setWindowTitle("Transformation");
	progress.setMinimumDuration(500);
	progress.setRange(0, 100);
	progress.setValue(0);
	progress.setCancelButton(NULL);

	int numberOfArrays = pointData->GetNumberOfArrays();

	// Compute the step size for the progress bar
	int progressStepSize = image->GetNumberOfPoints() / 25;
	progressStepSize += (progressStepSize == 0) ? 1 : 0;

	// Loop through all scalar arrays
	for (int arrayIndex = 0; arrayIndex < numberOfArrays; ++arrayIndex)
	{
		// Get the current scalar array
		vtkDataArray * currentArray = pointData->GetArray(arrayIndex);

		if (!currentArray)
			continue;

		// Make sure the dimensions of the scalar array match those of the image
		if (currentArray->GetNumberOfTuples() != image->GetNumberOfPoints())
			continue;

		progress.setLabelText("Flipping scalar array #" + QString::number(arrayIndex));

		ImageType arrayType = IT_Scalars;

		// Check if we've got DTI tensors
		QString arrayName = QString(currentArray->GetName());
		if (ds->getKind() == "DTI" && arrayName == "Tensors" && currentArray->GetNumberOfComponents() == 6)
			arrayType = IT_DTI;

		int dims[3];
		image->GetDimensions(dims);

		// Compute the dimension limits
		int dimLimits[3] = {dims[0], dims[1], dims[2]};
		dimLimits[axis] = dimLimits[axis] / 2;
		if ((dims[axis] % 2) != 0)
			dimLimits[axis]++;

		int ijk[3];

		int voxelCount = 0;

		// Loop through half of the image
		for (ijk[0] = 0; ijk[0] < dimLimits[0]; ++(ijk[0]))					{
		for (ijk[1] = 0; ijk[1] < dimLimits[1]; ++(ijk[1]))					{
		for (ijk[2] = 0; ijk[2] < dimLimits[2]; ++(ijk[2]), ++voxelCount)	{

			// Update the progress bar
			if ((voxelCount % progressStepSize) == 0)
			{
				progress.setValue((int) (((float) voxelCount / (float) image->GetNumberOfPoints()) * 200.0f));
			}

			// Compute the ID of the current point
			vtkIdType currentPoint = image->ComputePointId(ijk);

			int ijkNew[3] = {ijk[0], ijk[1], ijk[2]};

			// Flip the point indices
			ijkNew[axis] = dims[axis] - ijk[axis] - 1;

			// Compute the ID of the target point
			vtkIdType targetPoint = image->ComputePointId(ijkNew);

			if (currentPoint == -1 || targetPoint == -1)
				continue;

			// For scalars, we can simply swap the values of the two points
			if (arrayType == IT_Scalars)
			{
				for (int componentIndex = 0; componentIndex < currentArray->GetNumberOfComponents(); ++componentIndex)
				{
					double tempComponent = currentArray->GetComponent(currentPoint, componentIndex);
					currentArray->SetComponent(currentPoint, componentIndex, currentArray->GetComponent(targetPoint, componentIndex));
					currentArray->SetComponent(targetPoint, componentIndex, tempComponent);
				}
			}

			// For DTI, we need to recompute the tensors
			else if (arrayType == IT_DTI)
			{
				double originalTensorA[6];
				double originalTensorB[6];

				currentArray->GetTuple(currentPoint, originalTensorA);
				currentArray->GetTuple(targetPoint, originalTensorB);
				
				double flippedTensorA[6];
				double flippedTensorB[6];

				vtkTensorMath::Flip(originalTensorA, flippedTensorA, axis);
				vtkTensorMath::Flip(originalTensorB, flippedTensorB, axis);

				currentArray->SetTuple(targetPoint,  flippedTensorA);
				currentArray->SetTuple(currentPoint, flippedTensorB);
			}
		
		} } }
	}

	// Done!
	progress.setValue(100);

	// Notify the data manager that we've modified the data set
	image->Modified();
	this->core()->data()->dataSetChanged(ds);
}


} // namespace


Q_EXPORT_PLUGIN2(libTransformationPlugin, bmia::TransformationPlugin)
