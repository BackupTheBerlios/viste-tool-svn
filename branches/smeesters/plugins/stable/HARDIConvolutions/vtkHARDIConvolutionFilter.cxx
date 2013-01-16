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
 * vtkHARDIConvolutionFilter.cxx
 *
 * 2011-08-01	Evert van Aart
 * - First version
 * 2011-09-12	Ralph Brecheisen
 * - Added conditionial compile for 'sprintf_s' and 'sprintf'
 *
 */


/** Includes */

#include "vtkHARDIConvolutionFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkHARDIConvolutionFilter)


//-----------------------------[ Constructor ]-----------------------------\\

vtkHARDIConvolutionFilter::vtkHARDIConvolutionFilter()
{
	// Set pointers to NULL
	this->generator				= NULL;
	this->niftiFileNames		= NULL;
	this->neighborhood			= NULL;
	this->kernelValues			= NULL;
	this->inputImage			= NULL;
	this->imageValues			= NULL;
	this->kernelMask			= NULL;

	// Set default parameter values
	this->Threshold				= 0.15;
	this->numberOfIterations	= 1;
	this->useRelativethreshold	= true;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkHARDIConvolutionFilter::~vtkHARDIConvolutionFilter()
{
	// Delete the temporary arrays
	if (this->neighborhood)
	{
		delete this->neighborhood;
		this->neighborhood = NULL;
	}

	if (this->kernelValues)
	{
		delete this->kernelValues;
		this->kernelValues = NULL;
	}

	if (this->kernelMask)
	{
		delete this->kernelMask;
		this->kernelMask = NULL;
	}
}


//----------------------------[ SimpleExecute ]----------------------------\\

void vtkHARDIConvolutionFilter::SimpleExecute(vtkImageData * input, vtkImageData * output)
{
	// Check if both input and output have been set
	if (input == NULL || output == NULL)
	{
		vtkErrorMacro(<< "Input and/or output not set!");
		return;
	}

	this->inputImage = input;

	// Get the dimensions of the input image
	int imageDim[3];
	this->inputImage->GetDimensions(imageDim);

	// Get the point data of the input
	vtkPointData * inPD = this->inputImage->GetPointData();

	if (!inPD)
	{
		vtkErrorMacro(<< "Input does not contain point data!");
		return;
	}

	// Get the angles array and the radii from the input image
	vtkDataArray * anglesArray = vtkDoubleArray::SafeDownCast(inPD->GetArray("Spherical Directions"));
	this->imageValues = vtkDoubleArray::SafeDownCast(inPD->GetArray("Vectors"));

	if (!anglesArray || !(this->imageValues))
	{
		vtkErrorMacro(<< "Input image missing values and/or directions!");
		return;
	}

	// Number of angle pairs should be equal to the number of values per voxel
	if (this->imageValues->GetNumberOfComponents() != anglesArray->GetNumberOfTuples())
	{
		vtkErrorMacro(<< "Number of directions mismatch!");
		return;
	}

	// Store the number of directions and the number of points
	int numberOfDirections = anglesArray->GetNumberOfTuples();
	int numberOfPoints = this->imageValues->GetNumberOfTuples();

	int kernelDim[3];

	// If we've got a kernel generator, it already has the correct dimensions,
	// so we can get the kernel dimensions directly.

	if (this->generator)
	{
		this->generator->UpdateDimensions();
		this->generator->GetDim(kernelDim);
	}

	// If we're reading the kernels from NIfTI files, we need to load the first
	// file, and get its dimensions. These dimensions will then be used to check
	// if the dimensions of the other NIfTI files are correct.

	else if (this->niftiFileNames && this->niftiFileNames->size() == numberOfDirections)
	{
		KernelNIfTIReader * reader = new KernelNIfTIReader;
		reader->setFileName(this->niftiFileNames->at(0));
		
		if (!(reader->getDimensions(kernelDim)))
		{
			vtkErrorMacro(<< "Error reading dimensions of the first NIfTI file!");
			return;
		}

		delete reader;
	}
	else
	{
		vtkErrorMacro(<< "No available input method!");
		return;
	}

	// Get the point data of the output
	vtkPointData * outPD = output->GetPointData();

	if (!outPD)
	{
		vtkErrorMacro(<< "Output does not contain point data!");
		return;
	}

	// Copy the structure of the input image to the output
	output->CopyStructure(this->inputImage);

	// Copy the angles array to the output
	vtkDoubleArray * outAnglesArray = vtkDoubleArray::New();
	outAnglesArray->DeepCopy(anglesArray);
	outAnglesArray->SetName("Spherical Directions");
	outPD->AddArray(outAnglesArray);
	outAnglesArray->Delete();

	vtkIntArray * trianglesArray = vtkIntArray::SafeDownCast(inPD->GetArray("Triangles"));

	// If available, copy the triangle array
	if (trianglesArray)
	{
		vtkIntArray * outTrianglesArray = vtkIntArray::New();
		outTrianglesArray->DeepCopy(trianglesArray);
		outTrianglesArray->SetName("Triangles");
		outPD->AddArray(outTrianglesArray);
		outTrianglesArray->Delete();
	}

	// Create the array for output values
	vtkDoubleArray * outValuesArray = vtkDoubleArray::New();
	outValuesArray->SetNumberOfComponents(this->imageValues->GetNumberOfComponents());
	outValuesArray->SetNumberOfTuples(this->imageValues->GetNumberOfTuples());
	outValuesArray->SetName("Vectors");
	outPD->AddArray(outValuesArray);
	outValuesArray->Delete();

	vtkDoubleArray * tempValuesArray = NULL;

	// If we're going to do more than one iterations, we need a temporary array
	// to store the output of the previous iterations (since we do not want to
	// overwrite the input values).

	if (this->numberOfIterations > 1)
	{
		tempValuesArray = vtkDoubleArray::New();
		tempValuesArray->SetNumberOfComponents(this->imageValues->GetNumberOfComponents());
		tempValuesArray->SetNumberOfTuples(this->imageValues->GetNumberOfTuples());
	}

	// Delete existing arrays
	if (this->neighborhood)
	{
		delete this->neighborhood;
		this->neighborhood = NULL;
	}

	if (this->kernelValues)
	{
		delete this->kernelValues;
		this->kernelValues = NULL;
	}

	if (this->kernelMask)
	{
		delete this->kernelMask;
		this->kernelMask = NULL;
	}

	// Compute the number of voxels in the kernel
	int kernelSize = kernelDim[0] * kernelDim[1] * kernelDim[2];

	// The neighborhood contains the point indices of the points around the current
	// voxel. This neighborhood has the same size as the kernel.
	this->neighborhood = new vtkIdType[kernelSize];

	// Array containing a single kernel image (possible mapped).
	this->kernelValues = new double[kernelSize * numberOfDirections];

	// Kernel map, containing for each mapped point its original point index (within
	// the kernel), and its direction index.
	this->kernelMask = new MapInfo[kernelSize * numberOfDirections];

	// The actual size of the map is determined while mapping
	this->maskSize = 0;

	double newValue;

	// Pointers to the input and output value arrays, used for multiple iterations
	vtkDoubleArray * valueSource = this->imageValues;
	vtkDoubleArray * valueDest   = outValuesArray;

	// Compute the total number of directions (used for the progress bar)
	double totalNumberOfDirections = (double) this->numberOfIterations * numberOfDirections;

	// Loop through all iterations
	for (int iterIndex = 0; iterIndex < this->numberOfIterations; ++iterIndex)
	{
		// Update the progress bar text
		char progressText[128];
#ifdef _WIN32
		sprintf_s(&(progressText[0]), 128, "Applying Convolution... (Iteration #%d)\0", iterIndex + 1);
#else
		sprintf(&(progressText[0]), "Applying Convolution... (Iteration #%d)\0", iterIndex + 1);
#endif
		this->SetProgressText(progressText);

		// Loop through all directions
		for (int firstDirectionID = 0; firstDirectionID < numberOfDirections; ++firstDirectionID)
		{
			// Update the progress
			double currentDirection = (double) (numberOfDirections * iterIndex + firstDirectionID);
			this->UpdateProgress(currentDirection / totalNumberOfDirections);

			// If we've got a generator, use it to compute a kernel now
			if (this->generator)
			{
				this->generator->BuildSingleKernel(firstDirectionID, kernelValues);
			}
			// Otherwise, we need to read the kernel from a NIfTI file
			else
			{
				// Create and setup the reader
				KernelNIfTIReader * reader = new KernelNIfTIReader;
				reader->setFileName(this->niftiFileNames->at(firstDirectionID));
				reader->setDimensions(kernelDim);
				reader->setNumberOfDirections(numberOfDirections);
				
				// Try to read the kernel image
				if (!(reader->readKernel(kernelValues)))
				{
					vtkErrorMacro(<< "Error reading NIfTI file!");
					delete reader;
					return;
				}

				delete reader;
			}

			// Mask the kernel to remove values lower than the threshold
			this->maskKernel(kernelSize, numberOfDirections);

			int ijk[3];

			// Loop through all values in the input image
			for (ijk[0] = 0; ijk[0] < imageDim[0]; ++ijk[0])	{
			for (ijk[1] = 0; ijk[1] < imageDim[1]; ++ijk[1])	{
			for (ijk[2] = 0; ijk[2] < imageDim[2]; ++ijk[2])	{

				// Compute the point indices of the neighborhood around the current voxel
				this->getPositionNeighbourhood(ijk, kernelDim, imageDim);

				double kernelValue;
				double imageValue;

				// Reset the output value
				newValue = 0.0;

				// Loop through all masked values
				for (int maskIndex = 0; maskIndex < this->maskSize; ++maskIndex)
				{
					// Get the direction and point index of the masked point
					int dirIndex   = this->kernelMask[maskIndex].dirId;
					int pointIndex = this->kernelMask[maskIndex].pointId;

					// Get the masked kernel value
					kernelValue = this->kernelValues[maskIndex];

					// Use the point index and direction to get the input value
					imageValue = valueSource->GetComponent(this->neighborhood[pointIndex], dirIndex);

					// Multiply the kernel value by the image value and add it to the output value
					newValue += kernelValue * imageValue;
				}

				// Add the computed output value to the output
				valueDest->SetComponent(this->inputImage->ComputePointId(ijk), firstDirectionID, newValue);

			}	}	} // for [all voxels in the input image]

		} // for [all directions]

		// If we've got more than one iteration, copy the output to a temporary array
		if (this->numberOfIterations > 1)
		{
			tempValuesArray->DeepCopy(valueDest);
			valueSource = tempValuesArray;
		}
	}

	// Finalize the progress bar
	this->UpdateProgress(1.0);

	// Delete the temporary arrays
	delete [] this->kernelValues;
	this->kernelValues = NULL;

	delete [] this->neighborhood;
	this->neighborhood = NULL;

	delete [] this->kernelMask;
	this->kernelMask = NULL;

	if (tempValuesArray)
		tempValuesArray->Delete();
}


//-----------------------[ getPositionNeighbourhood ]----------------------\\

void vtkHARDIConvolutionFilter::getPositionNeighbourhood(int * ijkBase, int * kernelDim, int * imageDim)
{
	// Divide kernel dimensions by two and floor the results
	int kernelSize[3];
	kernelSize[0] = (int) floor((double) kernelDim[0] / 2.0);
	kernelSize[1] = (int) floor((double) kernelDim[1] / 2.0);
	kernelSize[2] = (int) floor((double) kernelDim[2] / 2.0);

	vtkIdType kernelPointId;

	int ijk[3] = {0, 0, 0};

	// Get the number of directions
	int numberOfAngles = this->imageValues->GetNumberOfComponents();

	// Loop through all kernel points. For each point, compute the 3D indices of 
	// the neighboring point. If an index is located outside of the image, we 
	// clamp it to the edge of the image.

	for (int k = -kernelSize[2]; k <= kernelSize[2]; ++k)
	{
		ijk[2] = ijkBase[2] + k;

		if (ijk[2] < 0)
			ijk[2] = 0;

		if (ijk[2] >= imageDim[2])
			ijk[2]  = imageDim[2] - 1;

		for (int j = -kernelSize[1]; j <= kernelSize[1]; ++j)
		{
			ijk[1] = ijkBase[1] + j;
			
			if (ijk[1] < 0)
				ijk[1] = 0;

			if (ijk[1] >= imageDim[1])
				ijk[1]  = imageDim[1] - 1;

			for (int i = -kernelSize[0]; i <= kernelSize[0]; ++i)
			{
				ijk[0] = ijkBase[0] + i;

				if (ijk[0] < 0)
					ijk[0] = 0;

				if (ijk[0] >= imageDim[0])
					ijk[0]  = imageDim[0] - 1;

				// Use the 3D indices to get the point index of the neighboring point
				kernelPointId = this->inputImage->ComputePointId(ijk);

				if (kernelPointId == -1)
				{
					vtkErrorMacro(<< "Invalid point ID!");
					kernelPointId = 0;
				}

				// Store the point index in the neighborhood array
				this->neighborhood[	(i + kernelSize[0]) +
									(j + kernelSize[1]) * kernelDim[0] + 
									(k + kernelSize[2]) * kernelDim[0] * kernelDim[1] ] = kernelPointId;
			}
		}
	}
}


//------------------------------[ maskKernel ]-----------------------------\\

void vtkHARDIConvolutionFilter::maskKernel(int kernelSize, int numberOfDirections)
{
	int kernelIndex = 0;
	int maskIndex   = 0;

	double T;

	// If we're using a relative threshold, we first need to compute the maximum
	if (this->useRelativethreshold)
	{
		double maxValue = -1.0;

		for (int i = 0; i < kernelSize * numberOfDirections; ++i)
		{
			if (this->kernelValues[i] > maxValue)
				maxValue = this->kernelValues[i];
		}

		// Multiply the maximum by the threshold percentage
		T = maxValue * this->Threshold;
	}
	else
	{
		// Otherwise, just use the absolute threshold
		T = this->Threshold;
	}

	// Direction and point indices of the masked point
	unsigned short dirIndex = 0;
	vtkIdType pointIndex = 0;

	// Loop through all kernel values
	for (kernelIndex = 0; kernelIndex < kernelSize * numberOfDirections; ++kernelIndex)
	{
		// Get the current kernel value
		double kernelValue = this->kernelValues[kernelIndex];

		// If the value is larger than the threshold...
		if (kernelValue > T)
		{
			// ...move it to the front of the array...
			this->kernelValues[maskIndex] = kernelValue;

			// ...store its direction and point index in the mask...
			this->kernelMask[maskIndex].dirId   = dirIndex;
			this->kernelMask[maskIndex].pointId = pointIndex;

			// ...and increment the masking index.
			maskIndex++;
		}

		// The direction index is reset to zero when it reaches the number of directions
		if (++dirIndex == numberOfDirections)	
			dirIndex = 0;

		// The point index is reset to zero when it reaches the number of points in the kernel
		if (++pointIndex == kernelSize)			
			pointIndex = 0;
	}

	// The size of the mask is equal to its current index
	this->maskSize = maskIndex;
}


} // namespace
