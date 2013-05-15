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
 * vtkFiberTrackingGeodesicFilter.cxx
 *
 * 2011-05-31	Evert van Aart
 * - First Version.
 *
 * 2011-06-08	Evert van Aart
 * - Improved progress reporting: Progress now also depends on the number of additional
 *   angles, not just on the number of seed points.
 * 
 * 2011-07-06	Evert van Aart
 * - First version for the CUDA-enabled version. Changed the way the fibers are 
 *   computed and added to the output.
 *
 */


/** Includes */

#include "vtkFiberTrackingGeodesicFilter_CUDA.h"


namespace bmia {


vtkStandardNewMacro(vtkFiberTrackingGeodesicFilter);



//-----------------------------[ Constructor ]-----------------------------\\

vtkFiberTrackingGeodesicFilter::vtkFiberTrackingGeodesicFilter()
{
	// Set the default values for the parameters
	this->useAdditionAngles		= false;
	this->aaPattern				= AAP_Cone;
	this->aaConeNumberOfAngles	= 8;
	this->aaConeWidth			= 0.1;
	this->aaSphereNumberOfAnglesP = 8;
	this->aaSphereNumberOfAnglesT = 4;
	this->aaIcoTessOrder		= 3;
	this->useStopLength			= false;
	this->useStopAngle			= false;
	this->useStopScalar			= false;
	this->myODESolver			= OS_RK2_Heun;
	this->ppEnable				= false;
	this->ppSharpenMethod		= geodesicPreProcessor::SM_None;
	this->ppGain				= 2000;
	this->ppThreshold			= 0.1;
	this->ppExponent			= 2;
	this->maxNumberOfSteps		= 1024;
	this->numberOfFibersPerLoad = 4096;

	// Set pointers to NULL
	this->inTensorsA			= NULL;
	this->inTensorsB			= NULL;
	this->anglePrevPoints		= NULL;
	this->distanceArray			= NULL;
	this->scalarArray			= NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkFiberTrackingGeodesicFilter::~vtkFiberTrackingGeodesicFilter()
{
	// Clear all remaining seed points
	this->seedPointList.clear();

	// Loop through the map for fiber IDs
	for (QMap<int, outputFiberInfo>::iterator mapIter = this->fiberIdMap.begin(); mapIter != this->fiberIdMap.end(); ++mapIter)
	{
		// Get the current output information object
		outputFiberInfo info = mapIter.value();

		// Clear and delete the ID list of this fiber
		if (info.idList)
		{
			info.idList->clear();
			delete info.idList;
		}
	}

	// Clear the fiber ID map
	this->fiberIdMap.clear();

	// Delete the arrays
	if (this->inTensorsA)
		delete[] inTensorsA;

	if (this->inTensorsB)
		delete[] inTensorsB;

	if (this->anglePrevPoints)
		delete[] this->anglePrevPoints;

	if (this->distanceArray)
		delete[] this->distanceArray;
}


//---------------------------[ continueTracking ]--------------------------\\

bool vtkFiberTrackingGeodesicFilter::continueTracking(bmia::streamlinePoint * currentPoint, double testDot, vtkIdType currentCellId)
{
	// We don't actually use this function, so just return true
	return true;
}


//-------------------------------[ Execute ]-------------------------------\\

void vtkFiberTrackingGeodesicFilter::Execute()
{
	// Get the output data set
	vtkPolyData * output = this->GetOutput();

	// Get the input tensor image
	this->dtiImageData = (vtkImageData *) (this->GetInput());

	// Check if the tensor image exists
	if (!(this->dtiImageData))
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "No input DTI data defined!", 
			QMessageBox::Ok, QMessageBox::Ok);

		return;
	}

	// Get the point data of the tensor image
	this->dtiPointData = this->dtiImageData->GetPointData();

	// Check if the point data exists
	if (!(this->dtiPointData))
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "No point data for input DTI data!", 
			QMessageBox::Ok, QMessageBox::Ok);

		return;
	}

	// Get the tensors of the tensor image
	this->dtiTensors = this->dtiPointData->GetArray("Tensors");

	// Check if the tensors exist
	if (!(this->dtiTensors))
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "No tensors for input DTI data!", 
			QMessageBox::Ok, QMessageBox::Ok);

		return;
	}

	// Check if the number of tensor components is six
	if (this->dtiTensors->GetNumberOfComponents() != 6)
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "Number of tensor components is not equal to six!", 
			QMessageBox::Ok, QMessageBox::Ok);


		return;
	}

	// Get the Anisotropy Index image
	this->aiImageData  = GetAnisotropyIndexImage();

	// Check if the AI image exists
	if (!(this->aiImageData))
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "No input AI data defined!", 
			QMessageBox::Ok, QMessageBox::Ok);


		return;
	}

	// Get the point data of the AI image
	this->aiPointData = this->aiImageData->GetPointData();

	// Check if the point data exists
	if (!(this->aiPointData))
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "No point data for input AI!", 
			QMessageBox::Ok, QMessageBox::Ok);

		return;
	}

	// Get the scalars of the AI image
	this->aiScalars = this->aiPointData->GetScalars();

	// Check if the scalars exist
	if (!(this->aiScalars))
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "No scalars for input AI data!", 
			QMessageBox::Ok, QMessageBox::Ok);

		return;
	}

	// Get the seed points
	vtkUnstructuredGrid * seedPoints = (vtkUnstructuredGrid *) this->GetSeedPoints();

	// Check if the seed points exist
	if (!seedPoints)
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "No seed points defined!", 
			QMessageBox::Ok, QMessageBox::Ok);

		return;
	}

	// Pre-compute the tolerance variable, used in the "FindCell" functions
	double tolerance = this->dtiImageData->GetLength() / 1000.0;

	// Initialize the output data set
	this->initializeBuildingFibers();

	// Initialize the progress bar
	this->UpdateProgress(0.0);
	this->SetProgressText("Generating seed points...");

	// Reset the seed point counter
	this->seedPointCounter = 0;

	// Loop through all seed points
	for (int ptId = 0; ptId < seedPoints->GetNumberOfPoints(); ++ptId)
	{
		// Current seed point
		double seedPoint[3];

		// Get the coordinates of the current seed point
		seedPoints->GetPoint(ptId, seedPoint);

		// Generate the list of directions for this seed point
		this->generateFiberDirections(seedPoint);
	}

	// Get the image information (spacing and dimension)
	GC_imageInfo imageInfo;

	int dim[3];
	this->dtiImageData->GetDimensions(dim);
	imageInfo.su = dim[0];
	imageInfo.sv = dim[1];
	imageInfo.sw = dim[2];

	double spacing[3];
	this->dtiImageData->GetSpacing(spacing);
	imageInfo.du = spacing[0];
	imageInfo.dv = spacing[1];
	imageInfo.dw = spacing[2];

	// Create a struct for the tracking parameters
	trackingParameters trP;
	trP.loadSize = this->numberOfFibersPerLoad;
	trP.maxIter  = this->maxNumberOfSteps;

	// Try to initialize the GPU for CUDA
	if (!(GC_GPU_Init(imageInfo, trP)))
	{
		vtkErrorMacro(<< "Failed to initialize CUDA");
		GC_GPU_CleanUpPP();
		return;
	}

	// Preprocess the tensors and compute the derivatives
	if (!(this->preProcessTensors(imageInfo)))
	{
		vtkErrorMacro(<< "Failed to preprocess tensor data!");
		GC_GPU_CleanUpPP();
		return;
	}

	// Set string for the progress bar
	std::string progressText = "Tracking fibers for ROI '" + this->roiName.toStdString() + "'...";

	// Initialize the progress bar
	this->UpdateProgress(0.0);
	this->SetProgressText((char *) progressText.c_str());

	// Create an input information array. For each seed point, this array contains
	// the fiber ID of the resulting fiber, and its direction. The direction can
	// be 1 or -1; for each fiber ID, we have two seed points, one for each direction.

	inputFiberInfo * info = new inputFiberInfo[this->numberOfFibersPerLoad];

	// Create the seed point array. This contains the initial position and direction
	GC_fiberPoint * seeds = new GC_fiberPoint[this->numberOfFibersPerLoad];

	// Create an output array for the fibers. During each 'load', we compute at
	// most 'maxNumberOfSteps' steps per fiber, for 'numberOfFibersPerLoad' seed
	// points; therefore, the output array is of constant size.

	float * fiberPoints = new float[this->numberOfFibersPerLoad * this->maxNumberOfSteps * 3];

	// If we want to use the angle stopping criterion, we first need to allocate
	// a fiber point array for it. This array will contain, for each fiber that
	// has not yet terminated, the point coordinates of the last fiber point, and
	// the normalized last segment. When postprocessing the next load, these 
	// values can be used to check the angle between the last segment of the 
	// previous part, and the first segment of the new part.

	if (this->useStopAngle)
		this->anglePrevPoints = new GC_fiberPoint[this->numberOfFibersPerLoad];
	else
		this->anglePrevPoints = NULL;

	// If we want to use the maximum fiber distance stopping criterion, we first
	// need to allocate an array that can contain the current length for each
	// fiber. For each fiber that has not yet terminated, this array will hold
	// the current length. When postprocessing the next load of fibers, these 
	// values can be used as starting values for computing the total length.

	if (this->useStopLength)
		this->distanceArray = new float[this->numberOfFibersPerLoad];
	else
		this->distanceArray = NULL;

	// If we want to use the scalar threshold stopping criterion, we first need
	// to allocate an array for these scalar. We simply copy all scalar values
	// from the selected scalar image to this array.

	if (this->useStopScalar)
	{
		this->scalarArray = new float[aiScalars->GetNumberOfTuples()];
		for (int i = 0; i < aiScalars->GetNumberOfTuples(); ++i)
			this->scalarArray[i] = (float) aiScalars->GetTuple1(i);
	}
	else
		this->scalarArray = NULL;

	// Initialize all fiber information objects
	for (int i = 0; i < this->numberOfFibersPerLoad; ++i)
	{
		info[i].dir = 1;
		info[i].id = -1;
	}

	// A 'load' is a set number of fibers (usually 4096), each being computed for
	// a fixed maximum amount of steps (usually 1024). If a fiber has not yet
	// terminated after these steps, it writes it current position and direction
	// to the seed point array; during the next load, computation will be resumed
	// at that point. If a fiber does terminate, its seed point will be replaced
	// by one from the list of seed points before the next load. We limit the 
	// number of loads that can be processed to avoid excessive processing times.

	int maxNumberOfLoads = (int) ceil((float) this->seedPointList.size() / (float) this->numberOfFibersPerLoad) * 10;

	// Compute at most 'maxNumberOfLoads' loads
	for (int loadId = 0; loadId < maxNumberOfLoads; ++loadId)
	{
		// Update the progress, based on the load ID
		this->UpdateProgress((double) loadId / (double) maxNumberOfLoads);

		// Fill up the seed point array. Invalid seed points (i.e., points with
		// the ID equal to -1) will be replaced by new points from the seed
		// point list (if it is non-empty).

		this->fillSeedPointArray(info, seeds);

		// Track the fibers from the current set of seed points
		if (!this->trackFibers(imageInfo, seeds, fiberPoints))
		{
			vtkErrorMacro(<< "Error tracking fibers!");
		}


		// POSTPROCESSING STAGE: One or more postprocessing kernels are called,
		// which can terminate fibers based on different stopping criteria. The
		// last argument of each function call specifies if this is the last
		// postprocessing kernel; the postprocessed fibers will only be loaded
		// back to the CPU after the last kernel. If the order of these kernels
		// is changed, the "lastPostProcess" value should change too.


		// If the maximum fiber distance stopping criterion is DISABLED, we launch
		// the mobility kernel, which is used to detect and terminate looping 
		// fibers. We use five times the cell diagonal as the threshold; in other
		// words, if the distances between the first, middle and last point of the
		// fiber part are all less than five times the diagonal, we assume that the
		// fiber is looping (or has stopped completely), and we terminate it.

		if (!(this->useStopLength))
		{
			if (!(GC_GPU_StopMobility(	(GC_outBuffer *) fiberPoints, 
										5.0f * sqrtf(spacing[0] * spacing[0] + spacing[1] * spacing[1] + spacing[2] * spacing[2]), 
										trP, 
										!(this->useStopAngle || this->useStopLength || this->useStopScalar))))
			{
				vtkErrorMacro(<< "Failed to apply mobility stopping criterion!");
			}
		}

		// Terminate fibers if the length exceeds the specified maximum.

		if (this->useStopLength && this->distanceArray)
		{
			if (!(GC_GPU_StopDistance(	(GC_outBuffer *) fiberPoints, 
										this->distanceArray, 
										this->MaximumPropagationDistance, 
										trP, 
										!(this->useStopAngle || this->useStopScalar))))
			{
				vtkErrorMacro(<< "Failed to apply length stopping criterion!");
			}
		}

		// Terminate fibers if an angle sharper than the specified maximum is encountered.

		if (this->useStopAngle && this->anglePrevPoints)
		{
			if (!(GC_GPU_StopAngle(		this->anglePrevPoints, 
										(GC_outBuffer *) fiberPoints, 
										this->StopDotProduct, 
										trP, 
										!(this->useStopScalar))))
			{
				vtkErrorMacro(<< "Failed to apply angle stopping criterion!");
			}
		}

		// Terminate fibers if the local scalar value is out of range.

		if (this->useStopScalar && this->scalarArray)
		{
			if (!(GC_GPU_StopScalar(	(GC_outBuffer *) fiberPoints, 
										this->scalarArray, 
										imageInfo, 
										this->MinScalarThreshold, 
										this->MaxScalarThreshold, 
										trP, 
										true)))
			{
				vtkErrorMacro(<< "Failed to apply scalar stopping criterion!");
			}
		}

		// Add completed fibers to the output. Loops through all fiber points of 
		// the load, and adds them to the output points. If an invalid point is
		// discovered, the fiber is terminated, and its seed point is invalidated
		// (so that it will be replaced with a new one during the next load). If
		// both fibers for a fiber ID (positive and negative direction) have 
		// terminated, the length of the fiber is checked against the minimum
		// length, and if it passes this test, the fiber is actually added to
		// the output.

		this->addFibersToOutput(info, fiberPoints);

		// Check if we've got any seed points left
		if (!(this->fibersLeft(info)))
			break;
	}

	// Clean up the mess we've made
	GC_GPU_CleanUpPP();
	GC_GPU_CleanUpTR();

	// Exit CUDA. If we do not call this function, computing the fibers a second 
	// time will result in a number of errors (due to occupied resources).

	cudaThreadExit();

	// Done, finalize the progress bar
	this->UpdateProgress(1.0);

	// Squeeze the output to regain over-allocated memory
	output->Squeeze();

	// Delete arrays
	delete[] info;
	delete[] seeds;
	delete[] fiberPoints;

	// Delete optional arrays
	if (this->anglePrevPoints)
	{
		delete[] this->anglePrevPoints;
		this->anglePrevPoints = NULL;
	}

	if (this->distanceArray)
	{
		delete[] this->distanceArray;
		this->distanceArray = NULL;
	}

	if (this->scalarArray)
	{
		delete[] this->scalarArray;
		this->scalarArray = NULL;
	}
}


//--------------------------[ preProcessTensors ]--------------------------\\

bool vtkFiberTrackingGeodesicFilter::preProcessTensors(GC_imageInfo grid)
{
	// Delete existing tensor arrays
	if (this->inTensorsA)
		delete[] inTensorsA;

	if (this->inTensorsB)
		delete[] inTensorsB;

	// Get the number of tensors
	int arraySize = this->dtiTensors->GetNumberOfTuples();

	// Allocate room for the tensor fields
	this->inTensorsA = new float4[arraySize];
	this->inTensorsB = new float4[arraySize];

	// Loop through all tensors
	for (int i = 0; i < arraySize; ++i)
	{
		// Get the current tensor
		double tensor[6];
		this->dtiTensors->GetTuple(i, tensor);

		// Copy the six tensor elements to the "float4" structs
		this->inTensorsA[i].x = tensor[0];
		this->inTensorsA[i].y = tensor[1];
		this->inTensorsA[i].z = tensor[2];
		this->inTensorsA[i].w = tensor[3];
		this->inTensorsB[i].x = tensor[4];
		this->inTensorsB[i].y = tensor[5];

		// AI scalar value
		double scalar;

		// Get the current scalar, if it is within range. This should always be the case
		// if the DTI tensor has the same dimensions as the scalar image.

		if (i < this->aiScalars->GetNumberOfTuples())
			scalar = this->aiScalars->GetTuple1(i);
		else
			scalar = 1.0;

		// Copy the scalar to the last element of the second four-tuple.
		this->inTensorsB[i].w = scalar;

	} // for [all tensors]

	// Create a structure with the preprocessing options
	GC_ppParameters pp;
	pp.exponent			= this->ppExponent;
	pp.gain				= this->ppGain;
	pp.threshold		= this->ppThreshold;
	pp.sharpeningMethod = (int) this->ppSharpenMethod;

	// Preprocess the tensors on the GPU
	bool ppResult =  GC_GPU_PreProcessing(grid, pp, this->inTensorsA, this->inTensorsB, this->ppEnable);

	// Delete the two arrays
	delete[] this->inTensorsA;
	delete[] this->inTensorsB;

	this->inTensorsA = NULL;
	this->inTensorsB = NULL;

	// Return the result of the preprocessing stage
	return ppResult;
}


//-----------------------------[ trackFibers ]-----------------------------\\

bool vtkFiberTrackingGeodesicFilter::trackFibers(GC_imageInfo grid, GC_fiberPoint * seeds, float * outFibers)
{
	// Compute the actual step size using the cell size
	vtkCell * firstCell = this->dtiImageData->GetCell(0);
	float step = this->IntegrationStepLength * sqrt((double) firstCell->GetLength2());

	// Create a tracking parameters struct
	trackingParameters trP;
	trP.loadSize = this->numberOfFibersPerLoad;
	trP.maxIter  = this->maxNumberOfSteps;

	// Perform fiber tracking on the GPU
	return GC_GPU_Tracking(grid, step, trP, seeds, (GC_outBuffer *) outFibers);
}


//--------------------------[ fillSeedPointArray ]-------------------------\\

void vtkFiberTrackingGeodesicFilter::fillSeedPointArray(inputFiberInfo * info, GC_fiberPoint * seeds)
{
	// Loop through all entries of the seed point array
	for (int i = 0; i < this->numberOfFibersPerLoad; ++i)
	{
		// Check if the current seed point is invalid
		if (info[i].id == -1)
		{
			// If the seed list is empty, fully invalidate the seed point
			if (this->seedPointList.isEmpty())
			{
				seeds[i].x.u = -1.0f;
				seeds[i].x.v = -1.0f;
				seeds[i].x.w = -1.0f;
				seeds[i].d.u =  0.0f;
				seeds[i].d.v =  0.0f;
				seeds[i].d.w =  0.0f;
				info[i].dir = 1;
				info[i].id  = -1;
			}

			// Otherwise, add a single seed point to the array
			else
			{
				geodesicSeedPoint newSeed = this->seedPointList.takeFirst();
				seeds[i].x.u = newSeed.px;
				seeds[i].x.v = newSeed.py;
				seeds[i].x.w = newSeed.pz;
				seeds[i].d.u = newSeed.dx;
				seeds[i].d.v = newSeed.dy;
				seeds[i].d.w = newSeed.dz;
				info[i].dir = newSeed.dir;
				info[i].id  = newSeed.id;
			}

			// If we're using the angle stopping criterion, set the X-component of
			// the angle information to -1001.0f. This indicates that this is the 
			// first part of this fiber processed by the angle kernel.

			if (this->anglePrevPoints)
			{
				this->anglePrevPoints[i].x.u = -1001.0f;
			}

			// If we're using the maximum fiber length stopping criterion, set the
			// total distance to zero (since we've either got a new fiber, or an
			// invalid seed point).

			if (this->distanceArray)
			{
				this->distanceArray[i] = 0.0f;
			}
		}
	}
}


//------------------------------[ fibersLeft ]-----------------------------\\

bool vtkFiberTrackingGeodesicFilter::fibersLeft(inputFiberInfo * info)
{
	// If the seed list still contains items, we've got fibers left
	if (this->seedPointList.size() > 0)
		return true;

	// Loop through all seed points, and return true if a valid point is found
	for (int i = 0; i < this->numberOfFibersPerLoad; ++i)
	{
		if (info[i].id != -1)
			return true;
	}

	// Empty seed list and no valid seed points - we're done!
	return false;
}


//-----------------------[ generateFiberDirections ]-----------------------\\

void vtkFiberTrackingGeodesicFilter::generateFiberDirections(double * p)
{
	// Get the index of the voxel closest to the seed point
	vtkIdType pointId = this->dtiImageData->FindPoint(p[0], p[1], p[2]);

	// Return if the point is not located inside the volume
	if (pointId < 0)
		return;

	// DTI tensor of the closest voxel
	double dtiTensor[6];

	// Eigenvectors and -values
	double V[9];
	double W[3];

	// Get the DTI tensor, and compute its eigensystem
	this->dtiTensors->GetTuple(pointId, dtiTensor);
	vtkTensorMath::EigenSystemSorted(dtiTensor, V, W);
	
	// Normalize the main eigenvector
	double tempDir[3] = {V[0], V[1], V[2]};
	vtkMath::Normalize(tempDir);

	// Copy the main eigenvector to a new double array
	geodesicSeedPoint newSeed;
	newSeed.px = p[0];
	newSeed.py = p[1];
	newSeed.pz = p[2];

	// The seed point counter is incremented once for every pair of seed points
	// (positive and negative directions). 

	newSeed.id = this->seedPointCounter++;

	// For each seed point, we first add the positive direction...
	newSeed.dx = tempDir[0];
	newSeed.dy = tempDir[1];
	newSeed.dz = tempDir[2];
	newSeed.dir = 1;

	this->seedPointList.append(newSeed);

	// ...and then the negative direction. Both seed points have the same fiber ID.
	newSeed.dx = -tempDir[0];
	newSeed.dy = -tempDir[1];
	newSeed.dz = -tempDir[2];
	newSeed.dir = -1;

	this->seedPointList.append(newSeed);

	// If we don't want additional angles, we're done here
	if (!(this->useAdditionAngles))
		return;

	// Cone around the Main Eigenvector
	if (this->aaPattern == AAP_Cone)
	{
		// Angle between additional rays, distributed evenly over 0-2PI around main eigenvector
		float dTheta = (2 * vtkMath::Pi()) / this->aaConeNumberOfAngles;

		// Starting angle
		float theta = 0.0;

		// Loop through all the extra angles
		for (int dirId = 0; dirId < this->aaConeNumberOfAngles; ++dirId)
		{
			// New values for the main (largest) eigenvector
			double newEigenVector0[3];

			// Compute the new main eigenvector
			newEigenVector0[0] = V[0] + cos(theta) * this->aaConeWidth * V[3] + sin(theta) * this->aaConeWidth * V[6];
			newEigenVector0[1] = V[1] + cos(theta) * this->aaConeWidth * V[4] + sin(theta) * this->aaConeWidth * V[7];
			newEigenVector0[2] = V[2] + cos(theta) * this->aaConeWidth * V[5] + sin(theta) * this->aaConeWidth * V[8];

			// Normalize the new direction vector
			vtkMath::Normalize(newEigenVector0);

			// Increment angle
			theta += dTheta;

			newSeed.dx = newEigenVector0[0];
			newSeed.dy = newEigenVector0[1];
			newSeed.dz = newEigenVector0[2];
			newSeed.id = this->seedPointCounter++;
			newSeed.dir = 1;

			this->seedPointList.append(newSeed);

			newSeed.dx = -newEigenVector0[0];
			newSeed.dy = -newEigenVector0[1];
			newSeed.dz = -newEigenVector0[2];
			newSeed.dir = -1;

			this->seedPointList.append(newSeed);
		}

		return;
	}

	// Simple Sphere
	if (this->aaPattern == AAP_SimpleSphere)
	{
		// Define angles and angle steps. Phi rotates in the plane defined by the second and
		// third eigenvector; Theta angles from this plane (0) up to the main eigenvector (90).
		float dPhi   = (2.0f * vtkMath::Pi()) / this->aaSphereNumberOfAnglesP;
		float dTheta = (0.5f * vtkMath::Pi()) / this->aaSphereNumberOfAnglesT;
		float phi    = 0.0f;
		float theta  = 0.0f;

		// Loop through all values for phi
		for (int iPhi = 0; iPhi < this->aaSphereNumberOfAnglesP; iPhi++)
		{
			theta = 0.0f;

			// Loop through all values for theta
			for (int iTheta = 0; iTheta < this->aaSphereNumberOfAnglesT; iTheta++)
			{
				// New main eigenvector, which determines the initial direction
				double newEigenVector0[3];

				// Compute the new direction.
				newEigenVector0[0] = sin(theta) * V[0] + cos(phi) * V[3] + sin(phi) * V[6];
				newEigenVector0[1] = sin(theta) * V[1] + cos(phi) * V[4] + sin(phi) * V[7];
				newEigenVector0[2] = sin(theta) * V[2] + cos(phi) * V[5] + sin(phi) * V[8];

				// Normalize the new direction vector
				vtkMath::Normalize(newEigenVector0);

				// Increment theta
				theta += dTheta;

				newSeed.dx = newEigenVector0[0];
				newSeed.dy = newEigenVector0[1];
				newSeed.dz = newEigenVector0[2];
				newSeed.id = this->seedPointCounter++;
				newSeed.dir = 1;

				this->seedPointList.append(newSeed);

				newSeed.dx = -newEigenVector0[0];
				newSeed.dy = -newEigenVector0[1];
				newSeed.dz = -newEigenVector0[2];
				newSeed.dir = -1;

				this->seedPointList.append(newSeed);
			}

			// Increment phi
			phi += dPhi;
		}

		return;
	}

	// Tessellated icosahedron
	if (this->aaPattern == AAP_Icosahedron)
	{
		// Create a tessellator, and tessellate an icosahedron using the specified order
		Visualization::sphereTesselator<double> * tess = new Visualization::sphereTesselator<double>(Visualization::icosahedron);
		tess->tesselate(this->aaIcoTessOrder);

		// Get the tessellated sphere as VTK polydata
		vtkPolyData * tessSphere = vtkPolyData::New(); 
		tess->getvtkTesselation(true, tessSphere);

		// We're done with the tessellator, so delete it
		delete tess;

		// Get the number of tessellation point
		int numberOfTessPoints = tessSphere->GetNumberOfPoints();

		// Loop through all points in the tessellated sphere
		for (int i = 0; i < numberOfTessPoints; ++i)
		{
			// Get the point coordinates
			double v[3];
			tessSphere->GetPoint(i, v);
			
			newSeed.dx = v[0];
			newSeed.dy = v[1];
			newSeed.dz = v[2];
			newSeed.id = this->seedPointCounter++;
			newSeed.dir = 1;

			this->seedPointList.append(newSeed);

			newSeed.dx = -v[0];
			newSeed.dy = -v[1];
			newSeed.dz = -v[2];
			newSeed.dir = -1;

			this->seedPointList.append(newSeed);
		}

		return;
	}
}


//--------------------------[ addFibersToOutput ]--------------------------\\

void vtkFiberTrackingGeodesicFilter::addFibersToOutput(inputFiberInfo * info, float * fiberPoints)
{
	// Get the points and lines of the output
	vtkPoints * outputPoints = this->GetOutput()->GetPoints();

	// Index of newly added points
	vtkIdType newPointId;

	// Loop through all fibers in the current load
	for (int fiberCounter = 0; fiberCounter < this->numberOfFibersPerLoad; ++fiberCounter)
	{
		// Get the ID and direction of the fiber
		int fiberInputId  = info[fiberCounter].id;
		int fiberDir      = info[fiberCounter].dir;

		outputFiberInfo outputInfo;

		// If the current fiber has not yet been added to the output...
		if (!(this->fiberIdMap.contains(fiberInputId)))
		{
			// ...create a new list and initialize "finishedDirs" to zero...
			outputInfo.finishedDirs = 0;
			outputInfo.idList = new QList<int>;

			// ...and add this information to the map.
			this->fiberIdMap.insert(fiberInputId, outputInfo);
		}

		// If the fiber already exists, just get the existing information.
		else
		{
			outputInfo = this->fiberIdMap.value(fiberInputId);
		}

		// Get the current list of IDs
		QList<int> * currentIdList = outputInfo.idList;

		// Loop through all points in the array
		for (int pointCounter = 0; pointCounter < this->maxNumberOfSteps; ++pointCounter)
		{
			// Compute the base index of the 3-tuple containing the point coordinates
			int baseIndex = (fiberCounter * this->maxNumberOfSteps + pointCounter) * 3;

			// If all three coordinates are less than -1000, we've reached the end of the fiber
			if (fiberPoints[baseIndex] < -1000.0f && fiberPoints[baseIndex + 1] < -1000.0f && fiberPoints[baseIndex + 2] < -1000.0f)
			{
				// Set the input index to -1, so that it will be replaced with a new seed point
				info[fiberCounter].id = -1;

				// Increment the number of finished directions
				outputInfo.finishedDirs++;

				// Break here; all other points will be invalid
				break;
			}

			// Copy the point coordinates to a double array
			double newPoint[3] = {fiberPoints[baseIndex], fiberPoints[baseIndex + 1], fiberPoints[baseIndex + 2]};

			// Add the coordinates to the point set of the output
			newPointId = outputPoints->InsertNextPoint(newPoint);

			// Append or prepend the point, based on the direction
			if (fiberDir == 1)
				currentIdList->append(newPointId);
			else
				currentIdList->prepend(newPointId);

		} // for [pointCounter]

		// If we've got two finished directions, write the fiber to the output
		if (outputInfo.finishedDirs == 2)
		{
			this->addSingleFiberToOutput(outputInfo.idList);

			// Clear and delete the ID list
			outputInfo.idList->clear();
			delete outputInfo.idList;

			// Remove the entry from the map
			this->fiberIdMap.remove(fiberInputId);
		}

		// Otherwise, overwrite the existing output information object in the map
		else
		{
			this->fiberIdMap.insert(fiberInputId, outputInfo);
		}

	} // for [fiberCounter]

}


//------------------------[ addSingleFiberToOutput ]-----------------------\\

void vtkFiberTrackingGeodesicFilter::addSingleFiberToOutput(QList<int> * idList)
{
	// Do nothing if the list is empty
	if (idList->isEmpty())
		return;

	// Get the output lines
	vtkCellArray * outputLines = this->GetOutput()->GetLines();

	// If the minimum fiber size is positive, we check for it here
	if (this->MinimumFiberSize > 0.0)
	{
		vtkPoints * outputPoints = this->GetOutput()->GetPoints();

		double D = 0.0;			// Current fiber length
		double xCurrent[3];		// Current fiber point
		double xPrev[3];		// Previous fiber point

		// Get the first point
		outputPoints->GetPoint(idList->at(0), xPrev);

		// Loop through all fiber points
		for (QList<int>::iterator i = idList->begin(); i != idList->end(); ++i)
		{
			// Add segment length to the distance
			outputPoints->GetPoint(*i, xCurrent);
			D += sqrtf(vtkMath::Distance2BetweenPoints(xCurrent, xPrev));

			// Stop if the minimum length was exceeded
			if (D >= this->MinimumFiberSize)
				break;

			// Update the previous point
			xPrev[0] = xCurrent[0];
			xPrev[1] = xCurrent[1];
			xPrev[2] = xCurrent[2];
		}

		// Do not add the fiber if it is too short
		if (D < this->MinimumFiberSize)
			return;
	}

	// Create and allocate a new ID list
	vtkIdList * idListCopy = vtkIdList::New();
	idListCopy->Allocate(idList->size());

	// Copy all IDs to the new list
	for (QList<int>::iterator i = idList->begin(); i != idList->end(); ++i)
	{
		idListCopy->InsertNextId(*i);
	}

	// Insert the new ID list into the lines array
	outputLines->InsertNextCell(idListCopy);

	// Reset and delete the ID list
	idListCopy->Reset();
	idListCopy->Delete();
}


} // namespace bmia
