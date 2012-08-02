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
 */


/** Includes */

#include "vtkFiberTrackingGeodesicFilter.h"
#include "geodesicFiberTracker.h"


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
	this->performanceProfile	= PERF_NoProcomputation;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkFiberTrackingGeodesicFilter::~vtkFiberTrackingGeodesicFilter()
{
	// Clear direction list
	for (int i = 0; i < this->dirList.size(); ++i)
	{
		double * currentDir = this->dirList.at(i);
		delete[] currentDir;
	}

	this->dirList.clear();
}


//---------------------------[ continueTracking ]--------------------------\\

bool vtkFiberTrackingGeodesicFilter::continueTracking(bmia::streamlinePoint * currentPoint, double testDot, vtkIdType currentCellId)
{
	// Fiber has left the volume	
	if (currentCellId < 0)
		return false;

	// Maximum fiber length exceeded
	if (this->useStopLength && currentPoint->D > this->MaximumPropagationDistance)
		return false;

	// Maximum fiber angle exceeded
	if (this->useStopAngle && testDot < (double) this->StopDotProduct)
		return false;

	// Scalar value out of range
	if (this->useStopScalar && (currentPoint->AI > this->MaxScalarThreshold || currentPoint->AI < this->MinScalarThreshold))
		return false;

	return true;
}


//---------------------------[ initializeFiber ]---------------------------\\

bool vtkFiberTrackingGeodesicFilter::initializeFiber(double * seedPoint)
{
	// Clear the positive and negative point lists
	streamlinePointListPos.clear();
	streamlinePointListNeg.clear();

	// Create a new streamline point
	streamlinePoint seedSLPoint;

	// Set the coordinates of the seed point
	seedSLPoint.X[0] = seedPoint[0];
	seedSLPoint.X[1] = seedPoint[1];
	seedSLPoint.X[2] = seedPoint[2];

	// Set all other point values to zero
	seedSLPoint.V0[0] = 0.0;
	seedSLPoint.V0[1] = 0.0;
	seedSLPoint.V0[2] = 0.0;
	seedSLPoint.V1[0] = 0.0;
	seedSLPoint.V1[1] = 0.0;
	seedSLPoint.V1[2] = 0.0;
	seedSLPoint.V2[0] = 0.0;
	seedSLPoint.V2[1] = 0.0;
	seedSLPoint.V2[2] = 0.0;
	seedSLPoint.AI	  = 0.0;
	seedSLPoint.D     = 0.0;

	// Get the next direction from the direction list
	double * currentDir = this->dirList.takeFirst();

	// Copy the direction to the seed point
	seedSLPoint.dX[0] = currentDir[0];
	seedSLPoint.dX[1] = currentDir[1];
	seedSLPoint.dX[2] = currentDir[2];

	// Allocate space for the lists
	streamlinePointListPos.reserve(1000);
	streamlinePointListNeg.reserve(1000);

	// Add the first point to the positive list
	streamlinePointListPos.push_back(seedSLPoint);

	// Flip the direction
	seedSLPoint.dX[0] *= -1.0;
	seedSLPoint.dX[1] *= -1.0;
	seedSLPoint.dX[2] *= -1.0;

	// Add the point, with the flipped direction, to the negative list
	streamlinePointListNeg.push_back(seedSLPoint);

	// Delete the direction vector
	delete[] currentDir;

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

	// Create a new pre-processor object
	geodesicPreProcessor * pp = new geodesicPreProcessor;

	// Try to set the input DTI image
	if (!(pp->setInputImage(this->dtiImageData)))
	{
		QMessageBox::warning(NULL, "Fiber Tracking Filter", "Failed to set the DTI image of the pre-processor!", 
			QMessageBox::Ok, QMessageBox::Ok);

		return;
	}

	// If we're using scalars as a stopping criteria, try to set the scalar image
	if (this->useStopScalar)
	{
		if (!(pp->setScalarImage(this->aiImageData)))
		{
			QMessageBox::warning(NULL, "Fiber Tracking Filter", "Failed to set the scalar image of the pre-processor!", 
				QMessageBox::Ok, QMessageBox::Ok);

			return;
		}
	}

	// Set the pre-processing parameters
	pp->setEnable(this->ppEnable);
	pp->setSharpenExponent(this->ppExponent);
	pp->setSharpeningMethod(this->ppSharpenMethod);
	pp->setSharpenThreshold(this->ppThreshold);
	pp->setTensorGain(this->ppGain);

	// Preprocess and/or invert all tensors, if required
	if (this->performanceProfile == PERF_PreProcessAll || this->performanceProfile == PERF_PreProcessAndInvertAll)
		pp->preProcessFullImage();
	if (this->performanceProfile == PERF_PreProcessAndInvertAll)
		pp->invertFullImage();

	// Create the tracker
	geodesicFiberTracker * tracker = new geodesicFiberTracker;

	// Initialize pointers and parameters of the tracker
	tracker->initializeTracker(		this->dtiImageData, this->aiImageData, 
									this->dtiTensors,	this->aiScalars, 
									this, this->IntegrationStepLength, tolerance	);

	// Set the pre-processor pointer
	tracker->setPreProcessor(pp);

	// Set the ODE solver
	tracker->setSolver(this->myODESolver);

	// Initialize the output data set
	this->initializeBuildingFibers();

	// The number of fibers per seed point. We've always got at least one fiber (MEV direction)
	int numberOfFibersPerSeedPoint = 1;

	// Check if we're using additional angles
	if (this->useAdditionAngles)
	{
		// For the cone pattern, the number of additional angles is directly defined
		if (this->aaPattern == AAP_Cone)
		{
			numberOfFibersPerSeedPoint += this->aaConeNumberOfAngles;
		}

		// For the simple sphere, we get the number of additional angles by multiplying
		// the number of angles for phi by the number for theta.

		else if (this->aaPattern == AAP_SimpleSphere)
		{
			numberOfFibersPerSeedPoint += this->aaSphereNumberOfAnglesP * this->aaSphereNumberOfAnglesT;
		}

		// The number of additional angles for the icosahedron depends on the tessellation order
		else if (this->aaPattern == AAP_Icosahedron)
		{
			switch (this->aaIcoTessOrder)
			{
				case 1:		numberOfFibersPerSeedPoint += 12;		break;
				case 2:		numberOfFibersPerSeedPoint += 42;		break;
				case 3:		numberOfFibersPerSeedPoint += 162;		break;
				case 4:		numberOfFibersPerSeedPoint += 642;		break;
				case 5:		numberOfFibersPerSeedPoint += 2562;		break;
				case 6:		numberOfFibersPerSeedPoint += 10242;	break;
				default:	break;
			}
		}
	}

	// Compute how many fibers we're going to compute in total (used for the progress bar)
	int totalNumberOfFibers = numberOfFibersPerSeedPoint * seedPoints->GetNumberOfPoints();

	// Progress bar is updated once every "progressStepSize" seed points
	int progressStepSize = totalNumberOfFibers / 100;

	// Step size needs to be at least one
	if (progressStepSize == 0)
		progressStepSize = 1;

	// Set string for the progress bar
	std::string progressText = "Tracking fibers for ROI '" + this->roiName.toStdString() + "'...";

	// Initialize the progress bar
	this->UpdateProgress(0.0);
	this->SetProgressText((char *) progressText.c_str());

	// Fiber index
	int fiberId = 0;

	// Loop through all seed points
	for (int ptId = 0; ptId < seedPoints->GetNumberOfPoints(); ++ptId)
	{
		// Current seed point
		double seedPoint[3];

		// Get the coordinates of the current seed point
		seedPoints->GetPoint(ptId, seedPoint);

		// Generate the list of directions for this seed point
		this->generateFiberDirections(seedPoint);

		// Process all directions for this seed point
		while (this->dirList.isEmpty() == false)
		{
			// Initialize the fiber of the current seed point using the first direction
			if (!this->initializeFiber(seedPoint))
			{
				continue;
			}

			// Calculate the fiber in positive and negative direction
			tracker->calculateFiber(&streamlinePointListPos);
			tracker->calculateFiber(&streamlinePointListNeg);

			// Get the length of the resulting fiber
			double fiberLength = this->streamlinePointListPos.back().D + this->streamlinePointListNeg.back().D;

			// If the fiber is longer than the minimum length, add it to the output
			if(		(fiberLength > this->MinimumFiberSize) && 
					(this->streamlinePointListPos.size() + this->streamlinePointListNeg.size()) > 2	)
			{
				this->BuildOutput();
			}

			// Update the progress bar
			if ((fiberId % progressStepSize) == 0)
				this->UpdateProgress(fiberId / (float) totalNumberOfFibers);

			// Increment the fiber index
			fiberId++;
		}

	}

	// Delete the tracker
	delete tracker;

	// Squeeze the output to regain over-allocated memory
	output->Squeeze();

	// Clear point lists
	this->streamlinePointListPos.clear();
	this->streamlinePointListNeg.clear();
}


//-----------------------[ generateFiberDirections ]-----------------------\\

void vtkFiberTrackingGeodesicFilter::generateFiberDirections(double * p)
{
	// Clear existing list
	for (int i = 0; i < this->dirList.size(); ++i)
	{
		double * currentDir = this->dirList.at(i);
		delete[] currentDir;
	}

	this->dirList.clear();

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
	double * newDir = new double[3];
	newDir[0] = tempDir[0];
	newDir[1] = tempDir[1];
	newDir[2] = tempDir[2];

	// Add the direction to the list
	this->dirList.append(newDir);

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

			// Copy the direction to a new double array
			double * newDir = new double[3];
			newDir[0] = newEigenVector0[0];
			newDir[1] = newEigenVector0[1];
			newDir[2] = newEigenVector0[2];

			// Increment angle
			theta += dTheta;

			// Add the direction to the list
			this->dirList.append(newDir);
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

				// Copy the direction to a new double array
				double * newDir = new double[3];
				newDir[0] = newEigenVector0[0];
				newDir[1] = newEigenVector0[1];
				newDir[2] = newEigenVector0[2];

				// Increment theta
				theta += dTheta;

				// Add the direction to the list
				this->dirList.append(newDir);
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
			
			// Add the direction to the list
			double * newDir = new double[3];
			newDir[0] = v[0];
			newDir[1] = v[1];
			newDir[2] = v[2];

			this->dirList.append(newDir);
		}

		return;
	}
}


} // namespace bmia
