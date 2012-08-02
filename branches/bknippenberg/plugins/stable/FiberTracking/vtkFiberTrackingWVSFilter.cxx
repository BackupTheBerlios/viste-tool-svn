/*
 * vtkFiberTrackingWVSFilter.cxx
 *
 * 2010-09-20	Evert van Aart
 * - First version. 
 *
 * 2011-04-26	Evert van Aart
 * - Improved progress reporting.
 * - Slight speed improvements.
 *
 * 2011-06-06	Evert van Aart
 * - Changed the criterion for adding computed fibers to the output; in previous
 *   versions, fibers of length less than the minimum fiber length criterion could
 *   still be added to the output, which was deemed undesirable behavior.
 *
 */


/** Includes */

#include "vtkFiberTrackingWVSFilter.h"


using namespace std;


/** Used to compute the "tolerance" variable */
#define TOLERANCE_DENOMINATOR 1000


namespace bmia {


vtkStandardNewMacro(vtkFiberTrackingWVSFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkFiberTrackingWVSFilter::vtkFiberTrackingWVSFilter()
{
	// Initialize processing variables using default values
	SeedDistance			= 10.0;
	MinDistancePercentage	=  0.5;

	// Set extra seed points to NULL
	extraSeedPoints			= NULL;
}


//-----------------------------[ Destructor ]------------------------------\\

vtkFiberTrackingWVSFilter::~vtkFiberTrackingWVSFilter()
{
	// Remove the extra seed points
	if (this->extraSeedPoints != NULL)
	{
		this->extraSeedPoints->clear();
		delete extraSeedPoints;
	}
} 


//---------------------------[ continueTracking ]--------------------------\\

bool vtkFiberTrackingWVSFilter::continueTracking(streamlinePoint * currentPoint, double testDot, vtkIdType currentCellId)
{
	// Return value
	bool result = false;

	// Check if the current point is not too close to existing fibers
	this->bIsNotTooClose = distanceVolume.exactGoodDistance(currentPoint->X, this->minimumDistance);

	// Call the "continueTracking" function of the parent filter
	result = this->vtkFiberTrackingFilter::continueTracking(currentPoint, testDot, currentCellId) && this->bIsNotTooClose;

	// Return the result
	return result; 
};


//----------------------[ initializeExtraSeedPoints ]----------------------\\

void vtkFiberTrackingWVSFilter::initializeExtraSeedPoints(std::list<initialPoint> * seedPointList)
{
	// Do nothing if no extra seed points have been defined
	if(this->extraSeedPoints)
	{
		// Create an iterator for the list of extra seed point
		VSeedPoints::iterator extraSeedPointIter; 

		// Coordinates of extra seed point
		Point3d currentPoint;

		// New point for the "seedPointList" list
		initialPoint newPoint;

		// Loop through all extra seed points
		for (	extraSeedPointIter  = this->extraSeedPoints->begin(); 
				extraSeedPointIter != this->extraSeedPoints->end(); 
				extraSeedPointIter++									)
		{
			// Get the point's coordinates
			currentPoint = (*extraSeedPointIter);

			// Copy the coordinates to the new point
			newPoint.X[0] = currentPoint.x;
			newPoint.X[1] = currentPoint.y;
			newPoint.X[2] = currentPoint.z;

			// Set the AI to one to ensure maximum priority
			newPoint.AI   = 1.0;

			// Add the extra seed point to the list
			seedPointList->push_back(newPoint);
		}
	}
}


//-------------------------[ createInitialPoints ]-------------------------\\

void vtkFiberTrackingWVSFilter::createInitialPoints(std::list<initialPoint> * seedPointList)
{
	// Position and anisotropy index of grid point
	double X[3];
	double AI;

	// Class containing X and AI
	initialPoint newPoint;

	// Loop through all grid points
	for (vtkIdType ptId = 0; ptId < this->aiImageData->GetNumberOfPoints(); ptId++)
	{
		// Get the coordinates of the point
		this->aiImageData->GetPoint(ptId, X);

		// Get the anisotropy index at the point
		AI = this->aiScalars->GetTuple1(ptId);

		// Only add the point if it exceeds the AI threshold
		if (AI > this->MinScalarThreshold)
		{
			// Copy data to the new point
			newPoint.X[0] = X[0];
			newPoint.X[1] = X[1];
			newPoint.X[2] = X[2];
			newPoint.AI   = AI;

			// Append new element to the list
			seedPointList->push_back(newPoint);
		}
	}

	// Sort the list by descending AI values
	seedPointList->sort();
}


//---------------------------[ addNewSeedPoint ]---------------------------\\

void vtkFiberTrackingWVSFilter::addNewSeedPoint(double * point, queueOfPoints * pointQueue)
{
	// Check if the new position is not too close to existing fibers
	if (!(this->distanceVolume.goodDistance(point)))
		return;

	Point3d newPoint;

	// Copy coordinates to new point
	newPoint.x = point[0];
	newPoint.y = point[1];
	newPoint.z = point[2];

	// Add new point to the queue
	pointQueue->push_back(newPoint);
}


//------------------------[ generateNewSeedPoints ]------------------------\\

void vtkFiberTrackingWVSFilter::generateNewSeedPoints(queueOfPoints * pointQueue, std::vector<streamlinePoint> * streamlinePointList)
{
	// Coordinates of current fiber point
	double currentPoint[3];

	// Reverse iterator for the streamline point list
	std::vector<streamlinePoint>::reverse_iterator streamlineRIter;

	// Loop through all points in the streamline
	for (streamlineRIter = streamlinePointList->rbegin(); streamlineRIter != streamlinePointList->rend(); streamlineRIter++)
	{
		// Current point + Second eigenvector
		currentPoint[0] = (*streamlineRIter).X[0] + (this->SeedDistance + 0.001) * (*streamlineRIter).V1[0];
		currentPoint[1] = (*streamlineRIter).X[1] + (this->SeedDistance + 0.001) * (*streamlineRIter).V1[1];
		currentPoint[2] = (*streamlineRIter).X[2] + (this->SeedDistance + 0.001) * (*streamlineRIter).V1[2];

		// Add to the queue
		this->addNewSeedPoint(currentPoint, pointQueue);

		// Current point + Third eigenvector
		currentPoint[0] = (*streamlineRIter).X[0] + (this->SeedDistance + 0.001) * (*streamlineRIter).V2[0];
		currentPoint[1] = (*streamlineRIter).X[1] + (this->SeedDistance + 0.001) * (*streamlineRIter).V2[1];
		currentPoint[2] = (*streamlineRIter).X[2] + (this->SeedDistance + 0.001) * (*streamlineRIter).V2[2];

		// Add to the queue
		this->addNewSeedPoint(currentPoint, pointQueue);

		// Current point - Second eigenvector
		currentPoint[0] = (*streamlineRIter).X[0] - (this->SeedDistance + 0.001) * (*streamlineRIter).V1[0];
		currentPoint[1] = (*streamlineRIter).X[1] - (this->SeedDistance + 0.001) * (*streamlineRIter).V1[1];
		currentPoint[2] = (*streamlineRIter).X[2] - (this->SeedDistance + 0.001) * (*streamlineRIter).V1[2];

		// Add to the queue
		this->addNewSeedPoint(currentPoint, pointQueue);

		// Current point - Third eigenvector
		currentPoint[0] = (*streamlineRIter).X[0] - (this->SeedDistance + 0.001) * (*streamlineRIter).V2[0];
		currentPoint[1] = (*streamlineRIter).X[1] - (this->SeedDistance + 0.001) * (*streamlineRIter).V2[1];
		currentPoint[2] = (*streamlineRIter).X[2] - (this->SeedDistance + 0.001) * (*streamlineRIter).V2[2];

		// Add to the queue
		this->addNewSeedPoint(currentPoint, pointQueue);
	}
}


//--------------------------[ addFiberToDistance ]-------------------------\\

void vtkFiberTrackingWVSFilter::addFiberToDistance()
{
	// Create iterator for positive streamline
	std::vector<streamlinePoint>::iterator streamlineIter;

	// Loop through all points in the positive streamline
	for (	streamlineIter  = this->streamlinePointListPos.begin();
			streamlineIter != this->streamlinePointListPos.end(); 
			streamlineIter++										)
	{
		// Point coordinates
		double tempX[3];

		// Get the coordinates of the current point
		tempX[0] = (*streamlineIter).X[0];
		tempX[1] = (*streamlineIter).X[1];
		tempX[2] = (*streamlineIter).X[2];

		// Add point to the distance volume
		this->distanceVolume.addPointToDistance(tempX);
	}

	// Create iterator for the negative streamline
	streamlineIter = this->streamlinePointListNeg.begin();

	// Skip first point (since we already added the seed point)
	streamlineIter++;

	// Loop through the points in the negative streamline
	for (	;
			streamlineIter != this->streamlinePointListNeg.end(); 
			streamlineIter++										)
	{
		// Point coordinates
		double tempX[3];

		// Get the coordinates of the current point
		tempX[0] = (*streamlineIter).X[0];
		tempX[1] = (*streamlineIter).X[1];
		tempX[2] = (*streamlineIter).X[2];

		// Add point to the distance volume
		this->distanceVolume.addPointToDistance(tempX);
	}
}



//-------------------------------[ Execute ]-------------------------------\\

void vtkFiberTrackingWVSFilter::Execute()
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

	// Amount of voxels with a scalar value higher than the minimum threshold
	// and lower than the maximum threshold.

	int validVoxels = 0;

	// Compute how many voxel values are in the correct range
	for (int ptId = 0; ptId < this->aiScalars->GetNumberOfTuples(); ++ptId)
	{
		double currentScalar = this->aiScalars->GetTuple1(ptId);
		if (currentScalar > this->MinScalarThreshold && 
			currentScalar < this->MaxScalarThreshold)
			validVoxels++;
	}

	// Compute the percentage of voxels with a value in the correct range. This is
	// actually quite a good measure for the progress of the filter: The Distance
	// volume keeps track of what percentage of its distance elements (which may
	// be of different size than the voxels) contains at least one point, so if this
	// percentage approaches the valid percentage, we're almost done. It's not perfect
	// (sometimes it's done before it reaches 100%, sometimes it takes a bit longer),
	// but it's a decent measure of progress.

	double validPercentage = validVoxels / (double) this->aiScalars->GetNumberOfTuples();

	// Initialize the progress bar
	this->SetProgressText("Whole Volume Seeding - Initializing...");
	this->UpdateProgress(0.0);

	// Pre-compute the tolerance variable, used in the "FindCell" functions
	double tolerance = this->dtiImageData->GetLength() / TOLERANCE_DENOMINATOR;

	// Create the tracker
	streamlineTracker * tracker = new streamlineTracker;

	// Initialize pointers and parameters of the tracker
	tracker->initializeTracker(		this->dtiImageData, this->aiImageData, 
									this->dtiTensors,   this->aiScalars, 
									(vtkFiberTrackingFilter *) this, 
									this->IntegrationStepLength, tolerance	);

	// Initialize the output data set
	this->initializeBuildingFibers();

	// Get dimensions and spacing of the input images
	double * dtiSpacing    = this->dtiImageData->GetSpacing();
	int *    dtiDimensions = this->dtiImageData->GetDimensions();

	// Compute the scaled dimensions of the images
	double scaledDimensions[3];
	scaledDimensions[0] = dtiSpacing[0] * dtiDimensions[0];
	scaledDimensions[1] = dtiSpacing[1] * dtiDimensions[1];
	scaledDimensions[2] = dtiSpacing[2] * dtiDimensions[2];

	// Initialize the distance volume
	this->distanceVolume.initializeVolume(SeedDistance, scaledDimensions);

	// Create a list for the initial set of seed point
	std::list<initialPoint> initialPointsList;

	// Add extra seed points to the initial point list (optional)
	this->initializeExtraSeedPoints(&initialPointsList);

	// Create the initial points, uniformly distributed throughout the volume
	this->createInitialPoints(&initialPointsList);

	// Seed point coordinates
	double seedPoint[3];

	// Currently selected seed point
	initialPoint currentSeedPoint;

	// Pre-compute square of the seed distance and the minimum distance
	this->seedDistanceSquared = this->SeedDistance * this->SeedDistance;
	this->minimumDistance = this->seedDistanceSquared * this->MinDistancePercentage * this->MinDistancePercentage;

	// Queue containing newly added seed points
	queueOfPoints newFiberPoints;

	// Initialize progress
	this->SetProgressText("Whole Volume Seeding - First pass...");
	this->UpdateProgress(0.0);
	bool firstInitialPoint = true;

	// Compute the step size for updating the progress bar (in the second phase)
	int progressStepSizeInitialPoints = (int) (initialPointsList.size() / 25.0);

	if (progressStepSizeInitialPoints == 0)
		progressStepSizeInitialPoints = 1;

	// Create an iterator for the seed point list
	std::list<initialPoint>::iterator initialPointIter = initialPointsList.begin();

	// Index of the current initial point
	int initialPointId = 0;

	// Loop through all initial seed points
	for (	initialPointIter  = initialPointsList.begin();
			initialPointIter != initialPointsList.end();
			++initialPointIter, ++initialPointId	)
	{
		// Get the point coordinates
		currentSeedPoint = (initialPoint) (*initialPointIter);
		seedPoint[0] = currentSeedPoint.X[0];
		seedPoint[1] = currentSeedPoint.X[1];
		seedPoint[2] = currentSeedPoint.X[2];

		// False while there are still (new) seed points to process
		bool bFinishedNewFiberPoints = false;

		// Number of fibers computed for this point
		int numberOfFibers = 0;

		// Continue until no more new seed points exist
		while (!bFinishedNewFiberPoints)
		{
			if (this->distanceVolume.exactGoodDistance(seedPoint, this->seedDistanceSquared))
			{
				// Initialize the fibers of the current seed point
				this->initializeFiber(seedPoint);

				// Check if seed points were correctly added to streamline point lists
				if (this->streamlinePointListPos.size() != 0 && this->streamlinePointListNeg.size() != 0)
				{
					// Set global proximity value to false
					this->bIsNotTooClose = false;

					// Compute streamline in positive direction
					tracker->calculateFiber(1, &(this->streamlinePointListPos));

					// Copy proximity value for first streamline
					bool bIsNotTooClose1 = this->bIsNotTooClose;

					// Reset proximity value
					bIsNotTooClose = false;

					// Compute streamline in negative direction
					tracker->calculateFiber(-1, &(this->streamlinePointListNeg));

					// Compute length of combined fiber
					double fiberLength = this->streamlinePointListPos.back().D + this->streamlinePointListNeg.back().D;

// Evert: The if-statement below was originally used to check whether or not the 
// fibers should be added to the output. However, the way this if-statement is
// written, the "Minimum fiber length" criterion is ignored when a fiber is cut
// off due to proximity to other fibers. Not only does this seem like a strange
// thing to do, it also completely invalidates the "Minimum fiber length" criterion,
// allowing very short fibers to be added to the output. This does not seem like
// desirable behavior (in fact, I got a complaint about this from one of the users),
// so I'm commenting it out and replacing it with a simpler if-statement.

//					if( ( (fiberLength > this->MinimumFiberSize) || !(bIsNotTooClose1) || !(this->bIsNotTooClose) )		
//							&& (this->streamlinePointListPos.size() + this->streamlinePointListNeg.size()) > 2	  )

					// Check if the length of the fiber exceeds the minimum threshold,
					// and if the combined fibers contain at least three points.

					if( (fiberLength > this->MinimumFiberSize) && (this->streamlinePointListPos.size() + this->streamlinePointListNeg.size()) > 2)
					{
						// ...add the fiber points to the distance volume...
						this->addFiberToDistance();

						// ...generate new fiber points from the fiber points...
						this->generateNewSeedPoints(&newFiberPoints, &(this->streamlinePointListPos));
						this->generateNewSeedPoints(&newFiberPoints, &(this->streamlinePointListNeg));

						// ... and add the fiber points to the output.
						this->BuildOutput();
					}
				}
			}

			// If there are new fiber points, get one point from the queue
			if (!newFiberPoints.empty())
			{
				// Get the first point and remove it from the queue
				Point3d aux = newFiberPoints.front();
				newFiberPoints.pop_front();

				// Copy seed point coordinates to the "seedPoint" array
				seedPoint[0] = aux.x;
				seedPoint[1] = aux.y;
				seedPoint[2] = aux.z;
			}

			// If the list is now empty, we're done with the new fiber points
			bFinishedNewFiberPoints = newFiberPoints.empty();

			// Increment the number of fibers
			numberOfFibers++;

			// If this is the first initial point...
			if (firstInitialPoint)
			{
				// ...and the number of fibers computed for this point is a multiple
				// of 500 (to avoid overly frequent updating of the progress bar)...

				if ((numberOfFibers % 500) == 0)
				{
					// ...we update the progress bar, using the number of filled elements
					// reported by the distance volume and the target percentage computed
					// earlier on. The idea here is that most of the fibers are computed 
					// in the very first pass of this loop, since new seed points are 
					// constantly being added through "generateNewSeedPoints", and we won't 
					// move on to the second initial seed point until all of these new
					// seed points have been handled. In this case, the percentage of 
					// filled distance elements turns out to be a good measure of progress,
					// using "validPercentage" to take into account those areas where no
					// fibers will ever be drawn (outside the brain, etc). 

					double newProgress = (double) this->distanceVolume.getPercentageFilled() / validPercentage;

					// Limit the progress to 0.99. It's theoretically possibly for
					// "newProgress" to be 1.0 or higher, since the progress measure is
					// only an estimation. Therefore, we need to limit it.

					if (newProgress >= 1.0)
						newProgress = 0.99;

					this->UpdateProgress(newProgress);
				}
			}
		}

		if (firstInitialPoint)
		{
			// Once we've past the first initial seed point, we're actually almost done.
			// All that's left to do is process the rest of the initial points, 
			// and see if we can still generated fibers from them. So, we create
			// a second pass of the progress bar, and simply use it to track
			// the number of initial seed points processed.

			firstInitialPoint = false;
			this->SetProgressText("Whole Volume Seeding - Processing remaining points...");
			this->UpdateProgress(0.0);
		}
		else if ((initialPointId % progressStepSizeInitialPoints) == 0)
		{
			this->UpdateProgress(initialPointId / (double) initialPointsList.size());
		}

	} // for [all initial points]

	// Done, finalize the progress bar
	this->UpdateProgress(1.0);

	// Delete the tracker
	delete tracker;

	// Squeeze the output to regain over-allocated memory
	output->Squeeze();

	// Clear point lists
	this->streamlinePointListPos.clear();
	this->streamlinePointListNeg.clear();
	initialPointsList.clear();
	newFiberPoints.clear();
}


} // namespace bmia

