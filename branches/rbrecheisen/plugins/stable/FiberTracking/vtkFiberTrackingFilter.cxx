/*
 * vtkFiberTrackingFilter.cxx
 *
 * 2010-09-13	Evert van Aart
 * - First version. 
 *
 * 2010-09-15	Evert van Aart
 * - Fixed errors in the code that computes and checks the dot product.
 * - Removed "vtkMath::DegreesToRadians()", replaced by static value.
 * 
 * 2010-09-17	Evert van Aart
 * - Added message boxes for error scenarios.
 * - Added support for the new DTI tensor storage system.
 * - Fiber Tracking is now done in "streamlineTracker". The motivation
 *   for this is that other classes (like "vtkFiberTrackingWVSFilter")
 *   can reuse this class, with custom stopping criteria if needed.
 * - Added basic support for displaying filter progress.
 * - Replaced QLists by std::lists.
 *
 * 2010-09-20	Evert van Aart
 * - Added a progress bar.
 *
 * 2010-09-30	Evert van Aart
 * - Fixed a bug in the "fixVectors" function.
 * - Aligning consecutive line segments now works correctly for second-
 *   order Runge-Kutte solver.
 *
 * 2010-11-10	Evert van Aart
 * - Fixed a bug that caused infinite loops in Whole Volume Seeding.
 *
 * 2011-02-09	Evert van Aart
 * - Added support for maximum scalar threshold values.
 *
 * 2011-08-16	Evert van Aart
 * - Running out of memory when tracking fibers should no longer crash the program.
 *
 */


/** Includes */

#include "vtkFiberTrackingFilter.h"



/** Used to compute the "tolerance" variable */
#define TOLERANCE_DENOMINATOR 1000000



namespace bmia {



vtkStandardNewMacro(vtkFiberTrackingFilter);



//-----------------------------[ Constructor ]-----------------------------\\

vtkFiberTrackingFilter::vtkFiberTrackingFilter()
{
	// Default values for the user processing variables. These should be 
	// overwritten by the current GUI values immediately after creating the class.

	this->MaximumPropagationDistance	= 500.0f;
	this->IntegrationStepLength			=   0.1f;
	this->MinScalarThreshold			=   0.1f;
	this->MaxScalarThreshold			=   1.0f;
	this->StopDegrees					=  45.0f;
	this->MinimumFiberSize				=  10.0f;

	// Default values for the derived processing variables. Actual values
	// will be computed on execution of the filter.

	this->StopDotProduct				=	0.0f;

	// Initialize pointers to NULL

	this->dtiImageData	= NULL;
	this->aiImageData	= NULL;
	this->dtiPointData	= NULL;
	this->aiPointData	= NULL;
	this->aiScalars		= NULL;
	this->dtiTensors	= NULL;

	// Initialize the ROI name
	this->roiName = "Unnamed ROI";

	// Don't stop execution unless something goes seriously wrong
	this->stopExecution = false;
}


//-----------------------------[ Destructor ]------------------------------\\

vtkFiberTrackingFilter::~vtkFiberTrackingFilter()
{
	// Reset pointers to NULL

	this->dtiImageData	= NULL;
	this->aiImageData	= NULL;
	this->dtiPointData	= NULL;
	this->aiPointData	= NULL;
	this->aiScalars		= NULL;
	this->dtiTensors	= NULL;

	// Clear the point lists

	this->streamlinePointListPos.clear();
	this->streamlinePointListNeg.clear();
}


//----------------------------[ SetSeedPoints ]----------------------------\\

void vtkFiberTrackingFilter::SetSeedPoints(vtkDataSet * seedPoints)
{
	// Store the seed point set
	this->vtkProcessObject::SetNthInput(2, seedPoints);
}


//----------------------------[ GetSeedPoints ]----------------------------\\

vtkDataSet * vtkFiberTrackingFilter::GetSeedPoints()
{
	// Return a pointer to the seed point set
	if (this->NumberOfInputs < 3)
	{
		return NULL;
	}

	return (vtkDataSet *) (this->Inputs[2]);
}


//-----------------------[ SetAnisotropyIndexImage ]-----------------------\\

void vtkFiberTrackingFilter::SetAnisotropyIndexImage(vtkImageData * AIImage)
{
	// Store the Anisotropy Index image
	this->vtkProcessObject::SetNthInput(1, AIImage);
}


//-----------------------[ GetAnisotropyIndexImage ]-----------------------\\

vtkImageData * vtkFiberTrackingFilter::GetAnisotropyIndexImage()
{
	// Return a pointer to the Anisotropy Index image
	if (this->NumberOfInputs < 2)
    {
		return NULL;
    }
  
	return (vtkImageData *) (this->Inputs[1]);
}


//--------------------------[ continueTracking ]---------------------------\\

bool vtkFiberTrackingFilter::continueTracking(bmia::streamlinePoint * currentPoint, 
												double testDot, vtkIdType currentCellId)
{
	return (	currentCellId >= 0									&&	// Fiber has left the volume
				currentPoint->D <= this->MaximumPropagationDistance	&&	// Maximum fiber length exceeded
				testDot >= (double) this->StopDotProduct			&&	// Maximum fiber angle exceeded
				currentPoint->AI <= this->MaxScalarThreshold		&&  // High scalar value
				currentPoint->AI >= this->MinScalarThreshold		);	// Low scalar value
}


//---------------------------[ initializeFiber ]---------------------------\\

bool vtkFiberTrackingFilter::initializeFiber(double * seedPoint)
{
	// Clear the positive and negative point lists
	streamlinePointListPos.clear();
	streamlinePointListNeg.clear();

	// Return "false" if the point is not located inside the volume
	if (dtiImageData->FindPoint(seedPoint[0], seedPoint[1], seedPoint[2]) < 0)
	{
		return false;
	}

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

	// Allocate space for the lists
	streamlinePointListPos.reserve(1000);
	streamlinePointListNeg.reserve(1000);

	// Add the first point to both lists
	streamlinePointListPos.push_back(seedSLPoint);
	streamlinePointListNeg.push_back(seedSLPoint);

	return true;
}

//-----------------------[ initializeBuildingFibers ]----------------------\\

void vtkFiberTrackingFilter::initializeBuildingFibers()
{
	// Create and allocate the array containing the fiber points
	vtkPoints *	newPoints = vtkPoints::New();;
	newPoints->Allocate(2500);
 
	// Get the output data set
	vtkPolyData * output = this->GetOutput();

	// Get the point data of the output
    vtkPointData * outPD = output->GetPointData();
  
	// Create and allocate the array containing the output fiber lines.
	// The size estimation used is very rough, but its accuracy doesn't
	// really impact performance.

	vtkCellArray * newFiberLines = vtkCellArray::New();;
	newFiberLines->Allocate(newFiberLines->EstimateSize(2, VTK_CELL_SIZE));

	// Set the output point array, and delete the local reference
	output->SetPoints(newPoints);
	newPoints->Delete();

	// Set the output line array, and delete the local reference
	output->SetLines(newFiberLines);
	newFiberLines->Delete();
}


//------------------------------[ Execute ]--------------------------------\\

void vtkFiberTrackingFilter::Execute()
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
	double tolerance = this->dtiImageData->GetLength() / TOLERANCE_DENOMINATOR;

	// Create the tracker
	streamlineTracker * tracker = new streamlineTracker;

	// Initialize pointers and parameters of the tracker
	tracker->initializeTracker(		this->dtiImageData, this->aiImageData, 
									this->dtiTensors,	this->aiScalars, 
									this, this->IntegrationStepLength, tolerance	);

	// Initialize the output data set
	this->initializeBuildingFibers();

	// Progress bar is updated once every "progressStepSize" seed points
	int progressStepSize = seedPoints->GetNumberOfPoints() / 100;

	// Step size needs to be at least one
	if (progressStepSize == 0)
		progressStepSize = 1;

	// Set string for the progress bar
	std::string progressText = "Tracking fibers for ROI '" + this->roiName.toStdString() + "'...";

	// Initialize the progress bar
	this->UpdateProgress(0.0);
	this->SetProgressText((char *) progressText.c_str());

	// Loop through all seed points
	for (int ptId = 0; ptId < seedPoints->GetNumberOfPoints(); ptId++)
	{
		// Current seed point
		double seedPoint[3];
      
		// Get the coordinates of the current seed point
		seedPoints->GetPoint(ptId, seedPoint);

		// Initialize the fiber of the current seed point
		if (!this->initializeFiber(seedPoint))
		{
			continue;
		}

		// Calculate the fiber in positive and negative direction
		tracker->calculateFiber( 1, &streamlinePointListPos);
		tracker->calculateFiber(-1, &streamlinePointListNeg);

		// Get the length of the resulting fiber
		double fiberLength = this->streamlinePointListPos.back().D + this->streamlinePointListNeg.back().D;
			
		// If the fiber is longer than the minimum length, add it to the output
		if(		(fiberLength > this->MinimumFiberSize) && 
				(this->streamlinePointListPos.size() + this->streamlinePointListNeg.size()) > 2	)
		{
			this->BuildOutput();

			// If something went wrong, we break here
			if (this->stopExecution)
			{
				return;
			}
		}

		// Update the progress bar
		if ((ptId % progressStepSize) == 0)
			this->UpdateProgress(ptId / (float) seedPoints->GetNumberOfPoints());
	}

	// Delete the tracker
	delete tracker;

	// Squeeze the output to regain over-allocated memory
	output->Squeeze();

	// Clear point lists
	this->streamlinePointListPos.clear();
	this->streamlinePointListNeg.clear();
}


//----------------------------[ SetStopDegrees ]---------------------------\\

void vtkFiberTrackingFilter::SetStopDegrees(float StopDegrees)
{
	// Set stop degrees, and compute the dot product threshold
	if(this->StopDegrees != StopDegrees)
	{
		this->StopDegrees = StopDegrees;

		// Convert degrees to radians, and take the cosine
		this->StopDotProduct = cos(0.0174532925f * StopDegrees);
	}
}


//-----------------------------[ BuildOutput ]-----------------------------\\

void vtkFiberTrackingFilter::BuildOutput()
{
	// Get the output of the filter
	vtkPolyData * output = this->GetOutput();

	vtkPoints *		newPoints;			// Points of the fibers
 	vtkIdList *		idFiberPoints;		// List with point IDs
	vtkCellArray *	newFiberLines;		// Fiber lines of the output
	int				id;					// Temporary ID

	// Create and allocate a new ID list
	idFiberPoints = vtkIdList::New();
	idFiberPoints->Allocate(2500);

	// Get the point array
	newPoints = output->GetPoints();

	// Get the lines array
	newFiberLines = output->GetLines();

	// Create a reverse iterator for the point list of the positive direction
	std::vector<streamlinePoint>::reverse_iterator streamlineRIter;

	// Loop through all points in the fiber of the positive direction in reverse
	// order (starting at the last point, ending at the seed point).

	for (streamlineRIter = streamlinePointListPos.rbegin(); streamlineRIter != streamlinePointListPos.rend(); streamlineRIter++)
	{
		// Temporary fiber point
		double tempPoint[3];

		// Get the point coordinates
		tempPoint[0] = (*streamlineRIter).X[0];
		tempPoint[1] = (*streamlineRIter).X[1];
		tempPoint[2] = (*streamlineRIter).X[2];

		try
		{
			// Insert the coordinates in the point array
			id = newPoints->InsertNextPoint(tempPoint);

			// Save the new point ID in the ID list
			idFiberPoints->InsertNextId(id);
		}

		// Out of memory
		catch (vtkstd::bad_alloc)
		{
			vtkErrorMacro(<< "Out of memory! Please consider tracking less fibers.");
			this->stopExecution = true;
			return;
		}
	}

	// Create an iterator for the point list in the negative direction
	std::vector<streamlinePoint>::iterator streamlineIter = streamlinePointListNeg.begin();

	// Increment the iterator to skip the seed point (since we already added it)
	streamlineIter++;

	// Loop through all points in the fiber of the negative direction
	for ( ; streamlineIter != streamlinePointListNeg.end(); streamlineIter++)
	{
		// Temporary fiber point
		double tempPoint[3];

		// Get the point coordinates
		tempPoint[0] = (*streamlineIter).X[0];
		tempPoint[1] = (*streamlineIter).X[1];
		tempPoint[2] = (*streamlineIter).X[2];

		try
		{
			// Insert the coordinates in the point array
			id = newPoints->InsertNextPoint(tempPoint);

			// Save the new point ID in the ID list
			idFiberPoints->InsertNextId(id);
		}

		// Out of memory
		catch (vtkstd::bad_alloc)
		{
			vtkErrorMacro(<< "Out of memory! Please consider tracking less fibers.");
			this->stopExecution = true;
			return;
		}
	}

	// If more than one point was added...
	if (idFiberPoints->GetNumberOfIds() > 1)
	{
		// ...save the list of point IDs as a cell in the lines array...
		newFiberLines->InsertNextCell(idFiberPoints);

		// ...and reset the list
		idFiberPoints->Reset();
	}
   
	// Delete the IDs list
    idFiberPoints->Delete();
}


} // namespace bmia

