/*
 * HARDIdeterministicTracker.cxx
 *
 * 2010-09-17	Evert van Aart
 * - First version. 
 *
 * 2011-10-31 Bart van Knippenberg
 * - Added HARDI fiber tracking functionallity. 
 * - Added class MaximumFinder for finding local maxima on the ODF
 * - Adapted calculateFiber to work with HARDI data
 * - Added function for SH interpolation
 *
 * 2011-11-01 Bart van Knippenberg
 * - Removed a bug that caused good fibers to be deleted
 *
 * 2011-11-05 Bart van Knippenberg
 * - Improved speed by search space reduction for ODF maxima
 * - Improved fiber initialisation
 * - Removed a bug that caused premature tract termination
 * - Added overloaded function getOutput for GFA calculatian at beginning of fiber
 *
 * 2011-11-10 Bart van Knippenberg
 * - Using the radii of the ODF instead of the normalized radii for GFA calculation
 *
 
 */


/** Includes */

#include "HARDIdeterministicTracker.h"

namespace bmia {

//-----------------------------[ Constructor ]-----------------------------\\

HARDIdeterministicTracker::HARDIdeterministicTracker()
{
	// Set pointers to NULL
	this->HARDIimageData	= NULL;
	this->aiImageData		= NULL;
	this->HARDIArray		= NULL;
	this->aiScalars			= NULL;
	this->parentFilter		= NULL;
	this->cellHARDIData		= NULL;
	this->cellAIScalars		= NULL;

	this->unitVectors		= NULL;

	// Set parameters to default values
	this->stepSize			= 0.1;
	this->tolerance			= 1.0;

	//initializations

}


//-----------------------------[ Destructor ]------------------------------\\

HARDIdeterministicTracker::~HARDIdeterministicTracker()
{
	// Set pointers to NULL
	this->HARDIimageData		= NULL;
	this->aiImageData		= NULL;
	this->HARDIArray		= NULL;
	this->aiScalars			= NULL;
	this->parentFilter		= NULL;

	this->unitVectors		= NULL;

	// Delete the cell arrays
	this->cellHARDIData->Delete();
	this->cellAIScalars->Delete();

	// Set the array pointers to NULL
	this->cellHARDIData  = NULL;
	this->cellAIScalars  = NULL;
}

//-----------------------------[ Constructor ]-----------------------------\\

MaximumFinder::MaximumFinder(vtkIntArray* trianglesArray)
{
	
	this->trianglesArray	= trianglesArray;
	this->radii_norm.clear();
	this->radii.clear();
}

//-----------------------------[ Destructor ]------------------------------\\

MaximumFinder::~MaximumFinder()
{
	
	this->trianglesArray	= NULL;
	this->radii_norm.clear();
	this->radii.clear();
}

//--------------------------[ initializeTracker ]--------------------------\\

void HARDIdeterministicTracker::initializeTracker(	vtkImageData *				rHARDIimageData,
													vtkImageData *				rAIImageData,
													vtkDataArray *				rHARDIArray,
													vtkDataArray *				rAIScalars,
													vtkHARDIFiberTrackingFilter *rParentFilter,
													double						rStepSize,
													double						rTolerance		)
{
	// Store input values
	this->HARDIimageData= rHARDIimageData;
	this->aiImageData	=  rAIImageData;
	this->HARDIArray	= rHARDIArray;
	this->aiScalars		=  rAIScalars;	
	this->parentFilter	= rParentFilter;
	this->stepSize		= rStepSize;
	this->tolerance		= rTolerance;

	// Create the cell arrays
	this->cellHARDIData  = vtkDataArray::CreateDataArray(this->HARDIArray->GetDataType());
	this->cellAIScalars  = vtkDataArray::CreateDataArray(this->aiScalars->GetDataType());

	// Set number of components and tuples of the cell arrays
	this->cellHARDIData->SetNumberOfComponents(this->HARDIArray->GetNumberOfComponents());
	this->cellHARDIData->SetNumberOfTuples(8);

	this->cellAIScalars->SetNumberOfComponents(this->aiScalars->GetNumberOfComponents());
	this->cellAIScalars->SetNumberOfTuples(8);
}


//----------------------------[ calculateFiber ]---------------------------\\

void HARDIdeterministicTracker::calculateFiber(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,int numberOfIterations, bool CLEANMAXIMA, double TRESHOLD)
{
	vtkCell *	currentCell			= NULL;						// Cell of current point
	vtkIdType	currentCellId		= 0;						// Id of current cell
	double		closestPoint[3]		= {0.0, 0.0, 0.0};			// Used in "EvaluatePosition"
	double		pointDistance		= 0.0;						// Used in "EvaluatePosition"
	double		stepDistance		= 0.0;						// Length of current step
	int			subId				= 0;						// Used in "FindCell"
	double		pCoords[3]			= {0.0, 0.0, 0.0};			// Used in "FindCell"
	double		testDot				= 1.0;						// Dot product for current step
	bool		firstStep			= true;						// True during first integration step
	
	// Interpolation weights
	double *	weights = new double[8];

	// Initialize interpolation weights
	for (int i = 0; i < 8; ++i)
	{
		weights[i] = 0.0;
	}

	// Check if there's a point in the point list
	if (!pointList->empty())
	{
		// Get the first point, and clear the list
		currentPoint = pointList->front();
		pointList->clear();

		// Find the cell containing the seed point
		currentCellId = this->HARDIimageData->FindCell(currentPoint.X, NULL, 0, this->tolerance, subId, pCoords, weights);
		currentCell = this->HARDIimageData->GetCell(currentCellId);

		// Set the actual step size, depending on the voxel size
		this->step = direction * this->stepSize * sqrt((double) currentCell->GetLength2());

		// Load the HARDI cell info and AI values of the cell into the "cellHARDIData" and
		// "cellAIScalars" arrays, respectively
		this->HARDIArray->GetTuples(currentCell->PointIds, this->cellHARDIData);
		this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );

		//create a maximumfinder
		MaximumFinder DoIt = MaximumFinder(trianglesArray);
		
		//vector to store the Id's if the found maxima on the ODF
		std::vector<int> maxima;
		//vector to store the unit vectors of the found maxima
		std::vector<double *> outputlistwithunitvectors;
		//neede for search space reduction
		bool searchRegion;
		std::vector<int> regionList;
		//list with ODF values
		std::vector<double> ODFlist;

		//get number of SH components
		int numberSHcomponents = HARDIArray->GetNumberOfComponents();

		// Interpolate the SH at the seed point position
		double * SHAux = new double[numberSHcomponents];
		this->interpolateSH(SHAux, weights, numberSHcomponents);
		
		//get the ODF
		DoIt.getOutput(SHAux, this->parentFilter->shOrder, anglesArray);

		//deallocate memory
		delete [] SHAux;

		// Get the AI scalar at the seed point position
		DoIt.getGFA(&(currentPoint.AI));

		// Set the total distance to zero
		currentPoint.D = 0.0;

		// Re-add the seed point (which now contains eigenvectors and AI)
		pointList->push_back(currentPoint);

		// Set the previous point equal to the current point
		prevPoint = currentPoint;

		// Initialize the previous segment to zero
		this->prevSegment[0] = 0.0;
		this->prevSegment[1] = 0.0;
		this->prevSegment[2] = 0.0;

		// Loop until a stopping condition is met
		while (1) 
		{
			// Compute the next point of the fiber using a Euler step.
			if (!this->solveIntegrationStep(currentCell, currentCellId, weights))
				break;

			// Check if we've moved to a new cell
			vtkIdType newCellId = this->HARDIimageData->FindCell(nextPoint.X, currentCell, currentCellId, 
															this->tolerance, subId, pCoords, weights);

			// If we're in a new cell, and we're still inside the volume...
			if (newCellId >= 0 && newCellId != currentCellId)
			{
				// ...store the ID of the new cell...
				currentCellId = newCellId;

				// ...set the new cell pointer...
				currentCell = this->HARDIimageData->GetCell(currentCellId);

				// ...and fill the cell arrays with the data of the new cell
				this->HARDIArray->GetTuples(currentCell->PointIds, this->cellHARDIData);
				this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );
			}
			// If we've left the volume, break here
			else if (newCellId == -1)
			{
				break;
			}

			// Compute interpolated SH at new position
			double * SHAux = new double[numberSHcomponents];
			this->interpolateSH(SHAux, weights, numberSHcomponents);

			//create a maximum finder
			MaximumFinder DoIt = MaximumFinder(trianglesArray);

			//clear search region list
			regionList.clear();
			double tempVector[3];
			
			//for all directions
			for (unsigned int i = 0; i < anglesArray.size(); ++i)
			{
				searchRegion = false;
				//if its not the first step
				if (!firstStep)
				{
					//get the direction
					tempVector[0] = this->unitVectors[i][0];
					tempVector[1] = this->unitVectors[i][1];
					tempVector[2] = this->unitVectors[i][2];
					//calculate dot product
					testDot = vtkMath::Dot(this->prevSegment, tempVector);
					searchRegion = this->parentFilter->continueTrackingTESTDOT(testDot);
				}
				else
				{
					//if its the first step, search all directions
					searchRegion = true;
					regionList.push_back(i);
				}

				if (searchRegion)
				{
					//add search directions to list
					regionList.push_back(i);
				}
			}	
			
			//get local maxima
			DoIt.getOutput(SHAux, this->parentFilter->shOrder,TRESHOLD, anglesArray,  maxima, regionList);
			
			//if no maxima are found
			if (!(maxima.size() > 0))	
			{
				//break here
				break;
			}

			//clear vector
			outputlistwithunitvectors.clear();
			ODFlist.clear();

			//if the maxima should be cleaned (double and triple maxima) -> get from UI
			if (CLEANMAXIMA)
			{
				//clean maxima
				DoIt.cleanOutput(maxima, outputlistwithunitvectors,SHAux, ODFlist, this->unitVectors, anglesArray);
			}
			else
			{
				//for every found maximum
				for (unsigned int i = 0; i < maxima.size(); ++i)
				{
					//add the unit vector
					double * tempout = new double[3];
					tempout[0] = (this->unitVectors[maxima[i]])[0];
					tempout[1] = (this->unitVectors[maxima[i]])[1];
					tempout[2] = (this->unitVectors[maxima[i]])[2];
					outputlistwithunitvectors.push_back(tempout);
					//add the ODF value
					ODFlist.push_back(DoIt.radii_norm[(maxima[i])]);
				}
			}

			//deallocate memory
			delete [] SHAux;

			//define current maximum at zero (used to determine if a point is a maximum)
			float currentMax = 0.0;
			double tempDirection[3];
			testDot = 0.0;
			//value to compare local maxima (either random value or dot product)
			double value;

			//for all local maxima
			for (unsigned int i = 0; i < outputlistwithunitvectors.size(); ++i)
			{
				//set current direction
				tempDirection[0] = outputlistwithunitvectors[i][0];
				tempDirection[1] = outputlistwithunitvectors[i][1];
				tempDirection[2] = outputlistwithunitvectors[i][2];

				//in case of the first step of a fiber, the dot product cannot be calculated -> set to 1.0
				if (firstStep)
				{
					//set the highest ODF value as condition
					value = ODFlist[i];	
					//get the same directions (prevent missing/double fibers)
					if (tempDirection[0] < 0)
					{
						value = 0.0;
					}
				}
				else
				{
					//calculate the dot product
					value = vtkMath::Dot(this->prevSegment, tempDirection);
				}

				//in case of semi-probabilistic tracking
				if (numberOfIterations > 1)
				{
					//select direction based on a random number
					value = ((double)rand()/(double)RAND_MAX);
				}
				
				//get the "best" value within the range of the user-selected angle
				if (value > currentMax)
				{
					currentMax = value;
					this->newSegment[0] = tempDirection[0];
					this->newSegment[1] = tempDirection[1];
					this->newSegment[2] = tempDirection[2];
				}
			}

			testDot = vtkMath::Dot(this->prevSegment, this->newSegment);
			
			if (firstStep)
			{
				//set the testDot to 1.0 for continueTracking function
				testDot = 1.0;
				
			}
			
			// Interpolate the AI value at the current position
			if (currentCellId >= 0)
			{
				DoIt.getGFA(&(nextPoint.AI));
				//this->interpolateScalar(&(nextPoint.AI), weights);
			}

			// Update the total fiber length
			stepDistance = sqrt((double) vtkMath::Distance2BetweenPoints(currentPoint.X, nextPoint.X));
			this->nextPoint.D = this->currentPoint.D + stepDistance;

			// Call "continueTracking" function of parent filter to determine if
			// one of the stopping criteria has been met.
			 if (!(this->parentFilter->continueTracking(&(this->nextPoint), testDot, currentCellId)))
			{
				// If so, stop tracking.
				break;
			}
			
			// Add the new point to the point list
			pointList->push_back(this->nextPoint);

			// If necessary, increase size of the point list
			if (pointList->size() == pointList->capacity())
			{
				pointList->reserve(pointList->size() + 1000);
			}

			// Update the current and previous points
			this->prevPoint = this->currentPoint;
			this->currentPoint = this->nextPoint;

			// Update the previous line segment
			this->prevSegment[0] = this->newSegment[0];
			this->prevSegment[1] = this->newSegment[1];
			this->prevSegment[2] = this->newSegment[2];

			// This is no longer the first step
			firstStep = false;
		}
	}

	delete [] weights;
}


//-------------------------[ solveIntegrationStep ]------------------------\\


bool HARDIdeterministicTracker::solveIntegrationStep(vtkCell * currentCell, vtkIdType currentCellId, double * weights)
{
		
	// Compute the new segment of the fiber (Euler)
	this->newSegment[0] = this->newSegment[0] * (this->step);
	this->newSegment[1] = this->newSegment[1] * (this->step);
	this->newSegment[2] = this->newSegment[2] * (this->step);

	// Compute the next point
	this->nextPoint.X[0] = this->currentPoint.X[0] + this->newSegment[0];
	this->nextPoint.X[1] = this->currentPoint.X[1] + this->newSegment[1];
	this->nextPoint.X[2] = this->currentPoint.X[2] + this->newSegment[2];

	// Normalize the new line segment
	vtkMath::Normalize(this->newSegment);

	return true;
}


//--------------------------[ interpolateSH ]--------------------------\\

void HARDIdeterministicTracker::interpolateSH(double * interpolatedSH, double * weights, int numberSHcomponents)
{
	//set the output to zero
	for (int i = 0; i < numberSHcomponents; ++i)
	{
		interpolatedSH[i] = 0.0;
	}

	//for all 8 surrounding voxels
	for (int j = 0; j < 8; ++j)
	{
		//get the SH
		double * tempSH = new double[numberSHcomponents];
		this->cellHARDIData->GetTuple(j, tempSH);
		//add the weighted SH to the output array
		for (int i = 0; i < numberSHcomponents; ++i)
		{
			interpolatedSH[i] += weights[j] * tempSH[i];
		}
		delete [] tempSH;
	}
}

//--------------------------[ interpolateScalar ]--------------------------\\

void HARDIdeterministicTracker::interpolateScalar(double * interpolatedScalar, double * weights)
{
	// Set the output to zero
	(*interpolatedScalar) = 0.0;

	// For all eight surrounding voxels...
	for (int i = 0; i < 8; ++i)
	{
		// ...get the corresponding scalar...
		double tempScalar = this->cellAIScalars->GetTuple1(i);;

		// ...and add it to the interpolated scalar
		(*interpolatedScalar) += weights[i] * tempScalar;
	}
}

//--------------------------[ Find maxima ]--------------------------\\

void MaximumFinder::getOutput(double* pDarraySH, int shOrder,double treshold, std::vector<double*> anglesArray,  std::vector<int>& output, std::vector<int> &input)
{
	//clear the output
	output.clear();
	//get radii
	this->radii = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &anglesArray, shOrder);

	//list with neighborhood
	std::vector<int> neighborslist;
	
	//find maximum and minimum radii
	double min = *min_element((this->radii).begin(), (this->radii).end());
	double max = *max_element((this->radii).begin(), (this->radii).end());
	
	//min-max normalization
	//for all points on the sphere
	(this->radii_norm).clear();
	for (unsigned int i = 0; i < (this->radii).size(); ++i)
	{
		//dont divide by 0
		if (min != max)
		{
			//add normalized radius
			this->radii_norm.push_back(((this->radii[i])-min)/(max-min));
		}
		//in case of sphere (SH-order = 0)
		else
		{	//add 1.0 (sphere)    
			this->radii_norm.push_back(1.0);
		}
	}

	//for all points on the sphere
	for (unsigned int i = 0; i < (input.size()); ++i)
	{
		//get current radius
		double currentPointValue = this->radii_norm[input[i]];

		//if the value is high enough
		if (currentPointValue > (treshold))
		{
			//get the neighbors 
			getNeighbors(input[i], 1, neighborslist);
			double currentMax = 0.0;

			//find the highest valued neighbor
			for (unsigned int j = 0; j < neighborslist.size(); ++j)
			{
				if ((this->radii_norm)[(neighborslist[j])]>currentMax)
				{
					currentMax = this->radii_norm[(neighborslist[j])];
				}	
			}
			//if the current point value is higher than the highest valued neighbor
			if (currentMax <= currentPointValue)
			{
				//add point to the output
				output.push_back(input[i]);
			}	
		}
	}	
}

//--------------------------[ Find maxima ]--------------------------\\

void MaximumFinder::getOutput(double* pDarraySH, int shOrder, std::vector<double*> anglesArray)
{
	
	//get radii
	this->radii = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &anglesArray, shOrder);

	
	//list with neighborhood
	std::vector<int> neighborslist;
	
	//find maximum and minimum radii
	double min = *min_element((this->radii).begin(), (this->radii).end());
	double max = *max_element((this->radii).begin(), (this->radii).end());
	
	//min-max normalization
	//for all points on the sphere
	(this->radii_norm).clear();
	for (unsigned int i = 0; i < (this->radii).size(); ++i)
	{
		//dont divide by 0
		if (min != max)
		{
			//add normalized radius
			this->radii_norm.push_back(((this->radii[i])-min)/(max-min));
		}
		//in case of sphere (SH-order = 0)
		else
		{	//add 1.0 (sphere)    
			this->radii_norm.push_back(1.0);
		}
	}
}

//-----------------------------[ Get neighbors ]------------------------------\\

void MaximumFinder::getNeighbors(int i, int depth, std::vector<int> &neighborlist_final)
{
	//clear output vector
	neighborlist_final.clear();
	//set point i as input and find all direct neighbors of point i
	std::vector<int> input(1, i);
	this->getDirectNeighbors(input, neighborlist_final);
	
	if (depth > 1) //depth=2 or depth=3
	{
		std::vector<int> neighborslist2;
		//find direct neighbors
		this->getDirectNeighbors(neighborlist_final,  neighborslist2);
		for (unsigned int index21 = 0; index21<neighborslist2.size(); index21++)
		{
			//add them to the output list
			neighborlist_final.push_back(neighborslist2[index21]);
		}
	}

	//remove duplicates from the list
	std::sort(neighborlist_final.begin(), neighborlist_final.end());
	neighborlist_final.erase(std::unique(neighborlist_final.begin(), neighborlist_final.end()), neighborlist_final.end());
	
	//remove seedpoint i from the list
    neighborlist_final.erase(std::remove(neighborlist_final.begin(), neighborlist_final.end(), i), neighborlist_final.end());
}

//-----------------------------[ Get direct neighbors ]------------------------------\\

void MaximumFinder::getDirectNeighbors(std::vector<int> seedpoints, std::vector<int> &temp_neighbors)
{
	//clear output vector
	temp_neighbors.clear();
	//for every seed point
	for (unsigned int k =0; k<seedpoints.size(); k++)
	{
		//for every triangle
		for (int j=0;j<(this->trianglesArray->GetNumberOfTuples());j++)
		{
			//set index
			vtkIdType index = j;
			//get triangle
			double * triangle=this->trianglesArray->GetTuple(index);
			//if it contains the seedpoint index
			if ((triangle[0]==seedpoints[k]) || (triangle[1]==seedpoints[k]) || (triangle[2]==seedpoints[k]))
			{
				//add all triangle members to the list
				temp_neighbors.push_back(triangle[0]);
				temp_neighbors.push_back(triangle[1]);
				temp_neighbors.push_back(triangle[2]);
			}
		}
	}

	//remove duplicates from the list
	std::sort(temp_neighbors.begin(), temp_neighbors.end());
	temp_neighbors.erase(std::unique(temp_neighbors.begin(), temp_neighbors.end()), temp_neighbors.end());
}

//-----------------------------[ get GFA ]------------------------------\\

void MaximumFinder::getGFA(double * GFA)
{
	
	
	//get number of tesselation points
	int n = (this->radii).size();
	//set output to zero
	(*GFA) = 0.0;

	//calculate average radius
	double average = 0.0;
	for (int i = 0; i < n; ++i)
	{
		average += (this->radii)[i];
	}
	average = average/n;

	//calculate the numerator
	double numerator = 0.0;
	for (int i = 0; i < n; ++i)
	{
		numerator += pow(((this->radii)[i]-average),2);
	}
	numerator = numerator*n;

	//calculate the denominator
	double denominator = 0.0;
	for (int i = 0; i < n; ++i)
	{
		denominator += pow((this->radii)[i],2);
	}
	denominator = denominator*(n-1);

	//calculate the GFA
	(*GFA) = sqrt(numerator/denominator);
	
}

//-----------------------------[ Clean output ]------------------------------\\

void MaximumFinder::cleanOutput(std::vector<int> output, std::vector<double *>& outputlistwithunitvectors, double* pDarraySH, std::vector<double> &ODFlist, double** unitVectors, std::vector<double*> &anglesArray)
{
	//list with single maxima
	std::vector<int> goodlist;

	//list with double maxima (2 neighboring points)
	std::vector<int> doubtlist;
	
	//list with registered maxima
	std::vector<int> donelist;

	//temporary output vector
	std::vector<double *> outputlistwithunitvectors1;
	
	//list with radii of calculated points
	std::vector<double> radius;
	//temporary angle list
	std::vector<double*> tempAngleArray;
	//temporary ODF list
	std::vector<double> tempODFlist;

	//lists with neighbors
	std::vector<int> neighborhood1;
	std::vector<int> neighborhood2;
	
	//for every maximum, count the number of neighboring maxima with depth 1
	//and add maximum to correct list
	for (unsigned int i = 0; i < output.size(); ++i)
	{	
		//get neighbors
		this->getNeighbors(output[i],2,neighborhood1);
		int neighborcounter = 0;
		for (unsigned int j = 0; j < neighborhood1.size(); ++j)
		{
			//count neighbors
			if (std::find(output.begin(), output.end(), neighborhood1[j]) != output.end())
			{
				neighborcounter += 1;	
			}
		}
		//single- and multi-point maxima
		if ((neighborcounter == 0) || (neighborcounter > 2))
		{
			goodlist.push_back(output[i]);
		}
		//double and tiple maxima
		if ((neighborcounter == 1) || (neighborcounter == 2))
		{
			doubtlist.push_back(output[i]);
		}
	}
		
	//for all double and tirple maxima
	for (unsigned int i = 0; i < doubtlist.size(); ++i)
	{
		//check for triangle
		for (int j=0;j<(this->trianglesArray->GetNumberOfTuples());j++)
		{
			vtkIdType index = j;
			double * henk=this->trianglesArray->GetTuple(index);
			//typecasting
			int index0 = henk[0];
			int index1 = henk[1];
			int index2 = henk[2];
			
			//check for triple maximum
			if ((index0==doubtlist[i]) || (index1==doubtlist[i]) || (index2==doubtlist[i]))
			{
				if (
					(std::find(doubtlist.begin(), doubtlist.end(), index0) != doubtlist.end()) &&
					(std::find(doubtlist.begin(), doubtlist.end(), index1) != doubtlist.end())	&&
					(std::find(doubtlist.begin(), doubtlist.end(), index2) != doubtlist.end())
					)
				{
					//get angles of original and calculated average directions
					double * angles = new double[2];
					angles[0] = (1.0/3.0)*(anglesArray[index0][0])+(1.0/3.0)*(anglesArray[index1][0])+(1.0/3.0)*(anglesArray[index2][0]);
					angles[1] = (1.0/3.0)*(anglesArray[index0][1])+(1.0/3.0)*(anglesArray[index1][2])+(1.0/3.0)*(anglesArray[index2][3]);

					double * originals1 = new double[2];
					originals1[0] = (anglesArray[index0][0]);
					originals1[1] = (anglesArray[index0][1]);

					double * originals2 = new double[2];
					originals2[0] = (anglesArray[index1][0]);
					originals2[1] = (anglesArray[index1][1]);

					double * originals3 = new double[2];
					originals2[0] = (anglesArray[index2][0]);
					originals2[1] = (anglesArray[index2][1]);

					tempAngleArray.clear();
					tempAngleArray.push_back(angles);
					tempAngleArray.push_back(originals1);
					tempAngleArray.push_back(originals2);
					tempAngleArray.push_back(originals3);

					//get the corresponding radii
					radius.clear();
					radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);
					
					//delete pointers
					delete angles;
					delete originals1;
					delete originals2;
					delete originals3;

					//if the averagedirection has a higher ODF value than the original directions
					if ((fabs(radius[1]) < fabs(radius[0])) && (fabs(radius[2]) < fabs(radius[0])) && (fabs(radius[3]) < fabs(radius[0])))
					{
						double * tempout = new double[3];
						//calculate the average direction
						tempout[0] = (1.0/3.0)*(unitVectors[(index0)][0])+(1.0/3.0)*(unitVectors[(index1)][0])+(1.0/3.0)*(unitVectors[(index2)][0]);
						tempout[1] = (1.0/3.0)*(unitVectors[(index0)][1])+(1.0/3.0)*(unitVectors[(index1)][1])+(1.0/3.0)*(unitVectors[(index2)][0]);
						tempout[2] = (1.0/3.0)*(unitVectors[(index0)][2])+(1.0/3.0)*(unitVectors[(index1)][2])+(1.0/3.0)*(unitVectors[(index2)][0]);
						//add the calculated direction to the output
						outputlistwithunitvectors1.push_back(tempout);
						//add the ODF value to the output
						tempODFlist.push_back(fabs(radius[0]));
					}
					else
					{
						//add the original direction and ODF value to the output
						outputlistwithunitvectors1.push_back(unitVectors[doubtlist[i]]);
						if (doubtlist[i] == index0)
						{
							tempODFlist.push_back(fabs(radius[1]));
						}
						if (doubtlist[i] == index1)
						{
							tempODFlist.push_back(fabs(radius[2]));
						}
						if (doubtlist[i] == index2)
						{
							tempODFlist.push_back(fabs(radius[3]));
						}
					}
					donelist.push_back(i);
				}
			}
		}
		//for all points that are not a triple maximum
		if (!(std::find(donelist.begin(), donelist.end(), i) != donelist.end()))
		{
			//check for double point
			this->getNeighbors(doubtlist[i],1,neighborhood1);
			for (unsigned int j = 0; j < neighborhood1.size(); ++j)
			{
				if (std::find(doubtlist.begin(), doubtlist.end(), neighborhood1[j]) != doubtlist.end())
				{
					//get angles of original and calculated average directions
					double * angles = new double[2];
					angles[0] = 0.5*(anglesArray[(neighborhood1[j])][0])+0.5*(anglesArray[(doubtlist[i])][0]);
					angles[1] = 0.5*(anglesArray[(neighborhood1[j])][1])+0.5*(anglesArray[(doubtlist[i])][1]);

					double * originals1 = new double[2];
					originals1[0] = (anglesArray[(doubtlist[i])][0]);
					originals1[1] = (anglesArray[(doubtlist[i])][1]);

					double * originals2 = new double[2];
					originals2[0] = (anglesArray[(neighborhood1[j])][0]);
					originals2[1] = (anglesArray[(neighborhood1[j])][1]);

					tempAngleArray.clear();
					tempAngleArray.push_back(angles);
					tempAngleArray.push_back(originals1);
					tempAngleArray.push_back(originals2);

					//get the corresponding radii
					radius.clear();
					radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);
					
					//delete pointers
					delete angles;
					delete originals1;
					delete originals2;

					//if the averagedirection has a higher ODF value than the original directions
					if ((fabs(radius[0]) > fabs(radius[1])) && (fabs(radius[0]) > fabs(radius[2])))
					{
						double * tempout = new double[3];
						//calculate the average direction
						tempout[0] = 0.5*(unitVectors[(neighborhood1[j])][0])+0.5*(unitVectors[(doubtlist[i])][0]);
						tempout[1] = 0.5*(unitVectors[(neighborhood1[j])][1])+0.5*(unitVectors[(doubtlist[i])][1]);
						tempout[2] = 0.5*(unitVectors[(neighborhood1[j])][2])+0.5*(unitVectors[(doubtlist[i])][2]);

						//add the calculated direction to the output
						outputlistwithunitvectors1.push_back(tempout);
						//add the ODF value to the output
						tempODFlist.push_back(fabs(radius[0]));
					}
					else
					{
						//add the original direction and ODF value to the output
						outputlistwithunitvectors1.push_back(unitVectors[doubtlist[i]]);
						tempODFlist.push_back(fabs(radius[1]));
					}
					radius.clear();
				}
				else
				{
					//for all remaining points
					this->getNeighbors(doubtlist[i],1,neighborhood2);
					int neighborcounter = 0;
					//count the neighbors
					for (unsigned int j = 0; j < neighborhood2.size(); ++j)
					{
						if (std::find(output.begin(), output.end(), neighborhood2[j]) != output.end())
						{
							neighborcounter += 1;	
						}
					}
					//if there are more than 1 neighbor 
					if (neighborcounter != 1)
					{
						//add unit vector to the output list
						outputlistwithunitvectors1.push_back(unitVectors[doubtlist[i]]);
						//get angles
						double * originals1 = new double[2];
						originals1[0] = (anglesArray[(doubtlist[i])][0]);
						originals1[1] = (anglesArray[(doubtlist[i])][1]);
						tempAngleArray.clear();
						tempAngleArray.push_back(originals1);
						//get radius
						radius.clear();
						radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);
						delete originals1;
						//add radius to output list
						tempODFlist.push_back(fabs(radius[0]));
					}
				}
			}
		}
	}

	//add the single point maxima
	for (unsigned int i = 0; i < goodlist.size(); ++i)
	{
		//add unit vector to the output list
		outputlistwithunitvectors1.push_back(unitVectors[goodlist[i]]);

		//get angles
		double * originals1 = new double[2];
		originals1[0] = (anglesArray[(goodlist[i])][0]);
		originals1[1] = (anglesArray[(goodlist[i])][1]);
		tempAngleArray.clear();
		tempAngleArray.push_back(originals1);
		//get radius
		radius.clear();
		radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);
		delete originals1;
		//add radius to output list
		tempODFlist.push_back(radius[0]);
	}

	//delete any duplicates
	outputlistwithunitvectors.push_back(outputlistwithunitvectors1[0]);
	ODFlist.push_back(tempODFlist[0]);
	bool duplicate;
	//for every unit vector, ceck whether it is already in the final output list
	for (unsigned int i = 1; i < (outputlistwithunitvectors1).size(); ++i)
	{
		duplicate = false;
		for (unsigned int j = 0; j < outputlistwithunitvectors.size(); ++j)
		{
			if ((outputlistwithunitvectors1[i][0] == outputlistwithunitvectors[j][0]) && (outputlistwithunitvectors1[i][1] == outputlistwithunitvectors[j][1]) && (outputlistwithunitvectors1[i][2] == outputlistwithunitvectors[j][2]))
			{
				duplicate = true;
			}
		}
		//if it is not yet it the list
		if (!duplicate)
		{
			//add unit vector and ODF value to the list
			outputlistwithunitvectors.push_back(outputlistwithunitvectors1[i]);
			ODFlist.push_back(tempODFlist[i]);
		}
	}
}

//-----------------------------[ Set unit vectors ]------------------------------\\

void HARDIdeterministicTracker::setUnitVectors(double ** unitVectors)
{
	//set unit vectors
	this-> unitVectors = unitVectors;
}
} // namespace bmia