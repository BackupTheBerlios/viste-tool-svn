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
*  2013-03-15 Mehmet Yusufoglu, Bart Knippenberg
* -Can process a discrete sphere data which already have Spherical Directions and Triangles arrays. 
*  HARDIdeterministicTracker::CalculateFiberDS and MaximumFinderGetOutputDS functions were added.
*/



/** Includes */

#include "HARDIdeterministicTracker.h"
#include "Maximumfinder.h"
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
		//debug options
		printStepInfo =0;
		breakLoop=0;


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
			MaximumFinder MaxFinder = MaximumFinder(trianglesArray);

			//vector to store the Id's if the found maxima on the ODF
			std::vector<int> maxima;
			//vector to store the unit vectors of the found maxima
			std::vector<double *> outputlistwithunitvectors;
			//neede for search space reduction
			bool searchRegion;
			std::vector<int> meshPtIndexList;
			//list with ODF values
			std::vector<double> ODFlist;

			//get number of SH components
			int numberSHcomponents = HARDIArray->GetNumberOfComponents();

			// Interpolate the SH at the seed point position
			double * SHAux = new double[numberSHcomponents];
			this->interpolateSH(SHAux, weights, numberSHcomponents);

			//get the ODF
			MaxFinder.getOutput(SHAux, this->parentFilter->shOrder, anglesArray);

			//deallocate memory
			delete [] SHAux;

			// Get the AI scalar at the seed point position
			MaxFinder.getGFA(&(currentPoint.AI));

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
				MaximumFinder MaxFinder = MaximumFinder(trianglesArray);

				//clear search region list
				meshPtIndexList.clear();
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
						meshPtIndexList.push_back(i);
					}

					if (searchRegion)
					{
						//add search directions to list
						meshPtIndexList.push_back(i);
					}
				}	

				//get local maxima
				MaxFinder.getOutput(SHAux, this->parentFilter->shOrder,TRESHOLD, anglesArray,  maxima, meshPtIndexList);

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
					MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,SHAux, ODFlist, this->unitVectors, anglesArray);
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
						ODFlist.push_back(MaxFinder.radii_norm[(maxima[i])]);
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
					MaxFinder.getGFA(&(nextPoint.AI));
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





	void HARDIdeterministicTracker::findMax2( std::vector<double> &array, std::vector<double> &maxima, std::vector<double*> &maximaunitvectors, std::vector<double *> &anglesReturn)
	{
		// find indexes of maximum and the secondary max
		double max =0;
		int max_index1=0;
		int max_index2=0;
		for(int i=0; i< array.size(); i++)
			if(array.at(i) > max) { max = array.at(i); max_index1=i;  }
			max=0;
			for(int i=0; i< array.size(); i++)
				if(array.at(i) > max) { max = array.at(i); if(max_index1!=i) max_index2=i;  }
				maxima.push_back(max_index1);
				maxima.push_back(max_index2);

				double * sc1 = new double[2];
				sc1[0] = acos( maximaunitvectors[max_index1][2]);
				sc1[1] = atan2( maximaunitvectors[max_index1][1],  maximaunitvectors[max_index1][0]);
				anglesReturn.push_back(sc1);
				double * sc2 = new double[2];
				sc2[0] = acos( maximaunitvectors[max_index2][2]);
				sc2[1] = atan2( maximaunitvectors[max_index2][1],  maximaunitvectors[max_index2][0]);
				anglesReturn.push_back(sc2);
	}



	//----------------------------[ calculateFiber using Spherical Harmonics Directions Interpolation for ONLY one seed point]---------------------------\\

	void HARDIdeterministicTracker::calculateFiberSHDI(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,int numberOfIterations, bool CLEANMAXIMA, double TRESHOLD)
	{

		cout << "----------------  New Seed for a New Fiber - calculateFiberSHDI ---------------("<< direction <<")"<<  endl;
		vtkCell *	currentCell			= NULL;						// Cell of current point
		vtkIdType	currentCellId		= 0;						// Id of current cell
		double		closestPoint[3]		= {0.0, 0.0, 0.0};			// Used in "EvaluatePosition"
		double		pointDistance		= 0.0;						// Used in "EvaluatePosition"
		double		incrementalDistance		= 0.0;						// Length of current step
		int			subId				= 0;						// Used in "FindCell"
		double		pCoords[3]			= {0.0, 0.0, 0.0};			// Used in "FindCell"
		double		testDot				= 1.0;						// Dot product for current step
		bool		firstStep			= true;						// True during first integration step
		double  threshold = 0;
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
			currentPoint = pointList->front(); // first point
			pointList->clear();
			this->nextPoint.D = 0.0;
			// Find the cell containing the seed point
			currentCellId = this->HARDIimageData->FindCell(currentPoint.X, NULL, 0, this->tolerance, subId, pCoords, weights); // fills the weights
			currentCell = this->HARDIimageData->GetCell(currentCellId);

			// Set the actual step size, depending on the voxel size
			this->step = direction * this->stepSize * sqrt((double) currentCell->GetLength2());

			// Load the HARDI cell info and AI values of the cell into the "cellHARDIData" and
			// "cellAIScalars" arrays, respectively
			this->HARDIArray->GetTuples(currentCell->PointIds, this->cellHARDIData);
			this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );



			// IF From file 
			/*	
			this->maximaArrayFromFile->GetTuples(currentCell->PointIds, this->maximasCellFromFile);

			for(unsigned int nr = 0; nr <this->outUnitVectorListFromFile.size()  ; nr++)
			{		 
			this->outUnitVectorListFromFile.at(nr)->GetTuples(currentCell->PointIds, this->unitVectorCellListFromFile.at(nr));
			}
			*/
			//this->outUnitVectorListFromFile.at(0)->GetTuples(currentCell->PointIds, unitVectors1FromFile);


			//create a maximumfinder
			MaximumFinder MaxFinder = MaximumFinder(trianglesArray); // what does this arr do

			//vector to store the Id's if the found maxima on the ODF
			std::vector<int> maxima;
			//vector to store the unit vectors of the found maxima
			std::vector<double *> outputlistwithunitvectors;
			//neede for search space reduction
			
			std::vector<int> meshPtIndexList;
			//list with ODF values
			std::vector<double> ODFlist;
			std::vector<double> ODFlistMaxTwo;

			//get number of SH components
			int numberSHcomponents = HARDIArray->GetNumberOfComponents();

			// Interpolate the SH at the seed point position
			//double * SHAux = new double[numberSHcomponents];
			//this->interpolateSH(SHAux, weights, numberSHcomponents); //not interpolate now
			double * tempSH = new double[numberSHcomponents];


			double *avgMaxAng = new double[2];
			std::vector<double *> anglesBeforeInterpolation; 

			//initial regionlist includes all points not some points of the ODF
			if(meshPtIndexList.size()==0)
				for(int i=0;i<anglesArray.size();i++)
					meshPtIndexList.push_back(i);
			for (int j = 0; j < 8; ++j)
			{
				//get the SH

				this->cellHARDIData->GetTuple(j, tempSH);
				//this->cellHARDIData has 8 hardi coeffieint sets
				//get the ODF // get maxes like below 8 times


				MaxFinder.getOutput(tempSH, this->parentFilter->shOrder,TRESHOLD, anglesArray,  maxima, meshPtIndexList);// SHAux is empty now we will give 8 differen , radiusun buyuk oldugu yerdeki angellari dizer donen 
				// maxima has ids use them to get angles
				MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,tempSH, ODFlist, this->unitVectors, anglesArray);
				avgMaxAng[0]=0;
				avgMaxAng[1]=0;
				for(int i=0; i< maxima.size(); i++)
				{
					avgMaxAng[0]+=anglesArray.at(maxima.at(i))[0];   // choose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					avgMaxAng[1]+=anglesArray.at(maxima.at(i))[1];   // ose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					cout << "anglesOfmaxOfCorner"  << anglesArray.at(maxima.at(i))[0] << " " << anglesArray.at(maxima.at(i))[1] << endl;
				}
				avgMaxAng[0]/=maxima.size();
				avgMaxAng[1]/=maxima.size();
				cout << avgMaxAng[0] << " " << avgMaxAng[1] << endl;
				anglesBeforeInterpolation.push_back(avgMaxAng); // if angles are in the range of [-pi,pi] interpolation is ok
				outputlistwithunitvectors.clear();
				// TAKEN BEFORE THE AVERAGING MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,SHAux, ODFlist, this->unitVectors, anglesArray);
				ODFlist.clear();
				maxima.clear();


			}// for cell 8 
			double interpolatedDirection[2];
			this->interpolateAngles(anglesBeforeInterpolation,weights, interpolatedDirection); // this average will be used as initial value. 
			anglesBeforeInterpolation.clear();
			double tempDirection[3];
			tempDirection[0] = sinf(interpolatedDirection[0]) * cosf(interpolatedDirection[1]);
			tempDirection[1] = sinf(interpolatedDirection[0]) * sinf(interpolatedDirection[1]);
			tempDirection[2] = cosf(interpolatedDirection[0]);
			//	tempDirection already normalized!!!
			// use weights as interpolatin of angles...
			// add 
			//deallocate memory


			// Get the AI scalar at the seed point position
			//MaxFinder.getGFA(&(currentPoint.AI));
			this->interpolateScalar(&(currentPoint.AI), weights);
			// Set the total distance to zero
			currentPoint.D = 0.0;

			// Re-add the seed point  
			pointList->push_back(currentPoint);

			// Set the previous point equal to the current point
			prevPoint = currentPoint;

			// Initialize the previous segment to zero
			this->newSegment[0] = tempDirection[0]; // 0.0;
			this->newSegment[1] = tempDirection[1];// 0.0;
			this->newSegment[2] = tempDirection[2];//0.0;
			
			this->prevSegment[0]=this->prevSegment[1]=this->prevSegment[2]= 0.0; // CHECK!!!

			// Loop until a stopping condition is met


			cout <<"prev segment before:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
			cout <<"new segment before:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
			cout <<"currentpoint before:" << this->currentPoint.X[0] << " " << this->currentPoint.X[1]  << " "<< this->currentPoint.X[2]  << endl;

			// Compute the next point (nextPoint) of the fiber using a Euler step.
			if (!this->solveIntegrationStepSHDI(currentCell, currentCellId, weights)) //Add NEwSegment to Current Point to Determine NEXT Point!!!
			{ cout << "Problem at the first integratioon step"<< endl; return; } 
			cout <<"nextpoimt after:" << this->nextPoint.X[0] << " " << this->nextPoint.X[1]  << " "<< this->nextPoint.X[2]  << endl;

			// Update the current and previous points
			this->prevPoint = this->currentPoint;
			this->currentPoint = this->nextPoint;

			// Update the previous line segment
			this->prevSegment[0] = this->newSegment[0];
			this->prevSegment[1] = this->newSegment[1];
			this->prevSegment[2] = this->newSegment[2];

			//create a maximum finder
			//	M
			//initial regionlist includes all points not some points
			if(meshPtIndexList.size()==0)
				for(int i=0;i<anglesArray.size();i++)
					meshPtIndexList.push_back(i);
			// Check AI values of initial step, otherwise we cannot check the dot product etc
			while (1) 
			{

				cout << endl << "===== while ==================== " << direction << " ==" << endl;

				// Check if we've moved to a new cell. NEXT POINT is USE DTO FIND CURRENT CELL!!
				vtkIdType newCellId = this->HARDIimageData->FindCell(currentPoint.X, currentCell, currentCellId,this->tolerance, subId, pCoords, weights);
				if(this->breakLoop) break;
				//if(this->printStepInfo)
				//{
				//	cout << "newCellId"<< newCellId <<   endl;
				//	for (unsigned int i = 0; i <8; ++i)// angles array is constant for all voxels
				//	{
				//		cout <<  "weight[" << i << "]:" << weights[i] << endl;
				//	}

				//}
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
					cout << "NOT  A CELL; BREAK " << endl; 
					break;
				}


				double *interpolatedVector;
				interpolatedVector = findFunctionValue(TRESHOLD, anglesArray, weights,  trianglesArray, meshPtIndexList, maxima);

				testDot = 0.0;
				//value to compare local maxima (either random value or dot product)
				//double value;

				this->newSegment[0] = interpolatedVector[0]; // we will haveone unitvector !!! interpolation of angels will 
				this->newSegment[1] = interpolatedVector[1]; // produce an angle and we will calculate tempDirection!!!!
				this->newSegment[2] = interpolatedVector[2];

				if(this->breakLoop) break;
				if(this->printStepInfo)
				{
					//	cout <<"prev segment1:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
					cout <<"new segment1:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
					cout <<"currentpoint before:" << this->currentPoint.X[0] << " " << this->currentPoint.X[1]  << " "<< this->currentPoint.X[2]  << endl;				
					cout <<"this->step:" << this->step << endl;
					// Compute the next point (nextPoint) of the fiber using a Euler step.
				}

				if (!this->solveIntegrationStepSHDI(currentCell, currentCellId, weights)) //Add NEwSegment to Current Point to Determine NEXT Point!!!
					break;	

				if(this->printStepInfo)
				{
					cout <<"nextpoimt after:" << this->nextPoint.X[0] << " " << this->nextPoint.X[1]  << " "<< this->nextPoint.X[2]  << endl;
					cout <<"prev segment1.1:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
					cout <<"new segment1.1:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
				}	
				// Update the total fiber length
				incrementalDistance = sqrt((double) vtkMath::Distance2BetweenPoints(currentPoint.X, nextPoint.X)); // next point nerede dolar ??

				if(this->printStepInfo)
				{
					cout << "current point: "<< currentPoint.X[0] << " " << currentPoint.X[1] << " " << currentPoint.X[2] << " " << endl;
					cout << "next point: "<< nextPoint.X[0] << " " << nextPoint.X[1] << " " << nextPoint.X[2] << " " << endl;
					//cout << "incremental Distance" << incrementalDistance << endl;
				}
				this->nextPoint.D = this->currentPoint.D + incrementalDistance;
				// Interpolate the AI value at the current position
				if (currentCellId >= 0)
				{
					//MaxFinder.getGFA(&(currentPoint.AI));
					this->interpolateScalar(&(currentPoint.AI), weights); // WEIGHTS are OLD for next point?				 
				}

				testDot = vtkMath::Dot(this->prevSegment, this->newSegment); // stop condition new segment is normalized after the increment for dotproduct

				if(this->printStepInfo)
					cout << "testDot: " << testDot  <<  "current point AI: " << currentPoint.AI << endl;
				// Call "continueTracking" function of parent filter to determine if
				// one of the stopping criteria has been met.
				if (!(this->parentFilter->continueTracking(&(this->currentPoint), testDot, currentCellId)))// Current of NExt Point???
				{
					// If so, stop tracking.
					cout << "STOP TRACKING. testDot: " << testDot  <<   endl;
					break;
				}

				// Add the new point to the point list
				pointList->push_back(this->nextPoint);

				//if(this->printStepInfo)
				//	cout << "pointList.size"<< pointList->size() << endl;
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
				this->prevSegment[2] = this->newSegment[2]; // prevseg becomes automaticly normalized!!!

				if(this->breakLoop) break;
				//	if(this->printStepInfo) {
				//	cout <<"prev segment2:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
				//		cout <<"new segment2:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
				//	}

			} //while 
		}//if

		delete [] weights;
	}

	void HARDIdeterministicTracker::calculateFiberSHDIUseOfflineMaximaDirections(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,int numberOfIterations, bool CLEANMAXIMA, double TRESHOLD, int initCondition)
	{

		cout << "-----  New Seed for a New Fiber - calculateFiberSHDIMaxDirection Offline ------("<< direction <<")"<<  endl;
		vtkCell *	currentCell			= NULL;						// Cell of current point
		vtkIdType	currentCellId		= 0;						// Id of current cell
		double		closestPoint[3]		= {0.0, 0.0, 0.0};			// Used in "EvaluatePosition"
		double		pointDistance		= 0.0;						// Used in "EvaluatePosition"
		double		incrementalDistance		= 0.0;						// Length of current step
		int			subId				= 0;						// Used in "FindCell"
		double		pCoords[3]			= {0.0, 0.0, 0.0};			// Used in "FindCell"
		double		testDot				= 1.0;						// Dot product for current step
		bool		firstStep			= true;						// True during first integration step
		double  threshold = 0;
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
			currentPoint = pointList->front(); // first point
			pointList->clear();
			this->nextPoint.D = 0.0;
			// Find the cell containing the seed point
			currentCellId = this->HARDIimageData->FindCell(currentPoint.X, NULL, 0, this->tolerance, subId, pCoords, weights); // fills the weights
			currentCell = this->HARDIimageData->GetCell(currentCellId);

			// Set the actual step size, depending on the voxel size
			this->step = direction * this->stepSize * sqrt((double) currentCell->GetLength2());

			// Load the HARDI cell info and AI values of the cell into the "cellHARDIData" and
			// "cellAIScalars" arrays, respectively
			this->HARDIArray->GetTuples(currentCell->PointIds, this->cellHARDIData);
			this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );

			//From File /////////////////

			//this->maximaArrayFromFile->GetTuples(currentCell->PointIds, maximasCellFromFile);
			for(unsigned int nr = 0; nr <outUnitVectorListFromFile.size()  ; nr++)
			{
				this->outUnitVectorListFromFile.at(nr)->GetTuples(currentCell->PointIds, unitVectorCellListFromFile.at(nr));
			}
			//create a maximumfinder
			MaximumFinder MaxFinder = MaximumFinder(trianglesArray); // what does this arr do

			//vector to store the Id's if the found maxima on the ODF
			std::vector<int> maxima;
			//vector to store the unit vectors of the found maxima
			std::vector<double *> outputlistwithunitvectors;
			//neede for search space reduction
	
			




			std::vector<int> meshPtIndexList;
			//list with ODF values
			std::vector<double> ODFlist;
			std::vector<double> ODFlistMaxTwo;

			//get number of SH components
			int numberSHcomponents = HARDIArray->GetNumberOfComponents();

			// Interpolate the SH at the seed point position
			//double * SHAux = new double[numberSHcomponents];
			//this->interpolateSH(SHAux, weights, numberSHcomponents); //not interpolate now
			double * tempSH = new double[numberSHcomponents];


			double *avgMaxAng = new double[2];

			//IF FILE
			double *maximaOfAPointFromFile = new double[this->nMaximaForEachPoint];
			double **unitVectorsOfAPointFromFile = new double*[this->nMaximaForEachPoint];
			for (int j = 0; j < this->nMaximaForEachPoint; ++j)
				unitVectorsOfAPointFromFile[j] = new double[3];
			double tempDirection[3];
			////////////////////////////////////////////////////////////////////
			// INITIAL CONDITON PART
			// 
			    
			if(initCondition==0) {
			// Interpolate the SH at the seed point position
			double * SHAux = new double[numberSHcomponents];
			this->interpolateSH(SHAux, weights, numberSHcomponents);// uses this cellHARDIData

			//get the ODF
			//MaxFinder.getOutput(SHAux, this->parentFilter->shOrder, anglesArray); // get output
			MaxFinder.getOutput(SHAux, this->parentFilter->shOrder,TRESHOLD, anglesArray,  maxima, meshPtIndexList);// SHAux is empty now we will give 8 differen , radiusun buyuk oldugu yerdeki angellari dizer donen 
				// maxima has ids use them to get angles
				MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,SHAux, ODFlist, this->unitVectors, anglesArray);
	//		for (int j = 0; j < this->nMaximaForEachPoint; ++j)
		//		outputlistwithunitvectors[j] = this->unitVectors[j];

			// Find maximum of this interpolated values here and use as initial condition
			//deallocate memory
			delete [] SHAux;
			tempDirection[0] =  unitVectors[0][0];
			tempDirection[1] =  unitVectors[0][1];
			tempDirection[2] =  unitVectors[0][2];
			}
			///////////////////////////////////////////////////

			else if (initCondition==1 || initCondition==2 )
			{
			std::vector<double *> anglesBeforeInterpolation; 

			//initial regionlist includes all points not some points of the ODF
			if(meshPtIndexList.size()==0)
				for(int i=0;i<anglesArray.size();i++)
					meshPtIndexList.push_back(i);
			for (int j = 0; j < 8; ++j)// vertices if cell or voxel
			{
				//get the SH

				this->cellHARDIData->GetTuple(j, tempSH);
				//this->cellHARDIData has 8 hardi coeffieint sets
				//get the ODF // get maxes like below 8 times

				//IF FROM FILE
				//this->maximasCellFromFile->GetTuple(j,maximaOfAPointFromFile);
				for (int n = 0; n < this->nMaximaForEachPoint; ++n)
					// unitVectorCellListFromFile s each item is an array of 8 tuples each tuple has a unit vector
                    //  unitVectorCellListFromFile.at(0) has 8 vectors each is the longest maxima vector of the correponding vertex
                    //   unitVectorCellListFromFile.at(1) has 8 vectors each is the secong longest vector of the coorresponding vertex
					this->unitVectorCellListFromFile.at(n)->GetTuple(j,unitVectorsOfAPointFromFile[n] );//   unitVectorsOfAPointFromFile consists maxima vectors for a point

				for (int k = 0; k <this->nMaximaForEachPoint; ++k)
				{
					//maxima.push_back(maximaOfAPointFromFile[k]);
					outputlistwithunitvectors.push_back(unitVectorsOfAPointFromFile[k]);  // not necessary can be changed. maxima vectors for a point
				}

				// Angles from unit vectors is a better idea.
				///MaxFinder.getOutput(tempSH, this->parentFilter->shOrder,TRESHOLD, anglesArray,  maxima, meshPtIndexList);// SHAux is empty now we will give 8 differen , radiusun buyuk oldugu yerdeki angellari dizer donen 
				// maxima has ids use them to get angles
				//MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,tempSH, ODFlist, this->unitVectors, anglesArray);
				avgMaxAng[0]=0;
				avgMaxAng[1]=0;
				for(int i=0; i< this->nMaximaForEachPoint; i++)// START FROM HERE!!! this->n
				{	
					if (initCondition==1)
					if(i%2==0) {
					avgMaxAng[0]+= acos( outputlistwithunitvectors[i][2]);  // choose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					avgMaxAng[1]+= atan2( outputlistwithunitvectors[i][1],  outputlistwithunitvectors[i][0]);  // ose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					}//cout << "anglesOfmaxOfCorner"  << anglesArray.at(maxima.at(i))[0] << " " << anglesArray.at(maxima.at(i))[1] << endl;
				else if (initCondition==2)
					if(i%2==1) {
					avgMaxAng[0]+= acos( outputlistwithunitvectors[i][2]);  // choose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					avgMaxAng[1]+= atan2( outputlistwithunitvectors[i][1],  outputlistwithunitvectors[i][0]);  // ose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					}//co
				
				}
				avgMaxAng[0]=avgMaxAng[0]/(this->nMaximaForEachPoint/2); // 2 CAREFULL
				avgMaxAng[1]=avgMaxAng[1]/(this->nMaximaForEachPoint/2);
				//cout << avgMaxAng[0] << " " << avgMaxAng[1] << endl;
				anglesBeforeInterpolation.push_back(avgMaxAng); // if angles are in the range of [-pi,pi] interpolation is ok
				outputlistwithunitvectors.clear();
				// TAKEN BEFORE THE AVERAGING MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,SHAux, ODFlist, this->unitVectors, anglesArray);
				ODFlist.clear();
				maxima.clear();


			}// for cell 8 
			double interpolatedDirection[2];
			this->interpolateAngles(anglesBeforeInterpolation,weights, interpolatedDirection); // this average will be used as initial value. 
			anglesBeforeInterpolation.clear();
			
			tempDirection[0] = sinf(interpolatedDirection[0]) * cosf(interpolatedDirection[1]);
			tempDirection[1] = sinf(interpolatedDirection[0]) * sinf(interpolatedDirection[1]);
			tempDirection[2] = cosf(interpolatedDirection[0]);
			//	tempDirection already normalized!!!
			// use weights as interpolatin of angles...
			// add 
			//deallocate memory

			}
			// Get the AI scalar at the seed point position
			//MaxFinder.getGFA(&(currentPoint.AI));
			this->interpolateScalar(&(currentPoint.AI), weights);
			// Set the total distance to zero
			currentPoint.D = 0.0;

			// Re-add the seed point  
			pointList->push_back(currentPoint);

			// Set the previous point equal to the current point
			prevPoint = currentPoint;

			// Initialize the previous segment to zero
			this->newSegment[0] = tempDirection[0]; // 0.0;
			this->newSegment[1] = tempDirection[1];// 0.0;
			this->newSegment[2] = tempDirection[2];//0.0;
			
			this->prevSegment[0]=this->prevSegment[1]=this->prevSegment[2]= 0.0; // CHECK!!!

			// Loop until a stopping condition is met


			cout <<"prev segment before:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
			cout <<"new segment before:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
			cout <<"currentpoint before:" << this->currentPoint.X[0] << " " << this->currentPoint.X[1]  << " "<< this->currentPoint.X[2]  << endl;

			// Compute the next point (nextPoint) of the fiber using a Euler step.
			if (!this->solveIntegrationStepSHDI(currentCell, currentCellId, weights)) //Add NEwSegment to Current Point to Determine NEXT Point!!!
			{ 
				cout << "Problem at the first integratioon step"<< endl; 
				return; 
			} 
			cout <<"nextpoimt after:" << this->nextPoint.X[0] << " " << this->nextPoint.X[1]  << " "<< this->nextPoint.X[2]  << endl;

			// Update the current and previous points
			this->prevPoint = this->currentPoint;
			this->currentPoint = this->nextPoint;

			// Update the previous line segment
			this->prevSegment[0] = this->newSegment[0];
			this->prevSegment[1] = this->newSegment[1];
			this->prevSegment[2] = this->newSegment[2];

			//create a maximum finder
			//	M
			//initial regionlist includes all points not some points
			if(meshPtIndexList.size()==0)
				for(int i=0;i<anglesArray.size();i++)
					meshPtIndexList.push_back(i);
			// Check AI values of initial step, otherwise we cannot check the dot product etc
			while (1) 
			{

				cout << endl << "===== while ======== " << direction << " ==" << endl;

				// Check if we've moved to a new cell. NEXT POINT is USE DTO FIND CURRENT CELL!!
				vtkIdType newCellId = this->HARDIimageData->FindCell(currentPoint.X, currentCell, currentCellId,this->tolerance, subId, pCoords, weights);
				if(this->breakLoop) break;
				//if(this->printStepInfo)
				//{
				//	cout << "newCellId"<< newCellId <<   endl;
				//	for (unsigned int i = 0; i <8; ++i)// angles array is constant for all voxels
				//	{
				//		cout <<  "weight[" << i << "]:" << weights[i] << endl;
				//	}

				//}
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
					//this->maximaArrayFromFile->GetTuples(currentCell->PointIds, maximasCellFromFile);
					//outUnitVectorListFromFile is a vector of double array. array 0 is the array of largest unitvector of each vox
					// HARDIdeterministicTracker::FormMaxDirectionArrays(vtkImageData *maximaVolume) fills outUnitVectorListFromFile vector of arraypointers
					for(unsigned int nr = 0; nr <outUnitVectorListFromFile.size()  ; nr++)
					{
						this->outUnitVectorListFromFile.at(nr)->GetTuples(currentCell->PointIds, unitVectorCellListFromFile.at(nr));// get 8 nr th maxima 
					}
				}
				// If we've left the volume, break here
				else if (newCellId == -1)
				{
					cout << "NOT  A CELL; BREAK " << endl; 
					break;
				}


				double *interpolatedVector;
				//unitVectorCellListFromFile used in findFunctionValueUsingMaxFil
				interpolatedVector = findFunctionValueUsingMaximaFile(TRESHOLD, anglesArray, weights,  trianglesArray, meshPtIndexList, maxima);
				                     //NOT USE: findFunctionValueAtPointUsingMaximaFile(pos )  // newCEllId BREAK sorununu coz!!!
				
				                    // USE findRK4DeltaX() 1 tanesi disari cikarsa bulamadim de kes o zaman bastan celli hepsinden once etc...
				testDot = 0.0;
				//value to compare local maxima (either random value or dot product)
				//double value;

				this->newSegment[0] = interpolatedVector[0]; // we will haveone unitvector !!! interpolation of angels will 
				this->newSegment[1] = interpolatedVector[1]; // produce an angle and we will calculate tempDirection!!!!
				this->newSegment[2] = interpolatedVector[2];

				if(this->breakLoop) break;
				if(this->printStepInfo)
				{
					//	cout <<"prev segment1:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
					cout <<"new segment1:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
					cout <<"currentpoint before:" << this->currentPoint.X[0] << " " << this->currentPoint.X[1]  << " "<< this->currentPoint.X[2]  << endl;				
					cout <<"this->step:" << this->step << endl;
					// Compute the next point (nextPoint) of the fiber using a Euler step.
				}

				if (!this->solveIntegrationStepSHDI(currentCell, currentCellId, weights)) //Add NEwSegment to Current Point to Determine NEXT Point!!!
					break;	

				if(this->printStepInfo)
				{
					cout <<"nextpoimt after:" << this->nextPoint.X[0] << " " << this->nextPoint.X[1]  << " "<< this->nextPoint.X[2]  << endl;
					cout <<"prev segment1.1:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
					cout <<"new segment1.1:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
				}	
				// Update the total fiber length
				incrementalDistance = sqrt((double) vtkMath::Distance2BetweenPoints(currentPoint.X, nextPoint.X)); // next point nerede dolar ??

				if(this->printStepInfo)
				{
					cout << "current point: "<< currentPoint.X[0] << " " << currentPoint.X[1] << " " << currentPoint.X[2] << " " << endl;
					cout << "next point: "<< nextPoint.X[0] << " " << nextPoint.X[1] << " " << nextPoint.X[2] << " " << endl;
					//cout << "incremental Distance" << incrementalDistance << endl;
				}
				this->nextPoint.D = this->currentPoint.D + incrementalDistance;
				// Interpolate the AI value at the current position
				if (currentCellId >= 0)
				{
					//MaxFinder.getGFA(&(currentPoint.AI));
					this->interpolateScalar(&(currentPoint.AI), weights); // WEIGHTS are OLD for next point?				 
				}

				testDot = vtkMath::Dot(this->prevSegment, this->newSegment); // stop condition new segment is normalized after the increment for dotproduct

				if(this->printStepInfo)
					cout << "testDot: " << testDot  <<  "current point AI: " << currentPoint.AI << endl;
				// Call "continueTracking" function of parent filter to determine if
				// one of the stopping criteria has been met.
				if (!(this->parentFilter->continueTracking(&(this->currentPoint), testDot, currentCellId)))// Current of NExt Point???
				{
					// If so, stop tracking.
					cout << "STOP TRACKING. testDot: " << testDot  <<   endl;
					break;
				}

				// Add the new point to the point list
				pointList->push_back(this->nextPoint);

				//if(this->printStepInfo)
				//	cout << "pointList.size"<< pointList->size() << endl;
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
				this->prevSegment[2] = this->newSegment[2]; // prevseg becomes automaticly normalized!!!

				if(this->breakLoop) break;
				//	if(this->printStepInfo) {
				//	cout <<"prev segment2:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
				//		cout <<"new segment2:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
				//	}

			} //while 
		}//if

		delete [] weights;
	}


	void HARDIdeterministicTracker::calculateFiberSHDIUseOfflineMaximaDirectionsRK4(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,int numberOfIterations, bool CLEANMAXIMA, double TRESHOLD)
	{

		cout << "-----  New Seed for a New Fiber - calculateFiberSHDI MaxDirection From File ------("<< direction <<")"<<  endl;
		vtkCell *	currentCell			= NULL;						// Cell of current point
		vtkIdType	currentCellId		= 0;						// Id of current cell
		double		closestPoint[3]		= {0.0, 0.0, 0.0};			// Used in "EvaluatePosition"
		double		pointDistance		= 0.0;						// Used in "EvaluatePosition"
		double		incrementalDistance		= 0.0;						// Length of current step
		int			subId				= 0;						// Used in "FindCell"
		double		pCoords[3]			= {0.0, 0.0, 0.0};			// Used in "FindCell"
		double		testDot				= 1.0;						// Dot product for current step
		bool		firstStep			= true;						// True during first integration step
		double  threshold = 0;
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
			currentPoint = pointList->front(); // first point
			pointList->clear();
			this->nextPoint.D = 0.0;
			// Find the cell containing the seed point
			currentCellId = this->HARDIimageData->FindCell(currentPoint.X, NULL, 0, this->tolerance, subId, pCoords, weights); // fills the weights
			currentCell = this->HARDIimageData->GetCell(currentCellId);

			// Set the actual step size, depending on the voxel size
			this->step = direction * this->stepSize * sqrt((double) currentCell->GetLength2());

			// Load the HARDI cell info and AI values of the cell into the "cellHARDIData" and
			// "cellAIScalars" arrays, respectively
			this->HARDIArray->GetTuples(currentCell->PointIds, this->cellHARDIData);
			this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );

			//From File /////////////////

			//this->maximaArrayFromFile->GetTuples(currentCell->PointIds, maximasCellFromFile);
			for(unsigned int nr = 0; nr <outUnitVectorListFromFile.size()  ; nr++)
			{
				this->outUnitVectorListFromFile.at(nr)->GetTuples(currentCell->PointIds, unitVectorCellListFromFile.at(nr));
			}
			//create a maximumfinder
			MaximumFinder MaxFinder = MaximumFinder(trianglesArray); // what does this arr do

			//vector to store the Id's if the found maxima on the ODF
			std::vector<int> maxima;
			//vector to store the unit vectors of the found maxima
			std::vector<double *> outputlistwithunitvectors;
			//neede for search space reduction
		
			std::vector<int> meshPtIndexList;
			//list with ODF values
			std::vector<double> ODFlist;
			std::vector<double> ODFlistMaxTwo;

			//get number of SH components
			int numberSHcomponents = HARDIArray->GetNumberOfComponents();

			// Interpolate the SH at the seed point position
			//double * SHAux = new double[numberSHcomponents];
			//this->interpolateSH(SHAux, weights, numberSHcomponents); //not interpolate now
			double * tempSH = new double[numberSHcomponents];


			double *avgMaxAng = new double[2];

			//IF FILE
			double *maximaOfAPointFromFile = new double[this->nMaximaForEachPoint];
			double **unitVectorsOfAPointFromFile = new double*[this->nMaximaForEachPoint];
			for (int j = 0; j < this->nMaximaForEachPoint; ++j)
				unitVectorsOfAPointFromFile[j] = new double[3];


			std::vector<double *> anglesBeforeInterpolation; 

			//initial regionlist includes all points not some points of the ODF
			if(meshPtIndexList.size()==0)
				for(int i=0;i<anglesArray.size();i++)
					meshPtIndexList.push_back(i);
			for (int j = 0; j < 8; ++j)// vertices if cell or voxel
			{
				//get the SH

				this->cellHARDIData->GetTuple(j, tempSH);
				//this->cellHARDIData has 8 hardi coeffieint sets
				//get the ODF // get maxes like below 8 times

				//IF FROM FILE
				//this->maximasCellFromFile->GetTuple(j,maximaOfAPointFromFile);
				for (int n = 0; n < this->nMaximaForEachPoint; ++n)
					// unitVectorCellListFromFile s each item is an array of 8 tuples each tuple has a unit vector
                    //  unitVectorCellListFromFile.at(0) has 8 vectors each is the longest maxima vector of the correponding vertex
                    //   unitVectorCellListFromFile.at(1) has 8 vectors each is the secong longest vector of the coorresponding vertex
					this->unitVectorCellListFromFile.at(n)->GetTuple(j,unitVectorsOfAPointFromFile[n] );//   unitVectorsOfAPointFromFile consists maxima vectors for a point

				for (int k = 0; k <this->nMaximaForEachPoint; ++k)
				{
					//maxima.push_back(maximaOfAPointFromFile[k]);
					outputlistwithunitvectors.push_back(unitVectorsOfAPointFromFile[k]);  // not necessary can be changed. maxima vectors for a point
				}

				// Angles from unit vectors is a better idea.
				///MaxFinder.getOutput(tempSH, this->parentFilter->shOrder,TRESHOLD, anglesArray,  maxima, meshPtIndexList);// SHAux is empty now we will give 8 differen , radiusun buyuk oldugu yerdeki angellari dizer donen 
				// maxima has ids use them to get angles
				//MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,tempSH, ODFlist, this->unitVectors, anglesArray);
				avgMaxAng[0]=0;
				avgMaxAng[1]=0;
				for(int i=0; i< this->nMaximaForEachPoint; i++)// START FROM HERE!!! this->n
				{	
					//if(i%2==0) {
					avgMaxAng[0]+= acos( outputlistwithunitvectors[i][2]);  // choose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					avgMaxAng[1]+= atan2( outputlistwithunitvectors[i][1],  outputlistwithunitvectors[i][0]);  // ose the angle which is closer to ours keep in an array. Ilk ise elimizde previous yok ...
					//}
					//cout << "anglesOfmaxOfCorner"  << anglesArray.at(maxima.at(i))[0] << " " << anglesArray.at(maxima.at(i))[1] << endl;
				}
				avgMaxAng[0]/=this->nMaximaForEachPoint;
				avgMaxAng[1]/=this->nMaximaForEachPoint;
				//cout << avgMaxAng[0] << " " << avgMaxAng[1] << endl;
				anglesBeforeInterpolation.push_back(avgMaxAng); // if angles are in the range of [-pi,pi] interpolation is ok
				outputlistwithunitvectors.clear();
				// TAKEN BEFORE THE AVERAGING MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,SHAux, ODFlist, this->unitVectors, anglesArray);
				ODFlist.clear();
				maxima.clear();


			}// for cell 8 
			double interpolatedDirection[2];
			this->interpolateAngles(anglesBeforeInterpolation,weights, interpolatedDirection); // this average will be used as initial value. 
			anglesBeforeInterpolation.clear();
			double tempDirection[3];
			tempDirection[0] = sinf(interpolatedDirection[0]) * cosf(interpolatedDirection[1]);
			tempDirection[1] = sinf(interpolatedDirection[0]) * sinf(interpolatedDirection[1]);
			tempDirection[2] = cosf(interpolatedDirection[0]);
			//	tempDirection already normalized!!!
			// use weights as interpolatin of angles...
			// add 
			//deallocate memory


			// Get the AI scalar at the seed point position
			//MaxFinder.getGFA(&(currentPoint.AI));
			this->interpolateScalar(&(currentPoint.AI), weights);
			// Set the total distance to zero
			currentPoint.D = 0.0;

			// Re-add the seed point  
			pointList->push_back(currentPoint);

			// Set the previous point equal to the current point
			prevPoint = currentPoint;

			// Initialize the previous segment to zero
			this->newSegment[0] = tempDirection[0]; // 0.0;
			this->newSegment[1] = tempDirection[1];// 0.0;
			this->newSegment[2] = tempDirection[2];//0.0;
			
			this->prevSegment[0]=this->prevSegment[1]=this->prevSegment[2]= 0.0; // CHECK!!!

			// Loop until a stopping condition is met


			cout <<"prev segment before:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
			cout <<"new segment before:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
			cout <<"currentpoint before:" << this->currentPoint.X[0] << " " << this->currentPoint.X[1]  << " "<< this->currentPoint.X[2]  << endl;

			// Compute the next point (nextPoint) of the fiber using a Euler step.
			if (!this->solveIntegrationStepSHDI(currentCell, currentCellId, weights)) //Add NEwSegment to Current Point to Determine NEXT Point!!!
			{ 
				cout << "Problem at the first integratioon step"<< endl; 
				return; 
			} 
			cout <<"nextpoimt after:" << this->nextPoint.X[0] << " " << this->nextPoint.X[1]  << " "<< this->nextPoint.X[2]  << endl;

			// Update the current and previous points
			this->prevPoint = this->currentPoint;
			this->currentPoint = this->nextPoint;

			// Update the previous line segment
			this->prevSegment[0] = this->newSegment[0];
			this->prevSegment[1] = this->newSegment[1];
			this->prevSegment[2] = this->newSegment[2];

			//create a maximum finder
			//	M
			//initial regionlist includes all points not some points
			if(meshPtIndexList.size()==0)
				for(int i=0;i<anglesArray.size();i++)
					meshPtIndexList.push_back(i);
			// Check AI values of initial step, otherwise we cannot check the dot product etc
			while (1) 
			{

				cout << endl << "===== while ======== " << direction << " ==" << endl;

				// Check if we've moved to a new cell. NEXT POINT is USE DTO FIND CURRENT CELL!!
				vtkIdType newCellId = this->HARDIimageData->FindCell(currentPoint.X, currentCell, currentCellId,this->tolerance, subId, pCoords, weights);
				if(this->breakLoop) break;
				//if(this->printStepInfo)
				//{
				//	cout << "newCellId"<< newCellId <<   endl;
				//	for (unsigned int i = 0; i <8; ++i)// angles array is constant for all voxels
				//	{
				//		cout <<  "weight[" << i << "]:" << weights[i] << endl;
				//	}

				//}
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
					//this->maximaArrayFromFile->GetTuples(currentCell->PointIds, maximasCellFromFile);
					//outUnitVectorListFromFile is a vector of double array. array 0 is the array of largest unitvector of each vox
					// HARDIdeterministicTracker::FormMaxDirectionArrays(vtkImageData *maximaVolume) fills outUnitVectorListFromFile vector of arraypointers
					for(unsigned int nr = 0; nr <outUnitVectorListFromFile.size()  ; nr++)
					{
						this->outUnitVectorListFromFile.at(nr)->GetTuples(currentCell->PointIds, unitVectorCellListFromFile.at(nr));// get 8 nr th maxima 
					}
				}
				// If we've left the volume, break here
				else if (newCellId == -1)
				{
					cout << "NOT  A CELL; BREAK " << endl; 
					break;
				}


				double *interpolatedVector;
				//unitVectorCellListFromFile used in findFunctionValueUsingMaxFil
				interpolatedVector = findRK4DeltaX(currentPoint.X, currentCell, currentCellId,TRESHOLD, anglesArray, weights,  trianglesArray, meshPtIndexList, maxima);
				                      // newCEllId BREAK sorununu coz!!!
				
				                    // USE findRK4DeltaX() 1 tanesi disari cikarsa bulamadim de kes o zaman bastan celli hepsinden once etc...
				testDot = 0.0;
				//value to compare local maxima (either random value or dot product)
				//double value;

				this->newSegment[0] = interpolatedVector[0]; // we will haveone unitvector !!! interpolation of angels will 
				this->newSegment[1] = interpolatedVector[1]; // produce an angle and we will calculate tempDirection!!!!
				this->newSegment[2] = interpolatedVector[2];

				if(this->breakLoop) break;
				if(this->printStepInfo)
				{
					//	cout <<"prev segment1:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
					cout <<"new segment1:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
					cout <<"currentpoint before:" << this->currentPoint.X[0] << " " << this->currentPoint.X[1]  << " "<< this->currentPoint.X[2]  << endl;				
					cout <<"this->step:" << this->step << endl;
					// Compute the next point (nextPoint) of the fiber using a Euler step.
				}

				if (!this->solveIntegrationStepSHDI(currentCell, currentCellId, weights)) //Add NEwSegment to Current Point to Determine NEXT Point!!!
					break;	

				if(this->printStepInfo)
				{
					cout <<"nextpoimt after:" << this->nextPoint.X[0] << " " << this->nextPoint.X[1]  << " "<< this->nextPoint.X[2]  << endl;
					cout <<"prev segment1.1:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
					cout <<"new segment1.1:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
				}	
				// Update the total fiber length
				incrementalDistance = sqrt((double) vtkMath::Distance2BetweenPoints(currentPoint.X, nextPoint.X)); // next point nerede dolar ??

				if(this->printStepInfo)
				{
					cout << "current point: "<< currentPoint.X[0] << " " << currentPoint.X[1] << " " << currentPoint.X[2] << " " << endl;
					cout << "next point: "<< nextPoint.X[0] << " " << nextPoint.X[1] << " " << nextPoint.X[2] << " " << endl;
					//cout << "incremental Distance" << incrementalDistance << endl;
				}
				this->nextPoint.D = this->currentPoint.D + incrementalDistance;
				// Interpolate the AI value at the current position
				if (currentCellId >= 0)
				{
					//MaxFinder.getGFA(&(currentPoint.AI));
					this->interpolateScalar(&(currentPoint.AI), weights); // WEIGHTS are OLD for next point?				 
				}

				testDot = vtkMath::Dot(this->prevSegment, this->newSegment); // stop condition new segment is normalized after the increment for dotproduct

				if(this->printStepInfo)
					cout << "testDot: " << testDot  <<  "current point AI: " << currentPoint.AI << endl;
				// Call "continueTracking" function of parent filter to determine if
				// one of the stopping criteria has been met.
				if (!(this->parentFilter->continueTracking(&(this->currentPoint), testDot, currentCellId)))// Current of NExt Point???
				{
					// If so, stop tracking.
					cout << "STOP TRACKING. testDot: " << testDot  <<   endl;
					break;
				}

				// Add the new point to the point list
				pointList->push_back(this->nextPoint);

				//if(this->printStepInfo)
				//	cout << "pointList.size"<< pointList->size() << endl;
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
				this->prevSegment[2] = this->newSegment[2]; // prevseg becomes automaticly normalized!!!

				if(this->breakLoop) break;
				//	if(this->printStepInfo) {
				//	cout <<"prev segment2:" << this->prevSegment[0] << " " << this->prevSegment[1] << " "<< this->prevSegment[2] << endl;
				//		cout <<"new segment2:" << this->newSegment[0] << " " << this->newSegment[1] << " "<< this->newSegment[2] << endl;
				//	}

			} //while 
		}//if

		delete [] weights;
	}


	double *HARDIdeterministicTracker::findFunctionValue(int threshold, std::vector<double*> &anglesArray, double *weights,  vtkIntArray *trianglesArray, std::vector<int> &meshPointsList, std::vector<int> &maxima)

	{
		std::vector<double> ODFlist; // null can be used 


		std::vector<double *> outputlistwithunitvectors;
		int numberSHcomponents = HARDIArray->GetNumberOfComponents();
		double * tempSH = new double[numberSHcomponents];
		double **avgMaxVect = new double*[8];
		std::vector<double *> anglesBeforeInterpolation; // this consists 8 angles
		MaximumFinder MaxFinder = MaximumFinder(trianglesArray); // while icinde gerek var mi?!!!

		for (int j = 0; j < 8; ++j)
		{
			//get the SH
			avgMaxVect[j] = new double[3];
			this->cellHARDIData->GetTuple(j, tempSH); //fill tempSH
			avgMaxVect[j][0]=0;
			avgMaxVect[j][1]=0;
			avgMaxVect[j][2]=0;
			//this->cellHARDIData has 8 hardi coeffieint sets
			//get the ODF // get maxes like below 8 times


			//get maxima
			MaxFinder.getOutput(tempSH, this->parentFilter->shOrder,threshold, anglesArray,  maxima, meshPointsList);// SHAux is empty now we will give 8 differen , radiusun buyuk oldugu yerdeki angellari dizer donen 

			//Below 3 necessary?
			outputlistwithunitvectors.clear();
			//remove repeated maxima
			MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,tempSH, ODFlist, this->unitVectors, anglesArray);
			// maxima has ids use them to  get angles

			double value =-1 , angularSimilarity = -1;
			int indexHighestSimilarity=-1;
			//	cout << "choose the max with highest angular similarities" << endl;
			for( int i=0;i< outputlistwithunitvectors.size()  ;i++ )
			{ 
				angularSimilarity = vtkMath::Dot(this->newSegment, outputlistwithunitvectors.at(i)); // new segment is actually old increment for coming to xextpoint.

				if( value < angularSimilarity   ) 
				{  
					value = angularSimilarity; indexHighestSimilarity = i; cout << value << " ";

				}
			}
			if (!(maxima.size() > 0) || (indexHighestSimilarity==-1) )	
			{
				cout << "No Maxima or no similarity" << endl;
				break;
			}
			avgMaxVect[j][0]=outputlistwithunitvectors[indexHighestSimilarity][0];
			avgMaxVect[j][1]=outputlistwithunitvectors[indexHighestSimilarity][1];
			avgMaxVect[j][2]=outputlistwithunitvectors[indexHighestSimilarity][2];
			//avgMaxAng[j][0] = acos( outputlistwithunitvectors[indexHighestSimilarity][2]);
			//avgMaxAng[j][1] = atan2( outputlistwithunitvectors[indexHighestSimilarity][1],  outputlistwithunitvectors[indexHighestSimilarity][0]);
			//cout << "angles w/ highest ang sim."<< avgMaxAng[j][0] << " " << avgMaxAng[j][1] << " n highest :" << indexHighestSimilarity << endl;
			anglesBeforeInterpolation.push_back(avgMaxVect[j]); // give real pointers here DELETED!!!!
			//outputlistwithunitvectors.clear();
			//if no maxima are found
			// here interpolate and find the vectir?


			maxima.clear();
			ODFlist.clear();
		}// for cell 8 


		double interpolatedVector[3];
		this->interpolateVectors(anglesBeforeInterpolation,weights, interpolatedVector); // this average will be used as initial value. 
		anglesBeforeInterpolation.clear(); // INTERPOLATE VECTORS !!!
		return interpolatedVector;
	}

	// 
	double *HARDIdeterministicTracker::findFunctionValueUsingMaximaFile(int threshold, std::vector<double*> &anglesArray, double *weights,  vtkIntArray *trianglesArray, std::vector<int> &meshPtIndexList, std::vector<int> &maxima)

	{
		std::vector<double> ODFlist; // null can be used 


		std::vector<double *> outputlistwithunitvectors;
		int numberSHcomponents = HARDIArray->GetNumberOfComponents();
		double * tempSH = new double[numberSHcomponents];
		double **avgMaxVect = new double*[8];
		std::vector<double *> anglesBeforeInterpolation; // this consists 8 angles
		MaximumFinder MaxFinder = MaximumFinder(trianglesArray); // while icinde gerek var mi?!!!



		//IF FILE
		double *maximaOfAPointFromFile = new double[this->nMaximaForEachPoint];
		double **unitVectorsOfAPointFromFile = new double*[this->nMaximaForEachPoint];
		for (int j = 0; j < this->nMaximaForEachPoint; ++j)
			unitVectorsOfAPointFromFile[j] = new double[3];


		for (int j = 0; j < 8; ++j)
		{
			//get the SH
			avgMaxVect[j] = new double[3];
			 
			avgMaxVect[j][0]=0;
			avgMaxVect[j][1]=0;
			avgMaxVect[j][2]=0;
			//this->cellHARDIData has 8 hardi coeffieint sets
			//get the ODF // get maxes like below 8 times

			//IF FROM FILE
			//this->maximasCellFromFile->GetTuple(j,maximaOfAPointFromFile); // GetTupleValue instead of GetTuple since int array
			for (int n = 0; n < this->nMaximaForEachPoint; ++n)
				this->unitVectorCellListFromFile.at(n)->GetTuple(j,unitVectorsOfAPointFromFile[n] );

			for (int k = 0; k < this->nMaximaForEachPoint; ++k)
			{
				maxima.push_back(maximaOfAPointFromFile[k]);  // is not used anymore ?
				outputlistwithunitvectors.push_back(unitVectorsOfAPointFromFile[k]);
			}

			//get maxima; 
			//MaxFinder.getOutput(tempSH, this->parentFilter->shOrder,threshold, anglesArray,  maxima, meshPtIndexList);// SHAux is empty now we will give 8 differen , radiusun buyuk oldugu yerdeki angellari dizer donen 

			//Below 3 necessary?

			//remove repeated maxima
			//MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,tempSH, ODFlist, this->unitVectors, anglesArray);
			// maxima has ids use them to  get angles



			double value =-1 , angularSimilarity = -1;
			int indexHighestSimilarity=-1;
			//	cout << "choose the max with highest angular similarities" << endl;
			for( int i=0;i< outputlistwithunitvectors.size()  ;i++ )
			{ 
				angularSimilarity = vtkMath::Dot(this->newSegment, outputlistwithunitvectors.at(i)); // new segment is actually old increment for coming to xextpoint.

				if( value < angularSimilarity   ) 
				{  
					value = angularSimilarity; indexHighestSimilarity = i; 
					//cout << value << " ";

				}
			}
			if (!(maxima.size() > 0) || (indexHighestSimilarity==-1) )	
			{
				cout << "No Maxima or no similarity" << endl;
				break;
			}
			avgMaxVect[j][0]=outputlistwithunitvectors[indexHighestSimilarity][0];
			avgMaxVect[j][1]=outputlistwithunitvectors[indexHighestSimilarity][1];
			avgMaxVect[j][2]=outputlistwithunitvectors[indexHighestSimilarity][2];
			//avgMaxAng[j][0] = acos( outputlistwithunitvectors[indexHighestSimilarity][2]);
			//avgMaxAng[j][1] = atan2( outputlistwithunitvectors[indexHighestSimilarity][1],  outputlistwithunitvectors[indexHighestSimilarity][0]);
			//cout << "angles w/ highest ang sim."<< avgMaxAng[j][0] << " " << avgMaxAng[j][1] << " n highest :" << indexHighestSimilarity << endl;
			anglesBeforeInterpolation.push_back(avgMaxVect[j]); // give real pointers here DELETED!!!!
			//outputlistwithunitvectors.clear();
			//if no maxima are found
			// here interpolate and find the vectir?

			outputlistwithunitvectors.clear();
			maxima.clear();
			ODFlist.clear();
		}// for cell 8 


		double interpolatedVector[3];
		this->interpolateVectors(anglesBeforeInterpolation,weights, interpolatedVector); // this average will be used as initial value. 
		anglesBeforeInterpolation.clear(); // INTERPOLATE VECTORS !!!
		return interpolatedVector;
	}
 


     // Use this function if  RK4 and maxima unitvecr=tors from file
	double * HARDIdeterministicTracker::findFunctionValueAtPointUsingMaximaFile(double pos[3],vtkCell * currentCell, vtkIdType currentCellId, int threshold, std::vector<double*> &anglesArray,  vtkIntArray *trianglesArray, std::vector<int> &meshPtIndexList, std::vector<int> &maxima){
		double pCoords[3] = { 0.0, 0.0,0.0};
		int subId=0;
		double weights[8];
		double *interpolatedVector = new double[3];
		vtkIdType newCellId = this->HARDIimageData->FindCell(pos,currentCell, currentCellId,this->tolerance, subId, pCoords, weights);

		// If we're in a new cell, and we're still inside the volume...
		//if (newCellId >= 0 && newCellId != currentCellId)
		//{
		//	// ...store the ID of the new cell...
		//	currentCellId = newCellId;

		//	// ...set the new cell pointer...
		//	currentCell = this->HARDIimageData->GetCell(currentCellId);

		//	// ...and fill the cell arrays with the data of the new cell
		//	this->HARDIArray->GetTuples(currentCell->PointIds, this->cellHARDIData);
		//	this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );
		//}
		//// If we've left the volume, break here
		//else 
			if (newCellId == -1)
		{
			interpolatedVector[0]=interpolatedVector[0]=interpolatedVector[0]=0.0;
			return interpolatedVector;
		}
		 // outUnitVectorListFromFile (values for  the whole volume) if filled in formarraysfromfile
		for(unsigned int nr = 0; nr <outUnitVectorListFromFile.size()  ; nr++)
					{
						this->outUnitVectorListFromFile.at(nr)->GetTuples(currentCell->PointIds, unitVectorCellListFromFile.at(nr));// get 8 nr th maxima  for a cell
					}
		int numberSHcomponents = HARDIArray->GetNumberOfComponents();
		
		interpolatedVector =  findFunctionValueUsingMaximaFile(threshold, anglesArray, weights,  trianglesArray, meshPtIndexList, maxima);
		return interpolatedVector;
	}

	double *HARDIdeterministicTracker::findRK4DeltaX(double pos[3],vtkCell * currentCell, vtkIdType currentCellId, int threshold, std::vector<double*> &anglesArray, double *interpolatedVector,  vtkIntArray *trianglesArray, std::vector<int> &meshPtIndexList, std::vector<int> &maxima)
	{
		double H= this->step;
		double *K1= new double[3]; double *K2= new double[3]; double *K3= new double[3]; double *K4= new double[3];
		double *posK1= new double[3]; double *posK2= new double[3]; double *posK3= new double[3]; double *posK4= new double[3];
		double *local_new_segment= new double[3];

		K1 = findFunctionValueAtPointUsingMaximaFile( pos,currentCell, currentCellId,threshold, anglesArray, trianglesArray, meshPtIndexList,maxima);
	    
		posK2[0]=pos[0]+(H/2.0)*K1[0]; 
		posK2[1]=pos[1]+(H/2.0)*K1[1];
		posK2[2]=pos[2]+(H/2.0)*K1[2];
		K2 = findFunctionValueAtPointUsingMaximaFile( posK2,currentCell, currentCellId,threshold, anglesArray, trianglesArray, meshPtIndexList,maxima);

		posK3[0]=pos[0]+(H/2.0)*K2[0]; 
		posK3[1]=pos[1]+(H/2.0)*K2[1]; 
		posK3[2]=pos[2]+(H/2.0)*K2[2];
		K3 = findFunctionValueAtPointUsingMaximaFile( posK3,currentCell, currentCellId,threshold, anglesArray, trianglesArray, meshPtIndexList,maxima);

		posK4[0]=pos[0]+(H)*K3[0]; 
		posK4[1]=pos[1]+(H)*K3[1]; 
		posK4[2]=pos[2]+(H)*K3[2];
		K4 = findFunctionValueAtPointUsingMaximaFile( posK4,currentCell, currentCellId,threshold, anglesArray, trianglesArray, meshPtIndexList,maxima);

		for(int i=0;i<3;i++)
		local_new_segment[i]=  (1/6.0) *  (K1[i] + 2*K2[i] + 2*K3[i] + K4[i]);
		/* this return value should be multiplied by step and added to the original position */ 

		//double K2    = (H * f((x + 1 / 2 * H), (y + 1 / 2 * K1)));  // use new segment
		// double K3    = (H * f((x + 1 / 2 * H), (y + 1 / 2 * K2)));
		// double K4    = (H * f((x + H), (y + K3)));
		//double runge = (y + (1 / 6) *  (K1 + 2 * K2 + 2 * K3 + K4));
		
		//Set the current cell to previous value:
		K1 = findFunctionValueAtPointUsingMaximaFile( pos,currentCell, currentCellId,threshold, anglesArray, trianglesArray, meshPtIndexList,maxima);
	   
		return local_new_segment; //interpolated? NORMALIZE???

	}



	void HARDIdeterministicTracker::calculateFiberDS(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,int numberOfIterations, bool CLEANMAXIMA, double TRESHOLD)
	{
		vtkCell *	currentCell			= NULL;						// Cell of current point
		vtkIdType	currentCellId		= 0;						// Id of current cell
		double		closestPoint[3]		= {0.0, 0.0, 0.0};			// Used in "EvaluatePosition"
		double		pointDistance		= 0.0;						// Used in "EvaluatePosition"
		double		incrementalDistance		= 0.0;						// Length of current step
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
			MaximumFinder MaxFinder = MaximumFinder(trianglesArray);

			//vector to store the Id's if the found maxima on the ODF
			std::vector<int> maxima;
			//vector to store the unit vectors of the found maxima
			std::vector<double *> outputlistwithunitvectors;
			//neede for search space reduction
			bool searchRegion;
			std::vector<int> meshPtIndexList; 
			//list with ODF values
			std::vector<double> ODFlist;

			//get number of SH components
			//	int numberSHcomponents = HARDIArray->GetNumberOfComponents();
			int numberSHcomponents = this->HARDIimageData->GetNumberOfScalarComponents();

			// Interpolate the SH at the seed point position
			double * SHAux = new double[numberSHcomponents];
			//this->HARDIimageData->GetPoint()

			//	cout << "initial interpolation starts" << endl;
			this->interpolateSH(SHAux, weights, numberSHcomponents); // find two maximums then chose closer ones then interpolate

			MaxFinder.getOutputDS(SHAux, numberSHcomponents, anglesArray);// normalizes radii fill this->radii_norm

			//deallocate memory
			delete [] SHAux;

			// Get the AI scalar at the seed point position
			MaxFinder.getGFA(&(currentPoint.AI)); // 

			//cout << "gfa value:" << currentPoint.AI << endl;
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
				this->interpolateSH(SHAux, weights, numberSHcomponents);  // find two maximums then chose closer ones then interpolate


				//create a maximum finder
				MaximumFinder MaxFinder = MaximumFinder(trianglesArray);

				//clear search region list
				meshPtIndexList.clear();
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
						meshPtIndexList.push_back(i);
					}

					if (searchRegion)
					{
						//add search directions to list
						meshPtIndexList.push_back(i);
					}
				}	

				//get local maxima			 
				MaxFinder.getOutputDS(SHAux, numberSHcomponents,TRESHOLD, anglesArray,  maxima, meshPtIndexList);

				//if no maxima are found
				if (!(maxima.size() > 0))	
				{
					//break here
					//	cout << "break"<< endl;
					break;
				}

				//clear vector
				outputlistwithunitvectors.clear();
				ODFlist.clear();

				//if the maxima should be cleaned (double and triple maxima) -> get from UI
				if (CLEANMAXIMA)
				{
					//clean maxima
					MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,SHAux, ODFlist, this->unitVectors, anglesArray);
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
						ODFlist.push_back(MaxFinder.radii_norm[(maxima[i])]);
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
					MaxFinder.getGFA(&(nextPoint.AI));
					//this->interpolateScalar(&(nextPoint.AI), weights);
				}

				// Update the total fiber length
				incrementalDistance = sqrt((double) vtkMath::Distance2BetweenPoints(currentPoint.X, nextPoint.X));
				this->nextPoint.D = this->currentPoint.D + incrementalDistance;

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

	//-------------------------[ solveIntegrationStepSHDI ]------------------------\\


	bool HARDIdeterministicTracker::solveIntegrationStepSHDI(vtkCell * currentCell, vtkIdType currentCellId, double * weights)
	{

		// Compute the next point
		this->nextPoint.X[0] = this->currentPoint.X[0] +  this->newSegment[0] * (this->step);
		this->nextPoint.X[1] = this->currentPoint.X[1] + this->newSegment[1] * (this->step);
		this->nextPoint.X[2] = this->currentPoint.X[2] + this->newSegment[2] * (this->step);

		// Normalize the new line segment
		vtkMath::Normalize(this->newSegment);

		return true;
	}

	bool HARDIdeterministicTracker::solveIntegrationStepSHDIRK4(vtkCell * currentCell, vtkIdType currentCellId, double * weights)
	{

		// Compute the next point
		this->nextPoint.X[0] = this->currentPoint.X[0] +  this->newSegment[0] * (this->step);
		this->nextPoint.X[1] = this->currentPoint.X[1] + this->newSegment[1] * (this->step);
		this->nextPoint.X[2] = this->currentPoint.X[2] + this->newSegment[2] * (this->step);

		// Normalize the new line segment
		vtkMath::Normalize(this->newSegment);

		return true;
	}
	//--------------------------[ interpolateSH ]--------------------------\\

	void HARDIdeterministicTracker::interpolateSH(double * interpolatedSH, double * weights, int numberSHcomponents)
	{
		//cout << "interpolateSH" << endl;
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
				//cout << " data"<< i << ": " << tempSH[i];
				interpolatedSH[i] += weights[j] * tempSH[i];
			}
			//cout << endl << "weight["<< j << ":] " << weights[j] << endl;  
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

	//--------------------------[ interpolateScalar ]--------------------------\\

	void HARDIdeterministicTracker::interpolateAngles(std::vector<double *> &angles, double * weights, double *interpolatedAngle)
	{

		interpolatedAngle[0]=0.0;
		interpolatedAngle[1]=0.0;

		// For all eight surrounding voxels...
		for (int i = 0; i < 8; ++i)
		{


			// ...and add it to the interpolated scalar
			interpolatedAngle[0] += weights[i] * angles.at(i)[0];
			// ...and add it to the interpolated scalar
			interpolatedAngle[1] += weights[i] * angles.at(i)[1];

		}
	}


	//--------------------------[ interpolateScalar ]--------------------------\\

	void HARDIdeterministicTracker::interpolateVectors(std::vector<double *> &angles, double * weights, double *interpolatedVector)
	{

		interpolatedVector[0]=0.0;
		interpolatedVector[1]=0.0;
		interpolatedVector[2]=0.0;
		// For all eight surrounding voxels...
		for (int i = 0; i < 8; ++i)
		{


			// ...and add it to the interpolated scalar
			interpolatedVector[0] += weights[i] * angles.at(i)[0];
			// ...and add it to the interpolated scalar
			interpolatedVector[1] += weights[i] * angles.at(i)[1];
			interpolatedVector[2] += weights[i] * angles.at(i)[2];
		}
	}

	//-----------------------------[ Set unit vectors ]------------------------------\\

	void HARDIdeterministicTracker::setUnitVectors(double ** unitVectors)
	{
		//set unit vectors
		this-> unitVectors = unitVectors;
	}


	// visualisation of maxima directions if read from the file
	void HARDIdeterministicTracker::FormMaxDirectionVisualisation(vtkImageData *maximaVolume)
	{
		// Setup the arrows
    vtkArrowSource  *arrowSource =  vtkArrowSource::New();
  arrowSource->Update();
 vtkGlyph3D *glyphFilter =  vtkGlyph3D::New();
  glyphFilter->SetSourceConnection(arrowSource->GetOutputPort());
  glyphFilter->OrientOn();
  glyphFilter->SetVectorModeToUseVector(); // Or to use Normal

  glyphFilter->SetInput(maximaVolume);

  glyphFilter->Update();
    vtkPolyDataMapper *vectorMapper =  vtkPolyDataMapper::New();
  vectorMapper->SetInputConnection(glyphFilter->GetOutputPort());
   vtkActor *vectorActor =  vtkActor::New();
  vectorActor->SetMapper(vectorMapper);

	}

	// if image has n arrays . each array tuple has a unit vector. First array is for the 
	// largest sized unit vector. Second array is for the 
	void HARDIdeterministicTracker::FormMaxDirectionArrays(vtkImageData *maximaVolume)
	{
			//this->maxUnitVecDataList.at(this->ui->MaxUnitVecDataCombo->currentIndex())->getVtkImageData()
		if(!maximaVolume)
		{
			QString fileName = QFileDialog::getOpenFileName(nullptr,  "Read Maxima File","/", "Max. Unit Vector Image(*.vti)");
			if(fileName.isEmpty() || fileName.isNull())
			{  
				cout << "No file name"<< endl;
				return;
			}
			vtkXMLImageDataReader *readerXML = vtkXMLImageDataReader::New();                

			readerXML->SetFileName( fileName.toStdString().c_str() );
			readerXML->Update(); // Update other place
			maximaVolume = vtkImageData::SafeDownCast(readerXML->GetOutput());
		}
		//int i = readerXML->GetOutput()->GetPointData()->GetArray("maximas")->GetNumberOfComponents();

		// if the image of unit vectors and maxima indexes have been already prepared. 
		QString readArrayName("MaxDirectionUnitVectors");
		 this->nMaximaForEachPoint=0;  

		//if( readerXML->GetOutput()->GetPointData()->GetArray("maximas"))
		//{  
			nMaximaForEachPoint =  maximaVolume->GetPointData()->GetNumberOfArrays() -1;  // 1 for the original image N for the arrays added for unit vectors
			// first array is image scalars
			
			//->GetArray("maximas")->GetNumberOfComponents();
			//maximaArrayFromFile =  vtkIntArray::SafeDownCast(readerXML->GetOutput()->GetPointData()->GetArray("maximas"));
		//}
		
			QString arrName;
			for(unsigned int nr = 0; nr <maximaVolume->GetPointData()->GetNumberOfArrays()  ; nr++)
			{
				QString name(maximaVolume->GetPointData()->GetArrayName(nr));
				cout << name.toStdString() << endl;
				if(name=="") return;
				if ((maximaVolume->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( maximaVolume->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
				{		
					//arrName= readArrayName + QString::number(nr); 
					//outUnitVectorList.at(nr)->SetName( arrName.toStdString().c_str() );  //fist vector array for each point (keeps only the first vector)
					outUnitVectorListFromFile.push_back( vtkDoubleArray::SafeDownCast(maximaVolume->GetPointData()->GetArray( name.toStdString().c_str() )));
					 da=vtkDoubleArray::New();
					int numberOfTuples= outUnitVectorListFromFile.at(outUnitVectorListFromFile.size()-1)->GetNumberOfTuples();
					da->SetNumberOfComponents(3);
					//da->SetNumberOfTuples(numberOfTuples);
					// ADD the opposite of the vectors to form vectors array of opposite direction, there are maxima too.
					for(int i=0; i <numberOfTuples ;i++)
					{  
						double *d = new double[3]; 
						d=outUnitVectorListFromFile.at(outUnitVectorListFromFile.size()-1)->GetTuple(i);
						da->InsertNextTuple3(-1*d[0],-1*d[1],-1*d[2]);
					 
					}
					outUnitVectorListFromFile.push_back(da);
				}
			}
			nMaximaForEachPoint=outUnitVectorListFromFile.size();
		//CELL 
		vtkDataArray* ptr;
		for(unsigned int nr = 0; nr <nMaximaForEachPoint  ; nr++){
			ptr = vtkDataArray::CreateDataArray(  VTK_DOUBLE);
			this->unitVectorCellListFromFile.push_back(ptr); // each has 8 for nr th maxima values

		}
		for(unsigned int nr = 0; nr <nMaximaForEachPoint  ; nr++)
		{
			this->unitVectorCellListFromFile.at(nr)->SetNumberOfComponents(3);//unit vector has 3 components
			this->unitVectorCellListFromFile.at(nr)->SetNumberOfTuples(8);
		}
		// Create the cell arrays
		//this->maximasCellFromFile  = vtkIntArray::SafeDownCast(vtkDataArray::CreateDataArray(this->maximaArrayFromFile->GetDataType()));
		// Set number of components and tuples of the cell arrays
		//this->maximasCellFromFile->SetNumberOfComponents(this->maximaArrayFromFile->GetNumberOfComponents());
		//this->maximasCellFromFile->SetNumberOfTuples(8);




	}


} // namespace bmia

