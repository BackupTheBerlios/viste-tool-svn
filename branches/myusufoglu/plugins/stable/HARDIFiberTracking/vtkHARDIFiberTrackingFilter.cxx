/*
* vtkHARDIFiberTrackingFilter.cxx
*
* 2011-10-14	Anna Vilanova
* - First version. 
*
* 2011-10-31 Bart van Knippenberg
* - Added user-defined variables 
* - Added semi-probabilistic functionality
* - Changed standard stopDotProduct value to 45 degrees
*
*  2013-03-15 Mehmet Yusufoglu, Bart Knippenberg
* -Can process a discrete sphere data which already have Spherical Directions and Triangles arrays. 
* The Execute() function calls different CalculateFiber functions for different data.
* -HARDIFiberTrackingFilter has a data type parameter anymore (sphericalHarmonics 1 or 0), parameter is set depending on the data type read.
*/


/** Includes */

#include "vtkHARDIFiberTrackingFilter.h"



/** Used to compute the "tolerance" variable */
#define TOLERANCE_DENOMINATOR 1000000


namespace bmia {



	vtkStandardNewMacro(vtkHARDIFiberTrackingFilter);



	//-----------------------------[ Constructor ]-----------------------------\\

	vtkHARDIFiberTrackingFilter::vtkHARDIFiberTrackingFilter()
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

		this->StopDotProduct				=	0.7071067818f;

		// Initialize pointers to NULL

		this->HARDIimageData	= NULL;
		this->aiImageData	= NULL;
		this->HARDIPointData	= NULL;
		this->aiPointData	= NULL;
		this->aiScalars		= NULL;
		this->HARDIArray	= NULL;

		// Initialize the ROI name
		this->roiName = "Unnamed ROI";

		//debug options
		bool printStepInfo =0;
		bool breakLoop=0;
	}


	//-----------------------------[ Destructor ]------------------------------\\

	vtkHARDIFiberTrackingFilter::~vtkHARDIFiberTrackingFilter()
	{
		// Reset pointers to NULL

		this->HARDIimageData	= NULL;
		this->aiImageData	= NULL;
		this->HARDIPointData	= NULL;
		this->aiPointData	= NULL;
		this->aiScalars		= NULL;
		this->HARDIArray	= NULL;

		// Clear the point lists

		this->streamlinePointListPos.clear();
		this->streamlinePointListNeg.clear();
	}


	//----------------------------[ SetSeedPoints ]----------------------------\\

	void vtkHARDIFiberTrackingFilter::SetSeedPoints(vtkDataSet * seedPoints)
	{
		// Store the seed point set
		this->vtkProcessObject::SetNthInput(2, seedPoints);
	}


	//----------------------------[ GetSeedPoints ]----------------------------\\

	vtkDataSet * vtkHARDIFiberTrackingFilter::GetSeedPoints()
	{
		// Return a pointer to the seed point set
		if (this->NumberOfInputs < 3)
		{
			return NULL;
		}

		return (vtkDataSet *) (this->Inputs[2]);
	}


	//-----------------------[ SetAnisotropyIndexImage ]-----------------------\\

	void vtkHARDIFiberTrackingFilter::SetAnisotropyIndexImage(vtkImageData * AIImage)
	{
		// Store the Anisotropy Index image
		this->vtkProcessObject::SetNthInput(1, AIImage);
	}


	//-----------------------[ GetAnisotropyIndexImage ]-----------------------\\

	vtkImageData * vtkHARDIFiberTrackingFilter::GetAnisotropyIndexImage()
	{
		// Return a pointer to the Anisotropy Index image
		if (this->NumberOfInputs < 2)
		{
			return NULL;
		}

		return (vtkImageData *) (this->Inputs[1]);
	}


	//--------------------------[ continueTracking ]---------------------------\\

	bool vtkHARDIFiberTrackingFilter:: continueTracking(bmia::HARDIstreamlinePoint * thisPoint, 
		double testDot, vtkIdType currentCellId)
	{
		cout << "ContTracking?:"<< (thisPoint->D <= this->MaximumPropagationDistance) << (testDot >= (double) this->StopDotProduct) << (thisPoint->AI <= this->MaxScalarThreshold) << (thisPoint->AI >= this->MinScalarThreshold) << endl;
		cout << "Intensity current point" << thisPoint->AI << endl;
		return (	currentCellId >= 0									&&	// Fiber has left the volume
			 thisPoint->D <= this->MaximumPropagationDistance	&&	// Maximum fiber length exceeded
			testDot >= (double) this->StopDotProduct			&&	// Maximum fiber angle exceeded
			thisPoint->AI <= this->MaxScalarThreshold		&&  // High scalar value
			thisPoint->AI >= this->MinScalarThreshold		);	// Low scalar value
	}


	//functions which only checks the angle
	bool vtkHARDIFiberTrackingFilter::continueTrackingTESTDOT(double testDot)
	{
		return (testDot >= (double) this->StopDotProduct);	
	}

	//---------------------------[ initializeFiber ]---------------------------\\

	bool vtkHARDIFiberTrackingFilter::initializeFiber(double * seedPoint)
	{
		// Clear the positive and negative point lists
		streamlinePointListPos.clear();
		streamlinePointListNeg.clear();

		// Return "false" if the point is not located inside the volume
		if (HARDIimageData->FindPoint(seedPoint[0], seedPoint[1], seedPoint[2]) < 0)
		{
			return false;
		}

		// Create a new streamline point
		HARDIstreamlinePoint seedSLPoint;

		// Set the coordinates of the seed point
		seedSLPoint.X[0] = seedPoint[0];
		seedSLPoint.X[1] = seedPoint[1];
		seedSLPoint.X[2] = seedPoint[2];

		// Set all other point values to zero

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

	void vtkHARDIFiberTrackingFilter::initializeBuildingFibers()
	{
		// Create and allocate the array containing the fiber points
		vtkPoints *	newPoints = vtkPoints::New();;
		newPoints->Allocate(2500);

		// Get the output data set
		vtkPolyData * output = this->GetOutput();

		// Get the point data of the output
		vtkPointData * outPD = output->GetPointData();;

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

	  

	void vtkHARDIFiberTrackingFilter::Execute()
	{
		 
		//get settings from the ui
		unsigned int NUMBEROFITERATIONS = this->Iterations;
		bool CLEANMAXIMA = this->CleanMaxima;
		double TRESHOLD = this->Treshold;
		unsigned int TESSORDER = this->TesselationOrder;

		// Get the output data set
		vtkPolyData * output = this->GetOutput();

		// Get the input tensor image
		this->HARDIimageData = (vtkImageData *) (this->GetInput());


		// Check if the tensor image exists
		if (!(this->HARDIimageData))
		{
			QMessageBox::warning(NULL, "HARDI Fiber Tracking Filter", "No input HARDI data defined!", 
				QMessageBox::Ok, QMessageBox::Ok);

			return;
		}

		// Get the point data of the tensor image
		this->HARDIPointData = this->HARDIimageData->GetPointData();

		// Check if the point data exists
		if (!(this->HARDIPointData))
		{
			QMessageBox::warning(NULL, "HARDI Fiber Tracking Filter", "No point data for input HARDI data!", 
				QMessageBox::Ok, QMessageBox::Ok);

			return;
		}


		// Get the coefficients
		this->HARDIArray = this->HARDIPointData->GetScalars();

		// Check if there are coefficients
		if (!(this->HARDIArray))
		{
			QMessageBox::warning(NULL, "Fiber Tracking Filter", "No tensors for input DTI data!", 
				QMessageBox::Ok, QMessageBox::Ok);

			return;
		}

		this->shOrder;
		if(this->sphericalHarmonics ==1) // not discrete sphere
		{

			// Get the SH order, based on the number of coefficients
			switch(this->HARDIArray->GetNumberOfComponents())
			{
			case 1:		this->shOrder = 0;	break;
			case 6:		this->shOrder = 2;	break;
			case 15:	this->shOrder = 4;	break;
			case 28:	this->shOrder = 6;	break;
			case 45:	this->shOrder = 8;	break;

			default:
				vtkErrorMacro(<< "Number of SH coefficients is not supported!");
				return;
			}

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
		double tolerance = this->HARDIimageData->GetLength() / TOLERANCE_DENOMINATOR;

		// Create the tracker
		HARDIdeterministicTracker * tracker = new HARDIdeterministicTracker;

		// Initialize pointers and parameters of the tracker
		tracker->initializeTracker(		this->HARDIimageData, this->aiImageData, 
			this->HARDIArray,	this->aiScalars, 
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

		//if semi-probabilistic tracking
		if (NUMBEROFITERATIONS > 1)
		{
			//seed the random number sequence with the time (sort-of random)
			srand((unsigned)time(NULL));
		}

		//create geometry object
		bmia::vtkGeometryGlyphFromSHBuilder* obj = bmia::vtkGeometryGlyphFromSHBuilder::New();
		//compute geometry

		double **unitVectors; // = obj->getUnitVectors();
		std::vector<double*> anglesArray; // = obj->getAnglesArray();
		vtkIntArray *trianglesArray ;
		vtkDoubleArray * anglesArrayVTK; 

		if(this->sphericalHarmonics) 
		{
			//cout << "Spherical Harmonics Data" << endl;
			obj->computeGeometry(TESSORDER);
			// get unit vectors, angles and triangles
			unitVectors = obj->getUnitVectors();
			anglesArray = obj->getAnglesArray();
			trianglesArray = obj->getTrianglesArray();
		}
		else //discrete sphere
		{

			//this->computeGeometryFromDirections( unitVectors, anglesArray ,  trianglesArray);


			vtkPointData * imagePD = this->HARDIimageData->GetPointData(); //point data
			if ((imagePD->GetArray("Spherical Directions")))
			{
				//cout << " Spherical directions ie angles already in the discrete sphere data." << endl;
				anglesArrayVTK = vtkDoubleArray::SafeDownCast(imagePD->GetArray("Spherical Directions"));

			}
			if ( (imagePD->GetArray("Triangles")) )
			{
				//cout << " Triangles exist in the discrete sphere data. " << endl;
				trianglesArray =vtkIntArray::SafeDownCast( imagePD->GetArray("Triangles") ); //SafeDownCast!!
			}

			 
			//cout << "Number of angle tuples ie directions:" << anglesArrayVTK->GetNumberOfTuples()  << endl;
			unitVectors = new double*[anglesArrayVTK->GetNumberOfTuples() ];

			// Loop through all angles, convert polar to cartesian
			for (int i = 0; i < anglesArrayVTK->GetNumberOfTuples(); ++i)
			{
				unitVectors[i] = new double[3];

				// Get the two angles (azimuth and zenith)
				double * angles = anglesArrayVTK->GetTuple2(i);
				anglesArray.push_back(angles);
				// Compute the 3D coordinates for these angles on the unit sphere
				unitVectors[i][0] = sinf(angles[0]) * cosf(angles[1]);
				unitVectors[i][1] = sinf(angles[0]) * sinf(angles[1]);
				unitVectors[i][2] = cosf(angles[0]);

			}

		}//if ends

		//set the unit vectors
		tracker->setUnitVectors(unitVectors);

		//for every iteration
		for (unsigned int iterations = 0; iterations < NUMBEROFITERATIONS; ++iterations)
		{
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
				if(this->sphericalHarmonics) {
					if(true)
					{
					tracker->calculateFiberSHDI( 1, &streamlinePointListPos, anglesArray, trianglesArray, NUMBEROFITERATIONS, CLEANMAXIMA, TRESHOLD);
					tracker->calculateFiberSHDI(-1, &streamlinePointListNeg, anglesArray, trianglesArray, NUMBEROFITERATIONS, CLEANMAXIMA, TRESHOLD);
					}
					else {
					tracker->calculateFiber( 1, &streamlinePointListPos, anglesArray, trianglesArray, NUMBEROFITERATIONS, CLEANMAXIMA, TRESHOLD);
					tracker->calculateFiber(-1, &streamlinePointListNeg, anglesArray, trianglesArray, NUMBEROFITERATIONS, CLEANMAXIMA, TRESHOLD);
					}
				}
				else
				{
					tracker->calculateFiberDS( 1, &streamlinePointListPos, anglesArray, trianglesArray, NUMBEROFITERATIONS, CLEANMAXIMA, TRESHOLD);
					tracker->calculateFiberDS(-1, &streamlinePointListNeg, anglesArray, trianglesArray, NUMBEROFITERATIONS, CLEANMAXIMA, TRESHOLD);
				}
				// Get the length of the resulting fiber
				double fiberLength = this->streamlinePointListPos.back().D + this->streamlinePointListNeg.back().D;

				// If the fiber is longer than the minimum length, add it to the output
				if(		(fiberLength > this->MinimumFiberSize) && 
					(this->streamlinePointListPos.size() + this->streamlinePointListNeg.size()) > 2	)
				{
					this->BuildOutput();
				}

				// Update the progress bar (adapted for multiple iterations)
				if ((ptId % progressStepSize) == 0)
					this->UpdateProgress(ptId/ (float) seedPoints->GetNumberOfPoints());
			}
		}

		//print the time to output screen
		//std::cout<<"Time needed to calculate the fibers from "<<seedPoints->GetNumberOfPoints()<<" seed points: "<<((double) clock()-startTime)/((double) CLOCKS_PER_SEC)<<" sec"<<endl;

		// Delete the tracker
		delete tracker;

		// Squeeze the output to regain over-allocated memory
		output->Squeeze();

		// Clear point lists
		this->streamlinePointListPos.clear();
		this->streamlinePointListNeg.clear();
	}

	// if the directions are ready fill trianglesarray and anglesarray. Trianglesarray has the result of triangulation now.
	//Discrete Sphere data comes with its own directions and triangles. No need to use this function.
	void vtkHARDIFiberTrackingFilter::computeGeometryFromDirections(double **unitVectors, std::vector<double*> &anglesArray2 ,vtkIntArray * trianglesArray) {

		//cout << "Compute geometry from directions \n";  
	 
		 
		vtkPoints *directionPoints =  vtkPoints::New();
		QString fileName = QFileDialog::getOpenFileName(0 , "Open File", "", "Data Files (*.dat *.txt )");
		if(!fileName.isEmpty())
			this->readDirectionsFile(directionPoints, fileName.toStdString() );
		else
			qDebug() << "Filename is empty!" << endl;
		//cout << "Directions read \n";
		unitVectors = (double **) malloc(sizeof(double)*3*directionPoints->GetNumberOfPoints());
		for(int i=0; i< directionPoints->GetNumberOfPoints();i++)
		{
			unitVectors[i]=(double *) malloc(sizeof(double)*3);
			double *pt = new double[3];
			directionPoints->GetPoint(i,pt);
			unitVectors[i][0]=pt[0]; unitVectors[i][1]=pt[1];unitVectors[i][2]=pt[2];
		}

		// Triangulate the tessellated sphere
		vtkIntArray *trianglesIndexes = vtkIntArray::New();
		trianglesIndexes->SetName("Triangles");
		SphereTriangulator * triangulator = new SphereTriangulator;
		triangulator->triangulateFromUnitVectors(directionPoints, trianglesIndexes);
		delete triangulator;
		trianglesArray=trianglesIndexes;

		int numberOfTessPoints = directionPoints->GetNumberOfPoints();
		 
		vtkDoubleArray *anglesArray =  vtkDoubleArray::New();
		anglesArray->SetNumberOfComponents(2);
		anglesArray->SetName("DirectionSphericCoord");
		// Store the number of tessellation points (angles)


		// Convert each point to spherical coordinates, and add it to the array
		for (int i = 0; i < numberOfTessPoints; ++i)
		{
			double * sc = new double[2];
			double *pt = new double[3];

			directionPoints->GetPoint(i,pt);
			sc[0] = atan2((sqrt(pow ( pt[0] , 2) + pow(pt[1], 2))), pt[2]);
			sc[1] = atan2(pt[1], pt[0]);
			anglesArray->InsertNextTuple2(sc[0],sc[1]);
			anglesArray2.push_back(sc);
		}
	}

	//if a file is ready to read directions this functions can be used
	void vtkHARDIFiberTrackingFilter::readDirectionsFile( vtkPoints *points, std::string filename)
	{

		std::ifstream fin(filename.c_str());

		std::string line;


		while(std::getline(fin, line))
		{
			double x,y,z;
			std::stringstream linestream;
			linestream << line;
			linestream >> x >> y >> z;

			points->InsertNextPoint(x, y, z);
		}

		fin.close();
	}

	//----------------------------[ SetStopDegrees ]---------------------------\\

	void vtkHARDIFiberTrackingFilter::SetStopDegrees(float StopDegrees)
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

	void vtkHARDIFiberTrackingFilter::BuildOutput()
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
		std::vector<HARDIstreamlinePoint>::reverse_iterator streamlineRIter;

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

			// Insert the coordinates in the point array
			id = newPoints->InsertNextPoint(tempPoint);

			// Save the new point ID in the ID list
			idFiberPoints->InsertNextId(id);
		}

		// Create an iterator for the point list in the negative direction
		std::vector<HARDIstreamlinePoint>::iterator streamlineIter = streamlinePointListNeg.begin();

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

			// Insert the coordinates in the point array
			id = newPoints->InsertNextPoint(tempPoint);

			// Save the new point ID in the ID list
			idFiberPoints->InsertNextId(id);
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

