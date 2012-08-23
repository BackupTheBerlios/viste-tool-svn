/**
 * FiberOutput.cxx
 *
 * 2008-01-17	Jasper Levink
 * - First version
 *
 * 2010-12-15	Evert van Aart
 * - Ported to DTITool3, refactored some code
 *
 */


/** Includes */

#include "FiberOutput.h"


using namespace std;


namespace bmia {



//-----------------------------[ Constructor ]-----------------------------\\

FiberOutput::FiberOutput()
{
	// Set default parameters
	this->dataSource				= DS_Fibers;
	this->perVoxel					= true;
	this->meanAndVar				= true;	
	this->numberOfSelectedROIs		= 0;	
	this->numberOfSelectedMeasures	= 0;
	this->fileName					= NULL;

	// By default, none of the optional outputs are enabled
	this->selectedTensorOutput		= false;
	this->selectedEigenVectorOutput = false;
	this->selectedFiberLengthOutput = false;
	this->selectedFiberVolumeOutput = false;

	// Initialize the information of the tensor image and the eigensystem image
	this->tensorImageInfo.data		= NULL;
	this->tensorImageInfo.name		= "";
	this->eigenImageInfo.data		= NULL;
	this->eigenImageInfo.name		= "";
}



//------------------------------[ Destructor ]-----------------------------\\

FiberOutput::~FiberOutput()
{
	// Clear the input lists
	this->scalarImageList.clear();
	this->seedList.clear();
	this->fiberList.clear();

	// Close the output file
	if (this->outfile.is_open())
	{
		this->outfile.close();
	}
}


//----------------------------[ addScalarImage ]---------------------------\\

void FiberOutput::addScalarImage(vtkObject * data, std::string name)
{
	InputInfo newInfo;
	newInfo.data = data;
	newInfo.name = name;
	this->scalarImageList.push_back(newInfo);
}


//----------------------------[ addSeedPoints ]----------------------------\\

void FiberOutput::addSeedPoints(vtkObject * data, std::string name)
{
	InputInfo newInfo;
	newInfo.data = data;
	newInfo.name = name;
	this->seedList.push_back(newInfo);
}


//------------------------------[ addFibers ]------------------------------\\

void FiberOutput::addFibers(vtkObject * data, std::string name)
{
	InputInfo newInfo;
	newInfo.data = data;
	newInfo.name = name;
	this->fiberList.push_back(newInfo);
}


//----------------------------[ setTensorImage ]---------------------------\\

void FiberOutput::setTensorImage(vtkObject * data, std::string name)
{
	this->tensorImageInfo.data = data;
	this->tensorImageInfo.name = name;
}


//----------------------------[ setEigenImage ]----------------------------\\

void FiberOutput::setEigenImage(vtkObject * data, std::string name)
{
	this->eigenImageInfo.data = data;
	this->eigenImageInfo.name = name;
}


//-----------------------------[ checkInputs ]-----------------------------\\

bool FiberOutput::checkInputs()
{
	// Scalar images
	for (int currentMeasure = 0; currentMeasure < (int) this->scalarImageList.size(); ++currentMeasure)
	{
		// Get the current image
		vtkImageData * currentImage = (vtkImageData *) (this->scalarImageList.at(currentMeasure)).data;

		// Update the image to make sure it contains data
		currentImage->Update();

		// Check image data pointer, point data, and scalar array
		if (!currentImage)
			return false;
		if (!(currentImage->GetPointData()))
			return false;
		if (!(currentImage->GetPointData()->GetScalars()))
			return false;
	}

	// Seed points (ROIs)
	for (int currentROI = 0; currentROI < (int) this->seedList.size(); ++currentROI)
	{
		// Get the current image
		vtkUnstructuredGrid * currentSeeds = (vtkUnstructuredGrid *) (this->seedList.at(currentROI)).data;

		// Check data set pointer, and if the number of points is positive
		if (!currentSeeds)
			return false;
		if (!(currentSeeds->GetNumberOfPoints() > 0))
			return false;
	}

	// Seed points (ROIs)
	for (int currentFiberIndex = 0; currentFiberIndex < (int) this->fiberList.size(); ++currentFiberIndex)
	{
		// Get the current image
		vtkPolyData * currentFibers = (vtkPolyData *) ((this->fiberList.at(currentFiberIndex)).data);

		// Check data set pointer, lines array, number of points, and number of fibers
		if (!currentFibers)
			return false;
		if (!(currentFibers->GetLines()))
			return false;
		if (!(currentFibers->GetNumberOfPoints() > 0))
			return false;
		if (!(currentFibers->GetNumberOfLines() > 0))
			return false;
	}

	// DTI tensors
	vtkImageData * tensorImage = (vtkImageData *) tensorImageInfo.data;

	// Check image data pointer, point data, and tensors array
	if (!tensorImage)
		return false;
	if (!(tensorImage->GetPointData()))
		return false;
	if (!(tensorImage->GetPointData()->GetArray("Tensors")))
		return false;

	// Eigensystem
	vtkImageData * eigenImage = (vtkImageData *) eigenImageInfo.data;

	// Check image data pointer, point data, and eigenvector arrays
	if (!eigenImage)
		return false;
	if (!(eigenImage->GetPointData()))
		return false;
	if (!(eigenImage->GetPointData()->GetArray("Eigenvector 1")))
		return false;
	if (!(eigenImage->GetPointData()->GetArray("Eigenvector 2")))
		return false;
	if (!(eigenImage->GetPointData()->GetArray("Eigenvector 3")))
		return false;

	return true;
}


//-------------------------------[ saveData ]------------------------------\\

string FiberOutput::saveData(char * fileName, DataSourceType ds, bool rPerVoxel, bool rMeanAndVar)
{
	// Store input data
	this->dataSource	= ds;
	this->perVoxel		= rPerVoxel;
	this->meanAndVar	= rMeanAndVar;
	this->fileName		= fileName;

	// Check if the input data set are correct
	if (!(this->checkInputs()))
	{
		return "FiberOutput: Invalid input data set(s)!";
	}

	// Create a progress bar dialog
	QProgressDialog progress("Initializing writing process", QString(), 0, 100);
	progress.setWindowModality(Qt::WindowModal);
	progress.setWindowTitle("Fiber Output");
	progress.setMinimumDuration(0);
	progress.setValue(1);

	// Get number of seed point sets and scalar images
	this->numberOfSelectedROIs     = this->seedList.size();
	this->numberOfSelectedMeasures = this->scalarImageList.size();

	// Compute how many columns the output table will have
	this->computeNumberOfColumnsNeededForMeasures();
	
	// Message returned by the function
	string returnMsg = "";

	// Variables for calculating mean over all ROIs or fibers
	double * AllMeans = new double[this->numberOfColumnsNeededForMeasures];
	double * AllVars  = new double[this->numberOfColumnsNeededForMeasures];

	// Set all values in both arrays to zero
	for(int currentMeasure = 0; currentMeasure < this->numberOfColumnsNeededForMeasures; ++currentMeasure) 
	{
		AllMeans[currentMeasure] = 0;
 		 AllVars[currentMeasure] = 0;
	}

	// Initialize the output file
	this->outputInit();

	// Write the header of the output file
	this->outputHeader();

	// Output ROIs
	if(this->dataSource == DS_ROI) 
	{
		// Update progress bar
		progress.setLabelText("Creating output for ROIs...");

		// Compute step size for progress bar
		int progressStepSize = (int) (this->numberOfSelectedROIs / 100);
		if (progressStepSize == 0)
		{
			progressStepSize = 1;
		}

		// Create the matrices for means and variances. Both have one row per ROI,
		// and one column per selected output measure.

		double ** means = new double * [this->numberOfSelectedROIs];
		double **  vars = new double * [this->numberOfSelectedROIs];

		for(int currentROI = 0; currentROI < this->numberOfSelectedROIs; ++currentROI) 
		{
			means[currentROI] = new double[this->numberOfColumnsNeededForMeasures];
			 vars[currentROI] = new double[this->numberOfColumnsNeededForMeasures];	
		}

		// Number of seed points in all ROIs
		int totalNumberOfPoints = 0;

		// Current seed point set
		vtkUnstructuredGrid * seedData;

		for(int currentROI = 0; currentROI < this->numberOfSelectedROIs; ++currentROI) 
		{
			// Update progress bar
			if ((currentROI % progressStepSize) == 0)
			{
				int progressVal = (int) ((100.0 * (double) currentROI) / (double) this->numberOfSelectedROIs);
				progress.setValue(progressVal);
			}

			// Get the seed data
			InputInfo currentInputInfo = this->seedList.at(currentROI);
			seedData = (vtkUnstructuredGrid * ) currentInputInfo.data;

			totalNumberOfPoints += seedData->GetNumberOfPoints();

			// Output for each ROI
			if(this->perVoxel) 
			{
				// Put every ROI its own worksheet
				this->outputInitWorksheet("ROI " + currentInputInfo.name);
	
				// Filling header row, first three columns are coordinates
				int numberOfColumns = this->numberOfColumnsNeededForMeasures + 3; 
				string * headerRow = new string[numberOfColumns];

			    headerRow[0] = (string) "coordx";
			    headerRow[1] = (string) "coordy";
			    headerRow[2] = (string) "coordz"; 

				// Add the rest of the headers
				this->addColumnHeaders(headerRow, 3);

				// Write the header to the output
				this->outputWriteRow(headerRow, numberOfColumns, 1);

				// Delete the header row
				delete [] headerRow;

				// Output the data, compute the means
				this->outputDataPerVoxel(means[currentROI], seedData->GetPoints(), numberOfColumns); 
				
				// Finalize the worksheet
				this->outputEndWorksheet();

				if(this->meanAndVar)
				{
					// Compute variances for current ROI
					this->calculateVars(vars[currentROI], means[currentROI], seedData->GetPoints(), seedData->GetPoints()->GetNumberOfPoints());
				}

				// Compute means and variances for all ROIs
				for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean) 
				{
					AllMeans[currentMean] += means[currentROI][currentMean] * seedData->GetPoints()->GetNumberOfPoints();
				}

			} // if [perVoxel]

			// If "perVoxel" is false, we only output the mean and variance
			else 
			{
				if(this->meanAndVar) 
				{
					// Compute the mean values for this ROI
					this->calculateMeans(means[currentROI], seedData->GetPoints());

					// Compute the variances for this ROI
					this->calculateVars(vars[currentROI], means[currentROI], seedData->GetPoints(), seedData->GetPoints()->GetNumberOfPoints());

					// Update means and variances for all ROIs
					for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean) 
					{
						AllMeans[currentMean] += means[currentROI][currentMean] * seedData->GetPoints()->GetNumberOfPoints();
					} 
				} // if [meanAndVar]
				else 
				{
					// If both "perVoxel" and "meanAndVar" are false, we cannot output anything,
					// so finalize the worksheet and the output, and return an output message.

					this->outputEndWorksheet();
					this->outputEnd();
					return("No output; Please select Per Voxel and/or Mean and Variance");
				}

			} // else [perVoxel]
			
		} // for [every ROI]

		// Finalize computation of the means and variances
		if(this->meanAndVar) 
		{
			// Update progress bar
			progress.setValue(100);
			progress.setLabelText("Writing Means and Variances...");

			// Divide current values in "AllMeans" array by the total number of seed points.
			for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean)
			{
				AllMeans[currentMean] /= (double) totalNumberOfPoints;
			}
			
			// Create an array for the variances
			double * currentVars = new double[this->numberOfColumnsNeededForMeasures];

			// Loop through all ROIs
			for(int currentROI = 0; currentROI < this->numberOfSelectedROIs; ++currentROI) 
			{
				// Get the seed data
				InputInfo currentInputInfo = this->seedList.at(currentROI);
				seedData = (vtkUnstructuredGrid * ) currentInputInfo.data;

				// Compute variances for this ROI
				this->calculateVars(currentVars, AllMeans, seedData->GetPoints(), totalNumberOfPoints);
				
				// Add the variances to the total variance array
				for(int currentVar = 0; currentVar < this->numberOfColumnsNeededForMeasures; ++currentVar) 
				{
					AllVars[currentVar] += currentVars[currentVar];
				}
			} // for [every ROI]

			delete [] currentVars;

			// Write the means and variances to the output
			this->outputMeanVarAndLength(means, vars, this->numberOfSelectedROIs, AllMeans, AllVars);

		} // if [meanAndVar]

		// Get rid of the temporary arrays
		for(int currentROI = 0; currentROI < this->numberOfSelectedROIs; ++currentROI) 
		{
			delete [] means[currentROI];
			delete []  vars[currentROI];	
		}

		delete [] means;
		delete [] vars;

	} // if [Data source == ROIs]

	// Output fibers
	else if(this->dataSource == DS_Fibers) 
	{
		// Check if a fiber data set has been added
		if (this->fiberList.size() == 0)
		{
			return "Fiber Output: No fibers selected!";
		}

		// Get the fibers
		InputInfo currentInputInfo = this->fiberList.at(0);
		vtkPolyData * fibers = (vtkPolyData *) currentInputInfo.data;

		// Number of fibers
		int numberOfFibers = fibers->GetNumberOfLines();

		// Get the lines array containing the fibers
		vtkCellArray * fiberData = fibers->GetLines();

		// Number of fiber points
		int totalNumberOfPoints = 0;

		// Update progress bar
		progress.setLabelText("Creating output for fibers...");

		// Compute step size for progress bar
		int progressStepSize = (int) (numberOfFibers / 100);
		if (progressStepSize == 0)
		{
			progressStepSize = 1;
		}

		// We should have at least one fiber
		if(numberOfFibers > 0) 
		{
			// Means and variances are calculated per fiber
			double ** means = new double * [numberOfFibers];
			double **  vars = new double * [numberOfFibers];
			
			// Create the matrices for the means and variances. 
			for(int currentFiber = 0; currentFiber < numberOfFibers; ++currentFiber) 
			{
				means[currentFiber] = new double[this->numberOfColumnsNeededForMeasures];
				 vars[currentFiber] = new double[this->numberOfColumnsNeededForMeasures];
			}

			// Create array for the fiber lengths
			double * lengths = new double[numberOfFibers];
			
			// The first four columns of output are the fiber number and the coordinates of the points
			int numberOfColumns = this->numberOfColumnsNeededForMeasures + 4;

			// Initialize header if we want to output data for each voxel
			if(this->perVoxel) 
			{
				// First column is fiber number, next three columns are coordinates
				string * headerRow = new string[numberOfColumns]; 

				headerRow[0] = (string) "fiberNo";
				headerRow[1] = (string) "coordx";
				headerRow[2] = (string) "coordy";
				headerRow[3] = (string) "coordz"; 

				// The next columns are measures
				this->addColumnHeaders(headerRow, 4);
				
				// Initialize the worksheet
				this->outputInitWorksheet("Fibers");

				// Write the headers to the worksheet
  				this->outputWriteRow(headerRow, numberOfColumns, 1);

				// Delete the header row
				delete [] headerRow;

			} // if [perVoxel]

			// Loop through all fibers
			for(int currentCellId = 0; currentCellId < fiberData->GetNumberOfCells(); ++currentCellId) 
			{
				// Update progress bar
				if ((currentCellId % progressStepSize) == 0)
				{
					int progressVal = (int) ((100.0 * (double) currentCellId) / (double) fiberData->GetNumberOfCells());
					progress.setValue(progressVal);
				}

				// Get the current cell
				vtkCell * currentCell = fibers->GetCell(currentCellId);

				// Do nothing if the cell doesn't exist
				if (!currentCell)
					continue;

				// Get the fiber points
				vtkPoints * currentPoints = currentCell->GetPoints();

				// Keep track of the total number of fiber points
				totalNumberOfPoints += currentPoints->GetNumberOfPoints();

				// Compute the fiber length if required
				if(this->selectedFiberLengthOutput) 
				{
					lengths[currentCellId] = computeFiberLength(currentPoints);
				}

				// If required, output data for each point of the fiber
				if(this->perVoxel) 
				{
					// Output data for each point, compute means. Use the fiber ID as label.
					this->outputDataPerVoxel(means[currentCellId], currentPoints, numberOfColumns, currentCellId);

					// Compute variances for this fiber
					this->calculateVars(vars[currentCellId], means[currentCellId], currentPoints, currentPoints->GetNumberOfPoints());

					// Compute means for all fibers
					for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean) 
					{
						AllMeans[currentMean] += means[currentCellId][currentMean] * currentPoints->GetNumberOfPoints();
					}
						
				} // if [perVoxel]

				// Otherwise, compute only the mean and variances
				else 
				{
					if(this->meanAndVar) 
					{
						// Calculate means and variances
						this->calculateMeans(means[currentCellId], currentPoints);
						this->calculateVars(vars[currentCellId], means[currentCellId], currentPoints, currentPoints->GetNumberOfPoints());

						// Compute means for all fibers
						for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean) 
						{
							AllMeans[currentMean] += means[currentCellId][currentMean] * currentPoints->GetNumberOfPoints();
						}
					}
					else 
					{
						// If both "perVoxel" and "meanAndVar" are false, we cannot output anything,
						// so finalize the worksheet and the output, and return an output message.

						this->outputEndWorksheet();
						this->outputEnd();

						return("No output; please select Per Point and/or Mean and Variance");					
					}
				} // else [perVoxel]

			} // for [every line]

			// Finalize the current worksheet
			if(this->perVoxel) 
			{
				this->outputEndWorksheet();
			}

			// Write the means and variances
			if(this->meanAndVar) 
			{
				// Update progress bar
				progress.setValue(0);
				progress.setLabelText("Writing Means and Variances...");

				// Divide the values in the "AllMeans" array by the total number of fiber points
				for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean) 
				{
					AllMeans[currentMean] = AllMeans[currentMean] / (double) totalNumberOfPoints;
				}

				// Create an array for the variances
				double * currentVars = new double[this->numberOfColumnsNeededForMeasures];

				// Loop through all fibers
				for(int currentCellId = 0; currentCellId < fiberData->GetNumberOfCells(); ++currentCellId) 
				{
					// Update progress bar
					if ((currentCellId % progressStepSize) == 0)
					{
						int progressVal = (int) ((100.0 * (double) currentCellId) / (double) fiberData->GetNumberOfCells());
						progress.setValue(progressVal);
					}

					// Get the current cell
					vtkCell * currentCell = fibers->GetCell(currentCellId);

					// Do nothing if the cell doesn't exist
					if (!currentCell)
						continue;

					// Get the fiber points
					vtkPoints * currentPoints = currentCell->GetPoints();

					// Compute the variances for this fiber
					this->calculateVars(currentVars, AllMeans, currentPoints, totalNumberOfPoints);
					
					// Loop through all columns
					for(int currentVar = 0; currentVar < this->numberOfColumnsNeededForMeasures; ++currentVar) 
					{
						// Add the current variance to the total
						AllVars[currentVar] += currentVars[currentVar];
					}

				} // for [every line]

				// Delete the variances array
				delete [] currentVars;
			}

			// Update progress bar
			progress.setValue(100);

			// Write the means and variances to the output
			if(this->meanAndVar || this->selectedFiberLengthOutput) 
			{
				this->outputMeanVarAndLength(means, vars, numberOfFibers, AllMeans, AllVars, lengths);
			}
			
			// Output the volume of the fibers
			this->outputVolume();

			// Delete all temporary arrays
			for(int currentFiber = 0; currentFiber < numberOfFibers; ++currentFiber) 
			{
				delete [] means[currentFiber];
				delete []  vars[currentFiber];
			}

			delete [] means;
			delete [] vars;
			delete [] lengths;

		} // if [numberOfFibers]

		// If the input contains no fibers...
		else 
		{
			// ... finalize the output...
			this->outputEndWorksheet();
			this->outputEnd();

			// ...and return an error message
			return("No output; No fibers found!");				
		}

	} // if [Data source == fibers]

	// Finalize the output
	this->outputEnd();

	// Clean up
	delete [] AllMeans;
	delete [] AllVars;
	
	// Done!
	return returnMsg;
}


//--------------------------[ outputDataPerVoxel ]-------------------------\\

void FiberOutput::outputDataPerVoxel(double * means, vtkPoints * points, int numberOfColumns, int label) 
{
	// Reset the means to zero
	for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean) 
	{
		means[currentMean] = 0.0;
	}

	// Number of points in the input
	int numberOfPoints = points->GetNumberOfPoints();

	// Current point coordinates
	double x[3]; 

	// ID of voxel closest to the current point
	vtkIdType pointId = 0;

	// Data of the current row
	double * dataRow = new double[numberOfColumns];

	// Current column of the output table
	int currentColumn;

	// Index of the current mean
	int currentMeanIndex;

	// Image used to find the points
	vtkImageData * image = (vtkImageData *) this->tensorImageInfo.data;

	// Loop through all input points
	for(int currentPoint = 0; currentPoint < numberOfPoints; ++currentPoint) 
	{
		// Reset column index to the first column
		currentColumn = 0;
	
		// Get the coordinates of the current point
		points->GetPoint(currentPoint, x);

		// First column can contain a label, which is either the point ID or fiber number
		if(label != -1) 
		{
			dataRow[currentColumn] = label;
			currentColumn++;
		}

		// Copy point coordinates to the output
		for(int currentCoord = 0; currentCoord < 3; ++currentCoord) 
		{
			dataRow[currentColumn] = x[currentCoord];
			currentColumn++;
		}

		currentMeanIndex = 0;

		// Find the ID of the voxel closest to the current point
		pointId = image->FindPoint(x);

		// If a point is not valid (because it is not inside the image), we write only zeros
		bool isValid = (pointId != -1);

		// Loop through all selected measures
		for(int currentMeasure = 0; currentMeasure < this->numberOfSelectedMeasures; ++currentMeasure) 
		{
			// Get the current scalar measure image
			InputInfo currentInputInfo = this->scalarImageList.at(currentMeasure);
			vtkImageData * currentScalarImage = (vtkImageData *) currentInputInfo.data;

			// Get the scalar, update the mean
			double scalar = (isValid ? currentScalarImage->GetPointData()->GetScalars()->GetTuple1(pointId) : 0.0);
			dataRow[currentColumn++]   =  scalar;
			means[currentMeanIndex++] += scalar / (double) numberOfPoints;
		}

		// Write tensors to the output
		if (this->selectedTensorOutput) 
		{
			// Current tensor
			double tensor[6];

			// Get the tensor from the input image
			if (isValid)
			{
				((vtkImageData *) this->tensorImageInfo.data)->GetPointData()->GetArray("Tensors")->GetTuple(pointId, tensor);
			}
			else
			{
				// Create zero tensor
				tensor[0] = 0.0;
				tensor[1] = 0.0;
				tensor[2] = 0.0;
				tensor[3] = 0.0;
				tensor[4] = 0.0;
				tensor[5] = 0.0;
			}
			
			// Nine-element tensor
			double tensor9[9];

			// Convert six-element tensor to nine-element tensor
			vtkTensorMath::Tensor6To9(tensor, tensor9);

			// Loop through all nine elements of the tensor
			for(int currentTensorIndex = 0; currentTensorIndex < 9; ++currentTensorIndex) 
			{
				// Add tensor values to output
				dataRow[currentColumn++]   = tensor9[currentTensorIndex];

				// Update mean values
				means[currentMeanIndex++] += tensor9[currentTensorIndex] / (double) numberOfPoints;
			}

		} // if [selectedTensorOutput]

		// Write eigenvectors to the output
		if(this->selectedEigenVectorOutput) 
		{
			// Eigenvectors of current tensor
			double  firstEigenVector[3];
			double secondEigenVector[3];
			double  thirdEigenVector[3];

			// Get the eigenvectors from the eigensystem image
			if (isValid)
			{
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 1")->GetTuple(pointId,  firstEigenVector);
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 2")->GetTuple(pointId, secondEigenVector);
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 3")->GetTuple(pointId,  thirdEigenVector);
			}
			else
			{
				// Set all eigenvectors to zero
				firstEigenVector[0] = 0.0;		secondEigenVector[0] = 0.0;			thirdEigenVector[0] = 0.0;
				firstEigenVector[1] = 0.0;		secondEigenVector[1] = 0.0;			thirdEigenVector[1] = 0.0;
				firstEigenVector[2] = 0.0;		secondEigenVector[2] = 0.0;			thirdEigenVector[2] = 0.0;
			}
			
			for(int currentArrayIndex = 0; currentArrayIndex < 3; ++currentArrayIndex) 
			{
				// Add eigenvectors to output
				dataRow[currentColumn    ] = firstEigenVector [currentArrayIndex];
				dataRow[currentColumn + 3] = secondEigenVector[currentArrayIndex];
				dataRow[currentColumn + 6] = thirdEigenVector [currentArrayIndex];

				// Update the mean values
				means[currentMeanIndex    ] +=  firstEigenVector[currentArrayIndex] / (double) numberOfPoints;
				means[currentMeanIndex + 3] += secondEigenVector[currentArrayIndex] / (double) numberOfPoints;
				means[currentMeanIndex + 6] +=  thirdEigenVector[currentArrayIndex] / (double) numberOfPoints;

				// Increment indices
				currentMeanIndex++;
				currentColumn++;
			}

			currentColumn    += 6;
			currentMeanIndex += 6;

		} // if [selectedEigenVectorOutput]

		// Write the current row to the output
		this->outputWriteRow(dataRow, numberOfColumns);

	} // for [every point]

	// Delete the row data array 
	delete [] dataRow;
}


//------------------------[ outputMeanVarAndLength ]-----------------------\\

void FiberOutput::outputMeanVarAndLength(double ** means, double ** vars, int numberOfElements, double * fullAllMeans, double * fullAllVars, double * lengths) 
{
	// Initialize output of mean and variance
	this->outputInitWorksheet("MeanAndVariance");

	// We need two columns per measure (one for mean, one for variance)
	int numberOfColumns = this->numberOfColumnsNeededForMeasures * 2 + 1;

	// Index of the column containing the fiber length
	int fiberLengthColumnIndex;
  
	// Add one column if we should output the fiber length
	if(this->selectedFiberLengthOutput) 
	{
		fiberLengthColumnIndex = numberOfColumns;
		numberOfColumns++;
	}

	// Create an array of strings for the headers
	string * headerRow = new string[numberOfColumns];

	// Array containing the default names for the measures (without prefixes)
	string * tmpHeaderRow = new string[this->numberOfColumnsNeededForMeasures];

	// Fill in the default names array
	this->addColumnHeaders(tmpHeaderRow, 0);

	// First entry of the header row is empty
	headerRow[0] = "";

	// Start at the second entry
	int currentColumn = 1;

	// Loop through all measures
	for(int measureId = 0; measureId < this->numberOfColumnsNeededForMeasures; ++measureId) 
	{
		// For each measure, create the name for its mean and variance column
		headerRow[currentColumn++] = "Mean " + tmpHeaderRow[measureId];
		headerRow[currentColumn++] = "Var "  + tmpHeaderRow[measureId];
	}

	// Create name for the fiber length column, if needed
	if(this->selectedFiberLengthOutput)
	{
		headerRow[fiberLengthColumnIndex] = "FiberLength";
	}

	// Write column headers
	this->outputWriteRow(headerRow, numberOfColumns, 1);

	// We're done with the header row, so delete it
	delete [] headerRow;
	delete [] tmpHeaderRow;
  
	// One row of data in the output table
	double * dataRow = new double[numberOfColumns - 1]; 

	string label;
  
	// Determine the prefix of the row label
	string labelPrefix;

	if (this->dataSource == DS_ROI)
	{
		labelPrefix = "ROI ";
	}
	else if(this->dataSource == DS_Fibers)
	{
		labelPrefix = "Fiber ";
	}
	else
	{
		// This should never happen
		labelPrefix = "Element ";
	}

	// Loop through all rows
	for(int currentRow = 0; currentRow < numberOfElements; ++currentRow) 
	{
		label = labelPrefix;

		// Copy the data to the row data array
		for(int currentColumn = 0; currentColumn < this->numberOfColumnsNeededForMeasures; ++currentColumn) 
		{
			dataRow[currentColumn * 2    ] = means[currentRow][currentColumn];
			dataRow[currentColumn * 2 + 1] =  vars[currentRow][currentColumn];
		}

		// Use ROIs as data source
		if(this->dataSource == DS_ROI) 
		{
			// Add ROI name
			InputInfo currentInputInfo = this->seedList.at(currentRow);
			label += currentInputInfo.name;
		}
		// Use fibers as data source
		else 
		{
			// Convert "currentRow" to string, append it to the label
			string str;
			stringstream ss;
			ss << currentRow;
			str = ss.str();
			label += str;
		}

		// Copy the fiber length to the row
		if(this->selectedFiberLengthOutput)
		{
			dataRow[fiberLengthColumnIndex - 1] = lengths[currentRow];
		}

		// Write one row to the output
		this->outputWriteRow(dataRow, numberOfColumns - 1, label);

	} // for [every row]

	// Delete the row data array
	delete [] dataRow;
  
	// Create a data array for the "ALL" row
	double * dataRowAll = new double[this->numberOfColumnsNeededForMeasures * 2]; 

	// Set the label for the ALL row
	label = labelPrefix + "ALL";

	// Copy the "ALL" data to the data array
	for(int measureId = 0; measureId < this->numberOfColumnsNeededForMeasures; ++measureId) 
	{
		dataRowAll[measureId * 2    ] = fullAllMeans[measureId];
		dataRowAll[measureId * 2 + 1] =  fullAllVars[measureId];
	}
  
	// Write the "ALL" data array to the output
	this->outputWriteRow(dataRowAll, this->numberOfColumnsNeededForMeasures * 2, label);

	// Delete the "ALL" data array
	delete [] dataRowAll;
  
	// Finalize the worksheet
	this->outputEndWorksheet();
}


//----------------------------[ calculateMeans ]---------------------------\\

void FiberOutput::calculateMeans(double * means, vtkPoints * points) 
{
	// Initialize the means array to zero
	for(int currentMean = 0; currentMean < this->numberOfColumnsNeededForMeasures; ++currentMean) 
	{
		means[currentMean] = 0;
	}

	// Number of points for which we need to compute the mean
	int numberOfPoints = points->GetNumberOfPoints();

	// Coordinates of current point
	double x[3];

	// Id of current point
	vtkIdType pointId = 0;

	// Image used to find the points
	vtkImageData * image = (vtkImageData *) this->tensorImageInfo.data;

	// Loop through all points
	for(int currentPoint = 0; currentPoint < numberOfPoints; ++currentPoint) 
	{
		// Get the current point coordinates
		points->GetPoint(currentPoint, x);
	  
		// Find the corresponding point in the image data
		pointId = image->FindPoint(x);

		// If a point is not valid (because it is not inside the image), we write only zeros
		bool isValid = (pointId != -1);

		// Current column in the output table
		int currentColumn = 0;
	  
		// Loop through all selected measures
		for(int currentMeasure = 0; currentMeasure < this->numberOfSelectedMeasures; ++currentMeasure) 
		{
			// Get the current scalar measure image
			InputInfo currentInputInfo = this->scalarImageList.at(currentMeasure);
			vtkImageData * currentScalarImage = (vtkImageData *) currentInputInfo.data;

			// Update mean for the current measure
			double scalar = (isValid ? currentScalarImage->GetPointData()->GetScalars()->GetTuple1(pointId) : 0.0);
			means[currentColumn++] += scalar / (double) numberOfPoints;
		}

		// Write tensors to the output
		if (this->selectedTensorOutput) 
		{
			// Current tensor
			double tensor[6];

			// Get the tensor from the input image
			if (isValid)
			{
				((vtkImageData *) this->tensorImageInfo.data)->GetPointData()->GetArray("Tensors")->GetTuple(pointId, tensor);
			}
			else
			{
				// Create zero tensor
				tensor[0] = 0.0;
				tensor[1] = 0.0;
				tensor[2] = 0.0;
				tensor[3] = 0.0;
				tensor[4] = 0.0;
				tensor[5] = 0.0;
			}

			// Nine-element tensor
			double tensor9[9];

			// Convert six-element tensor to nine-element tensor
			vtkTensorMath::Tensor6To9(tensor, tensor9);

			for(int currentTensorIndex = 0; currentTensorIndex < 9; ++currentTensorIndex) 
			{
				// Update mean values
				means[currentColumn++] += tensor9[currentTensorIndex] / (double) numberOfPoints;
			}

		} // if [selectedTensorOutput]

		// Write eigenvectors to the output
		if(this->selectedEigenVectorOutput) 
		{
			// Eigenvectors of current tensor
			double  firstEigenVector[3];
			double secondEigenVector[3];
			double  thirdEigenVector[3];

			// Get the eigenvectors from the eigensystem image
			if (isValid)
			{
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 1")->GetTuple(pointId,  firstEigenVector);
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 2")->GetTuple(pointId, secondEigenVector);
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 3")->GetTuple(pointId,  thirdEigenVector);
			}
			else
			{
				// Set all eigenvectors to zero
				firstEigenVector[0] = 0.0;		secondEigenVector[0] = 0.0;			thirdEigenVector[0] = 0.0;
				firstEigenVector[1] = 0.0;		secondEigenVector[1] = 0.0;			thirdEigenVector[1] = 0.0;
				firstEigenVector[2] = 0.0;		secondEigenVector[2] = 0.0;			thirdEigenVector[2] = 0.0;
			}

			for(int currentArrayIndex = 0; currentArrayIndex < 3; ++currentArrayIndex) 
			{
				// Compute means of eigenvectors
				means[currentColumn    ] +=  firstEigenVector[currentArrayIndex] / (double) numberOfPoints;
				means[currentColumn + 3] += secondEigenVector[currentArrayIndex] / (double) numberOfPoints;
				means[currentColumn + 6] +=  thirdEigenVector[currentArrayIndex] / (double) numberOfPoints;
				currentColumn++;
			}
			  
			currentColumn += 6;

		} // if [selectedEigenVectorOutput]

	} // for [every point]

} 


//----------------------------[ calculateVars ]----------------------------\\

void FiberOutput::calculateVars(double * vars, double * means, vtkPoints * points, int divPoints) 
{
	// Coordinates of current point
	double x[3];

	// Id of current point
	vtkIdType pointId = 0;

	// Initialize all variances to zero
	for(int currentVar = 0; currentVar < this->numberOfColumnsNeededForMeasures; ++currentVar) 
	{
		vars[currentVar] = 0.0;
	}

	// Image used to find the points
	vtkImageData * image = (vtkImageData *) this->tensorImageInfo.data;

	// Loop through all points
	for(int currentPoint = 0; currentPoint < points->GetNumberOfPoints(); ++currentPoint) 
	{
		// Get the point coordinates 
		points->GetPoint(currentPoint, x);

		// Find the corresponding point in the image data
		pointId = image->FindPoint(x);

		// If a point is not valid (because it is not inside the image), we write only zeros
		bool isValid = (pointId != -1);

		// Current column in the output table
		int currentColumn = 0;

		// Loop through all selected measures
		for(int currentMeasure = 0; currentMeasure < this->numberOfSelectedMeasures; ++currentMeasure) 
		{
			// Get the current scalar measure image
			InputInfo currentInputInfo = this->scalarImageList.at(currentMeasure);
			vtkImageData * currentScalarImage = (vtkImageData *) currentInputInfo.data;

			// Update the variance
			double scalar = (isValid ? currentScalarImage->GetPointData()->GetScalars()->GetTuple1(pointId) : 0.0);
			vars[currentColumn++] += pow(means[currentColumn] - scalar, 2.0) / (double) (divPoints - 1); 
		}

		// Write tensors to the output
		if (this->selectedTensorOutput) 
		{
			double tensor[6];

			// Get the tensor from the input image
			if (isValid)
			{
				((vtkImageData *) this->tensorImageInfo.data)->GetPointData()->GetArray("Tensors")->GetTuple(pointId, tensor);
			}
			else
			{
				// Create zero tensor
				tensor[0] = 0.0;
				tensor[1] = 0.0;
				tensor[2] = 0.0;
				tensor[3] = 0.0;
				tensor[4] = 0.0;
				tensor[5] = 0.0;
			}

			// Nine-element tensor
			double tensor9[9];

			// Convert six-element tensor to nine-element tensor
			vtkTensorMath::Tensor6To9(tensor, tensor9);

			// Loop through the tensor elements
			for(int currentTensorIndex = 0; currentTensorIndex < 9; ++currentTensorIndex)
			{
				// Update the variance
				vars[currentColumn++] += pow(means[currentColumn] - tensor9[currentTensorIndex], 2.0) / (double) (divPoints - 1);
			}

		} // if [selectedTensorOutput]

		// Write eigenvectors to the output
		if(this->selectedEigenVectorOutput) 
		{
			// Eigenvectors of current tensor
			double  firstEigenVector[3];
			double secondEigenVector[3];
			double  thirdEigenVector[3];

			// Get the eigenvectors from the eigensystem image
			if (isValid)
			{
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 1")->GetTuple(pointId,  firstEigenVector);
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 2")->GetTuple(pointId, secondEigenVector);
				((vtkImageData *) this->eigenImageInfo.data)->GetPointData()->GetArray("Eigenvector 3")->GetTuple(pointId,  thirdEigenVector);
			}
			else
			{
				// Set all eigenvectors to zero
				firstEigenVector[0] = 0.0;		secondEigenVector[0] = 0.0;			thirdEigenVector[0] = 0.0;
				firstEigenVector[1] = 0.0;		secondEigenVector[1] = 0.0;			thirdEigenVector[1] = 0.0;
				firstEigenVector[2] = 0.0;		secondEigenVector[2] = 0.0;			thirdEigenVector[2] = 0.0;
			}

			// Loop through eigenvectors
			for(int currentArrayIndex = 0; currentArrayIndex < 3; ++currentArrayIndex) 
			{
				vars[currentColumn    ] += pow(means[currentColumn    ] -  firstEigenVector[currentArrayIndex], 2.0) / (divPoints - 1);
				vars[currentColumn + 3] += pow(means[currentColumn + 3] - secondEigenVector[currentArrayIndex], 2.0) / (divPoints - 1);
				vars[currentColumn + 6] += pow(means[currentColumn + 6] -  thirdEigenVector[currentArrayIndex], 2.0) / (divPoints - 1);
			  
				currentColumn++;
			}

		} // if [selectedEigenVectorOutput]

	} // for [every point]

} 


//-----------------------------[ outputVolume ]----------------------------\\

void FiberOutput::outputVolume() 
{
	// Check if a fiber set has been added
	if (this->fiberList.size() == 0)
		return;

	// Initialize worksheet for the volume
	this->outputInitWorksheet("Volume");

	// Number of columns
	int numberOfColumns = 1;
  
	// Create header row, write it to the output
	string * headerRow = new string[numberOfColumns];
	headerRow[0] = "Volume";
	this->outputWriteRow(headerRow, numberOfColumns, 1);

	// Delete the header row
	delete [] headerRow;
	headerRow = NULL;
  
	// Label for the data row
	string label = "Fiber ALL";
  
	// Create the row data array
	double * dataRowAll = new double[1]; 

	InputInfo currentInputInfo = this->fiberList.at(0);
	vtkPolyData * fibers = (vtkPolyData *) currentInputInfo.data;

	// Compute the volume of the fibers
	dataRowAll[0] = calculateVolume(fibers);
  
	// Write the volume to the output
	this->outputWriteRow(dataRowAll, 1, label);
  
	// Delete the row data array
	delete [] dataRowAll;
	dataRowAll = NULL;
  
	// Finalize the worksheet
	this->outputEndWorksheet();
}


//---------------------------[ calculateVolume ]---------------------------\\

int FiberOutput::calculateVolume(vtkPolyData * fibers)
{
	// Output volume
	int volume = 0;

	// Create a set of IDs
	set<vtkIdType> IDs;

	// Get the number of points of the fibers
	int numberOfPoints = fibers->GetNumberOfPoints();
				
	// Current fiber point coordinates
	double    x[3];
	vtkIdType voxelId;

	// Image used to find the points
	vtkImageData * image = (vtkImageData *) this->tensorImageInfo.data;

	// Loop through all fiber points
	for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
    {
		//Get the current fiber point
		fibers->GetPoint(ptId, x);

		// Find the closest voxel in the image data
		voxelId = image->FindPoint(x);	
	
		// Do nothing if the point is not inside of the image
		if (voxelId == -1)
			continue;

		// Add the ID of the closest voxel to the set. Since a set can only contain
		// unique elements, duplicate IDs are not added.

		IDs.insert(voxelId);
	}
	
	// Get the volume, which is the number of IDs in the set
	volume = IDs.size();
	return volume;
}


//------------------------------[ outputInit ]-----------------------------\\

void FiberOutput::outputInit() 
{
	
}


//-----------------------------[ outputHeader ]----------------------------\\

void FiberOutput::outputHeader() 
{
	
}


//-------------------------[ outputInitWorksheet ]-------------------------\\

void FiberOutput::outputInitWorksheet(string titel)
{
	
}


//----------------------------[ outputWriteRow ]---------------------------\\

void FiberOutput::outputWriteRow(string * content, int contentSize, int styleID) 
{
	
}


//----------------------------[ outputWriteRow ]---------------------------\\

void FiberOutput::outputWriteRow(double * content, int contentSize, string label, int styleID) 
{
	
}


//--------------------------[ outputEndWorksheet ]-------------------------\\

void FiberOutput::outputEndWorksheet()
{
	
}


//------------------------------[ outputEnd ]------------------------------\\

void FiberOutput::outputEnd()
{
	
}


//---------------------------[ addColumnHeaders ]--------------------------\\

void FiberOutput::addColumnHeaders(string * headerRow, int currentColumn) 
{
	// Loop through all selected scalar measures
	for(int currentMeasure = 0; currentMeasure < this->numberOfSelectedMeasures; ++currentMeasure) 
	{
		// Add the name of the scalar measures
		InputInfo currentInputInfo = this->scalarImageList.at(currentMeasure);
		headerRow[currentColumn++] = currentInputInfo.name;
	}

	// Add nine columns for tensor values
	if(this->selectedTensorOutput) 
	{
		headerRow[currentColumn++] = "TEN11"; 
		headerRow[currentColumn++] = "TEN12";
		headerRow[currentColumn++] = "TEN13";
		headerRow[currentColumn++] = "TEN21";
		headerRow[currentColumn++] = "TEN22";
		headerRow[currentColumn++] = "TEN23";
		headerRow[currentColumn++] = "TEN31";
		headerRow[currentColumn++] = "TEN32";
		headerRow[currentColumn++] = "TEN33";
	}

	// Add nine columns for the eigenvectors
	if(this->selectedEigenVectorOutput)
	{
		headerRow[currentColumn++] = "EVn1_d1";
		headerRow[currentColumn++] = "EVn1_d2";
		headerRow[currentColumn++] = "EVn1_d3";
		headerRow[currentColumn++] = "EVn2_d1";
		headerRow[currentColumn++] = "EVn2_d2";
		headerRow[currentColumn++] = "EVn2_d3";
		headerRow[currentColumn++] = "EVn3_d1";
		headerRow[currentColumn++] = "EVn3_d2";
		headerRow[currentColumn++] = "EVn3_d3";
	}	
} 


//---------------[ computeNumberOfColumnsNeededForMeasures ]---------------\\

void FiberOutput::computeNumberOfColumnsNeededForMeasures() 
{	
	// We need one column for each scalar measure, nine for tensors, and  nine for
	// eigenvectors. We do not count the fiber length column, since it separate from
	// the other measures in the sense that we do not compute its mean and variance;
	// we do not include the fiber volume, since it is written to a separate file/worksheet.
	
	this->numberOfColumnsNeededForMeasures = this->numberOfSelectedMeasures +
												((this->selectedTensorOutput)      ? (9) : (0)) + 
												((this->selectedEigenVectorOutput) ? (9) : (0));
} 


//--------------------------[ computeFiberLength ]-------------------------\\

double FiberOutput::computeFiberLength(vtkPoints * points)
{
	// Fiber length
	double length = 0;

	// Number of points in the point set
	int numberOfPoints = points->GetNumberOfPoints();

	// Current and previous point coordinates
	double currentX[3];
	double prevX[3];

	// Check if the fiber contains points
	if(numberOfPoints > 0) 
	{
		// Get the first point
		points->GetPoint(0, prevX);

		// Loop through the rest of the points
		for(int currentPoint = 1; currentPoint < numberOfPoints; ++currentPoint) 
		{
			// Get the current point coordinates
			points->GetPoint(currentPoint, currentX);

			// Add the distance between the two points to the length
			length += vtkMath::Distance2BetweenPoints(currentX, prevX);
			
			prevX[0] = currentX[0];
			prevX[1] = currentX[1];
			prevX[2] = currentX[2];
		}
	}
	else
	{
		cout << "computeFiberLength: Encountered empty fiber!" << endl;
	}

	return length;
}


} //namespace bmia
