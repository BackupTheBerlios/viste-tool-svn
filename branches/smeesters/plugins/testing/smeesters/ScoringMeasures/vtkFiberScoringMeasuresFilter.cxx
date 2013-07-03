
/** Includes */

#include "vtkFiberScoringMeasuresFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkFiberScoringMeasuresFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkFiberScoringMeasuresFilter::vtkFiberScoringMeasuresFilter()
{
	// Set default options

}


//------------------------------[ Destructor ]-----------------------------\\

vtkFiberScoringMeasuresFilter::~vtkFiberScoringMeasuresFilter()
{

}

//-------------------------------[ SetInputVolume ]-------------------------------\\

void vtkFiberScoringMeasuresFilter::SetInputVolume(vtkImageData * image)
{
    // Do nothing if the image hasn't changed
	if (this->inputVolume == image)
		return;

	// Unregister the previous image
	//if (this->inputVolume)
	//	this->inputVolume->UnRegister((vtkObjectBase *) this);

	// Store the pointer
	this->inputVolume = image;

	// Register the new image
	//image->Register((vtkObjectBase *) this);
}

//-------------------------------[ Execute ]-------------------------------\\

void vtkFiberScoringMeasuresFilter::Execute()
{
    //
    //      CHECK POLYDATA INPUT
    //

	// Get the input
	vtkPolyData * input = this->GetInput();
	if(!input)
	{
		vtkErrorMacro(<< "Input has not been set.");
		return;
	}

	// Check if the input contains point data
	vtkPointData * inputPD = input->GetPointData();
	if (!inputPD)
	{
		vtkErrorMacro(<< "Input does not have point data.");
		return;
	}

	// Get the points of the input
	vtkPoints * inputPoints = input->GetPoints();
	if (!inputPoints)
	{
		vtkErrorMacro(<< "Input does not have points.");
		return;
	}

	// Get the lines array of the input
	vtkCellArray * inputLines = input->GetLines();
	if (!inputLines)
	{
		vtkErrorMacro(<< "Input does not have lines.");
		return;
	}

	// Get the output
	vtkPolyData * output = this->GetOutput();
	if (!output)
	{
		vtkErrorMacro(<< "Output has not been set.");
		return;
	}

	// Check if the output contains point data
	vtkPointData * outputPD = output->GetPointData();
	if (!outputPD)
	{
		vtkErrorMacro(<< "Output does not have point data.");
		return;
	}

    //
    //      CHECK GLYPH DATA INPUT
    //

	if (!this->inputVolume)
	{
		vtkErrorMacro(<< "Glyph data has not been set.");
		return;
	}

	vtkPointData * imagePD = this->inputVolume->GetPointData();
	if (!imagePD)
	{
		vtkErrorMacro(<< "Point data in glyph data have not been set.");
		return;
	}

	// Get the array containing the glyphs radii
	vtkDoubleArray * radiiArray = vtkDoubleArray::SafeDownCast(imagePD->GetArray("Vectors"));
	if (!radiiArray)
	{
		vtkErrorMacro(<< "Vectors in glyph data have not been set.");
		return;
	}

	// Get the array containing the angles for each glyph vertex
	vtkDoubleArray * anglesArray = vtkDoubleArray::SafeDownCast(imagePD->GetArray("Spherical Directions"));
	if (!anglesArray)
	{
		vtkErrorMacro(<< "Spherical directions in glyph data have not been set.");
		return;
	}

	// Angles array should have two components. Furthermore, the number of sets of
	// angles should match the number of radii.
	if (anglesArray->GetNumberOfComponents() != 2 || anglesArray->GetNumberOfTuples() != radiiArray->GetNumberOfComponents())
	{
	    vtkErrorMacro(<< "Angles and radii arrays in glyph data do not match.");
		return;
	}

    int numberOfAngles = anglesArray->GetNumberOfTuples();
	/*for(int i = 0; i<numberOfAngles; i++)
	{
	    // Get the two angles (azimuth and zenith)
		double * angles = anglesArray->GetTuple2(i);

		double r = radiiArray->GetComponent(0, i);

	    printf("i:%d, anglesarray:[%f,%f], radiiarray:%f \n",i,angles[0],angles[1],r);
	}*/

    //
    //      PREPARE OUTPUT POLYDATA
    //

    // Prepare scalars list
    QList<vtkDoubleArray*> outputScalarsList;
    int numberOfScalarTypes = inputPD->GetNumberOfArrays();
    for(int i = 0; i < numberOfScalarTypes; i++)
    {
        vtkDoubleArray* outputScalars = vtkDoubleArray::New();
        outputScalars->SetName(inputPD->GetArray(i)->GetName());
        outputScalarsList.append(outputScalars);
    }

    // Add new scalar list for SM
    vtkDoubleArray* SMScalars = vtkDoubleArray::New();
    SMScalars->SetName("SM");

	// Create a point set for the output
	vtkPoints * outputPoints = vtkPoints::New();
	output->SetPoints(outputPoints);
	outputPoints->Delete();

	// Create a line array for the output
	vtkCellArray * outputLines = vtkCellArray::New();
	output->SetLines(outputLines);
	outputLines->Delete();

	//
    //      PROCESS
    //

	// Number of points in the current fiber, and a list of its point IDs
	vtkIdType numberOfPoints;
	vtkIdType * pointList;

    // Setup progress bar
    int numberOfCells = inputLines->GetNumberOfCells();
	int progressStep = numberOfCells / 25;
	progressStep += (progressStep == 0) ? 1 : 0;
	this->SetProgressText("Scoring fibers...");
	this->UpdateProgress(0.0);

	// Loop through all input fibers
	for (vtkIdType lineId = 0; lineId < numberOfCells; ++lineId)
	{
	    // Update the progress bar
		if ((lineId % progressStep) == 0)
		{
			this->UpdateProgress((double) lineId / (double) numberOfCells);
		}

        // Get the data of the current fiber
        vtkCell * currentCell = input->GetCell(lineId);
        int numberOfFiberPoints = currentCell->GetNumberOfPoints();

        // Get cell containing fiber
		inputLines->GetNextCell(numberOfPoints, pointList);

		// Create an ID list for the output fiber
		vtkIdList * newFiberList = vtkIdList::New();

        // Previous point coordinates
        double prev_p[3];

		// Current point coordinates
		double p[3];

		double PI = 3.14159265358;

        // Loop through all points in the fiber
		for (int pointId = 0; pointId < numberOfFiberPoints; ++pointId)
		{
            // Get the point ID of the current fiber point
			vtkIdType currentPointId = currentCell->GetPointId(pointId);

			// Copy the point coordinates to the output
			inputPoints->GetPoint(currentPointId, p);
			vtkIdType newPointId = outputPoints->InsertNextPoint(p);
			newFiberList->InsertNextId(newPointId);

            for(int i = 0; i < numberOfScalarTypes; i++)
            {
                // Get the scalar value
                double scalar = inputPD->GetArray(i)->GetTuple1(currentPointId);

                // Copy the scalar value to the output
                outputScalarsList.at(i)->InsertNextTuple1(scalar);
            }

            // Compute score for current point
            if(pointId > 1)
            {
                // Find the corresponding voxel
                vtkIdType imagePointId = this->inputVolume->FindPoint(p[0], p[1], p[2]);

                // Check if the seed point lies inside the image
                if (imagePointId == -1)
                {
                    SMScalars->InsertNextTuple1(0.0);
                }
                else
                {
                    // compute difference vector
                    double dp[3];
                    dp[0] = p[0] - prev_p[0];
                    dp[1] = p[1] - prev_p[1];
                    dp[2] = p[2] - prev_p[2];

                    // transform to spherical coordinates
                    double theta = acos(dp[2]);
                    double phi = atan(dp[1]/dp[0]);

                    // find nearest vector in glyph data angles list
                    double cost = 1e30;
                    int matchedId;
                    for(int i = 0; i<numberOfAngles; i++)
                    {
                        double * angles = anglesArray->GetTuple2(i);
                        double ncost = ((theta+PI) - (angles[0]+PI))*((theta+PI) - (angles[0]+PI)) + ((phi+PI) - (angles[1]+PI))*((phi+PI) - (angles[1]+PI));
                        if(ncost < cost)
                        {
                            cost = ncost;
                            matchedId = i;
                        }
                    }

                    //double* matchedAngles = anglesArray->GetTuple2(matchedId);

                    double radius = radiiArray->GetComponent(imagePointId, matchedId);

                    SMScalars->InsertNextTuple1(radius);

                    //printf("p:%f %f %f, prev_p:%f %f %f, dp:%f %f %f \n", p[0], p[1], p[2], prev_p[0], prev_p[1], prev_p[2], dp[0], dp[1], dp[2]);
                    //printf("pointId: %d, theta:%f, phi:%f \n", pointId, theta, phi);
                    //printf("matched angles: theta:%f, phi:%f radius:%f\n", matchedAngles[0], matchedAngles[1],radius);
                }
            }
            else
            {
                SMScalars->InsertNextTuple1(0.0);
            }

            // Set previous point
            memcpy(prev_p,p,sizeof(p));
		}

		// Add the new fiber to the output
		outputLines->InsertNextCell(newFiberList);

		//break;
	}

	// Add scalar arrays
	for(int i = 0; i < numberOfScalarTypes; i++)
    {
        vtkDoubleArray* outputScalars = outputScalarsList.at(i);
        output->GetPointData()->AddArray(outputScalars);
    }

    // Add SM scalar array
    output->GetPointData()->AddArray(SMScalars);
    output->GetPointData()->SetActiveScalars("SM");

	// Finalize the progress bar
	this->UpdateProgress(1.0);
}

} // namespace bmia
