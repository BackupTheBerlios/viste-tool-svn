/*
 * vtkDiscreteSphereToScalarVolumeFilter.cxx
 *
 * 2011-04-29	Evert van Aart
 * - First version.
 *
 * 2011-05-04	Evert van Aart
 * - Added the volume measure.
 *
 */


/** Includes */

#include "vtkDiscreteSphereToScalarVolumeFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkDiscreteSphereToScalarVolumeFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkDiscreteSphereToScalarVolumeFilter::vtkDiscreteSphereToScalarVolumeFilter()
{
	// Set pointers to NULL
	this->trianglesArray	= NULL;
	this->anglesArray		= NULL;
	this->radiiArray		= NULL;
	this->unitVectors		= NULL;

	// Set default parameter values
	this->currentMeasure	= DSPHM_SurfaceArea;
	this->progressStepSize	= 1;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkDiscreteSphereToScalarVolumeFilter::~vtkDiscreteSphereToScalarVolumeFilter()
{
	if (!(this->anglesArray))
		return;

	// Delete the unit vector array
	if (this->unitVectors)
	{
		for (int i = 0; i < this->anglesArray->GetNumberOfTuples(); ++i)
		{
			delete[] (this->unitVectors[i]);
		}

		delete[] this->unitVectors;
		this->unitVectors = NULL;
	}
}


//-------------------------[ getShortMeasureName ]-------------------------\\

QString vtkDiscreteSphereToScalarVolumeFilter::getShortMeasureName(int index)
{
	if (index < 0 || index >= DSPHM_NumberOfMeasures)
		return "ERROR";

	// Return the short name of the selected measure
	switch(index)
	{
		case DSPHM_SurfaceArea:		return "Area";
		case DSPHM_Volume:			return "Volume";
		case DSPHM_Average:			return "Average";
		default:					return "ERROR";
	}
}


//--------------------------[ getLongMeasureName ]-------------------------\\

QString vtkDiscreteSphereToScalarVolumeFilter::getLongMeasureName(int index)
{
	if (index < 0 || index >= DSPHM_NumberOfMeasures)
		return "ERROR";

	// Return the long name of the selected measure
	switch(index)
	{
		case DSPHM_SurfaceArea:		return "Glyph Surface Area";
		case DSPHM_Volume:			return "Glyph Volume";
		case DSPHM_Average:			return "Average Radius";
		default:					return "ERROR";
	}
}


//----------------------------[ SimpleExecute ]----------------------------\\

void vtkDiscreteSphereToScalarVolumeFilter::SimpleExecute(vtkImageData * input, vtkImageData * output)
{
	// Start reporting the progress of this filter
	this->UpdateProgress(0.0);

	if (!input)
	{
		vtkErrorMacro(<<"No input has been set!");
		return;
	}

	if (!output)
	{
		vtkErrorMacro(<<"No output has been set!");
		return;
	}

	// Get the point data of the input
	vtkPointData * inPD = input->GetPointData();

	if (!inPD)
	{
		vtkErrorMacro(<<"Input does not contain point data!");
		return;
	}

	// Get the point data of the output
	vtkPointData * outPD = output->GetPointData();

	if (!outPD)
	{
		vtkErrorMacro(<<"Output does not contain point data!");
		return;
	}

	// Get the number of points in the input image
	int numberOfPoints = input->GetNumberOfPoints();

	if (numberOfPoints < 1)
	{
		vtkWarningMacro(<<"Number of points in the input is not positive!");
		return;
	}

	// Check if the triangles array has been set (if we need it)
	if (this->currentMeasure == DSPHM_SurfaceArea)
	{
		if (!(this->trianglesArray))	
		{
			vtkErrorMacro(<<"Triangles array has not been set!");
			return;
		}
	}

	// Get the array containing the angles of the sample points
	this->anglesArray = vtkDoubleArray::SafeDownCast(inPD->GetArray("Spherical Directions"));

	if (!(this->anglesArray))
	{
		vtkErrorMacro(<<"Spherical directions array has not been set!");
		return;
	}

	// Get the array containing the radius for each sample points per voxel
	this->radiiArray = vtkDoubleArray::SafeDownCast(inPD->GetArray("Vectors"));

	if (!(this->radiiArray))
	{
		vtkErrorMacro(<<"Radius array has not been set!");
		return;
	}

	// Check if the radius array matches the angles array
	if (this->anglesArray->GetNumberOfComponents() != 2 || this->anglesArray->GetNumberOfTuples() != this->radiiArray->GetNumberOfComponents())
	{
		vtkErrorMacro(<<"Radius array does not match angles array!");
		return;
	}

	// Set the dimensions of the output
	int dims[3];
	input->GetDimensions(dims);
	output->SetDimensions(dims);

	// Create the output scalar array
	vtkDoubleArray * outArray = vtkDoubleArray::New();
	outArray->SetNumberOfComponents(1);
	outArray->SetNumberOfTuples(numberOfPoints);
	
	// Compute the step size for the progress bar
	this->progressStepSize = numberOfPoints / 25;
	this->progressStepSize += (this->progressStepSize == 0) ? 1 : 0;

	// Set the progress bar text
	this->SetProgressText("Computing scalar measure for discrete sphere function...");

	// Compute the desired measure
	switch (this->currentMeasure)
	{
		case DSPHM_SurfaceArea:		this->computeSurfaceArea(outArray);		break;
		case DSPHM_Volume:			this->computeVolume(outArray);			break;
		case DSPHM_Average:			this->computeAverageRadius(outArray);	break;

		default:
			vtkErrorMacro(<<"Unknown scalar measure!");
			return;
	}

	// Add the scalar array to the output image
	outPD->SetScalars(outArray);
	outArray->Delete();
}


//--------------------------[ computeUnitVectors ]-------------------------\\

bool vtkDiscreteSphereToScalarVolumeFilter::computeUnitVectors()
{
	if (this->trianglesArray == NULL || this->anglesArray == NULL)
		return false;

	int numberOfTriangles = this->trianglesArray->GetNumberOfTuples();
	int numberOfAngles = this->anglesArray->GetNumberOfTuples();

	// Delete the unit vector array
	if (this->unitVectors)
	{
		for (int i = 0; i < numberOfAngles; ++i)
		{
			delete[] (this->unitVectors[i]);
		}

		delete[] this->unitVectors;
		this->unitVectors = NULL;
	}

	// Allocate new array for the unit vectors
	this->unitVectors = new double*[numberOfAngles];

	// Loop through all angles
	for (int i = 0; i < numberOfAngles; ++i)
	{
		this->unitVectors[i] = new double[3];

		// Get the two angles (azimuth and zenith)
		double * angles = anglesArray->GetTuple2(i);

		// Compute the 3D coordinates for these angles on the unit sphere
		this->unitVectors[i][0] = cosf(angles[0]) * cosf(vtkMath::Pi() / 2.0f - angles[1]);
		this->unitVectors[i][1] = sinf(angles[0]) * cosf(vtkMath::Pi() / 2.0f - angles[1]);
		this->unitVectors[i][2] = sinf(vtkMath::Pi() / 2.0f - angles[1]);
	}

	return true;
}


//--------------------------[ computeSurfaceArea ]-------------------------\\

bool vtkDiscreteSphereToScalarVolumeFilter::computeSurfaceArea(vtkDoubleArray * outArray)
{
	// First, compute the unit vectors
	if (!(this->computeUnitVectors()))
	{
		vtkErrorMacro(<<"Error computing unit vectors!");
		return false;
	}

	// Get the properties of the input image
	int numberOfTriangles = this->trianglesArray->GetNumberOfTuples();
	int numberOfAngles = this->anglesArray->GetNumberOfTuples();
	int numberOfPoints = this->radiiArray->GetNumberOfTuples();

	double d1[3];
	double d2[3];
	int T[3];

	// Loop through all voxels
	for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
	{
		// Get the vector of radii for the current voxel
		double * R = this->radiiArray->GetTuple(ptId);

		double area = 0.0;

		// Loop through all triangles
		for (int triangleId = 0; triangleId < numberOfTriangles; ++triangleId)
		{
			// Get the current triangle
			this->trianglesArray->GetTupleValue(triangleId, T);

			// Compute the area of the triangle as "A = 0.5 * |(V1 - V0) x (V2 - V0)|"
			d1[0] = R[T[1]] * this->unitVectors[T[1]][0] - R[T[0]] * this->unitVectors[T[0]][0];
			d1[1] = R[T[1]] * this->unitVectors[T[1]][1] - R[T[0]] * this->unitVectors[T[0]][1];
			d1[2] = R[T[1]] * this->unitVectors[T[1]][2] - R[T[0]] * this->unitVectors[T[0]][2];

			d2[0] = R[T[2]] * this->unitVectors[T[2]][0] - R[T[0]] * this->unitVectors[T[0]][0];
			d2[1] = R[T[2]] * this->unitVectors[T[2]][1] - R[T[0]] * this->unitVectors[T[0]][1];
			d2[2] = R[T[2]] * this->unitVectors[T[2]][2] - R[T[0]] * this->unitVectors[T[0]][2];

			double cross[3];

			vtkMath::Cross(d1, d2, cross);

			area += 0.5 * vtkMath::Norm(cross);
		}

		// Add the scalar measure value to the output array
		outArray->SetTuple1(ptId, area);

		// Update the progress bar
		if ((ptId % this->progressStepSize) == 0)
		{
			this->UpdateProgress((double) ptId / (double) numberOfPoints);
		}
	}

	return true;
}


//----------------------------[ computeVolume ]----------------------------\\

bool vtkDiscreteSphereToScalarVolumeFilter::computeVolume(vtkDoubleArray * outArray)
{
	// First, compute the unit vectors
	if (!(this->computeUnitVectors()))
	{
		vtkErrorMacro(<<"Error computing unit vectors!");
		return false;
	}

	// Get the properties of the input image
	int numberOfTriangles = this->trianglesArray->GetNumberOfTuples();
	int numberOfAngles = this->anglesArray->GetNumberOfTuples();
	int numberOfPoints = this->radiiArray->GetNumberOfTuples();

	double a[3];
	double b[3];
	double c[3];
	double bxc[3];
	int T[3];

	// Loop through all voxels
	for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
	{
		// Get the vector of radii for the current voxel
		double * R = this->radiiArray->GetTuple(ptId);

		double volume = 0.0;

		// Loop through all triangles
		for (int triangleId = 0; triangleId < numberOfTriangles; ++triangleId)
		{
			// Get the current triangle
			this->trianglesArray->GetTupleValue(triangleId, T);

			a[0] = R[T[0]] * this->unitVectors[T[0]][0];
			a[1] = R[T[0]] * this->unitVectors[T[0]][1];
			a[2] = R[T[0]] * this->unitVectors[T[0]][2];

			b[0] = R[T[1]] * this->unitVectors[T[1]][0];
			b[1] = R[T[1]] * this->unitVectors[T[1]][1];
			b[2] = R[T[1]] * this->unitVectors[T[1]][2];

			c[0] = R[T[2]] * this->unitVectors[T[2]][0];
			c[1] = R[T[2]] * this->unitVectors[T[2]][1];
			c[2] = R[T[2]] * this->unitVectors[T[2]][2];

			// Compute the volume of the tetrahedron formed by the three triangle
			// points and the glyph center. Since the glyph center is defined as
			// {0, 0, 0}, we can use the formula "V = |a . (b x c)| / 6".

			vtkMath::Cross(b, c, bxc);
			volume += abs(vtkMath::Dot(a, bxc)) / 6.0;
		}

		// Add the scalar measure value to the output array
		outArray->SetTuple1(ptId, volume);

		// Update the progress bar
		if ((ptId % this->progressStepSize) == 0)
		{
			this->UpdateProgress((double) ptId / (double) numberOfPoints);
		}
	}

	return true;
}


//-------------------------[ computeAverageRadius ]------------------------\\

bool vtkDiscreteSphereToScalarVolumeFilter::computeAverageRadius(vtkDoubleArray * outArray)
{
	// Get the properties of the input image
	int numberOfAngles = this->anglesArray->GetNumberOfTuples();
	int numberOfPoints = this->radiiArray->GetNumberOfTuples();

	// Loop through all voxels
	for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
	{
		// Get the vector of radii for the current voxel
		double * R = this->radiiArray->GetTuple(ptId);

		double avg = 0.0;

		// Compute the average radius
		for (int angleId = 0; angleId < numberOfAngles; ++angleId)
		{
			avg += R[angleId] / (double) numberOfAngles;
		}

		// Add the scalar measure value to the output array
		outArray->SetTuple1(ptId, avg);

		// Update the progress bar
		if ((ptId % this->progressStepSize) == 0)
		{
			this->UpdateProgress((double) ptId / (double) numberOfPoints);
		}
	}

	return true;
}


} // namespace bmia
