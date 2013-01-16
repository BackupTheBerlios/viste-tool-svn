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
 * vtkGeometryGlyphFromSHBuilder.cxx
 *
 * 2011-05-09	Evert van Aart
 * - First version. 
 * 
 * 2011-05-10	Evert van Aart
 * - Fixed a memory allocation bug.
 * 
 */
 

/** Includes */

#include "vtkGeometryGlyphFromSHBuilder.h"
#include "HARDI/sphereTesselator.h"
#include "HARDI/SphereTriangulator.h"


namespace bmia {


//---------------------------[ Constructor Call ]--------------------------\\

vtkGeometryGlyphFromSHBuilder * vtkGeometryGlyphFromSHBuilder::New()
{
	return new vtkGeometryGlyphFromSHBuilder;
}


//-----------------------------[ Constructor ]-----------------------------\\

vtkGeometryGlyphFromSHBuilder::vtkGeometryGlyphFromSHBuilder()
{
	// Create an input port and configure it
	this->SetNumberOfInputPorts(1);
	this->GetInputPortInformation(0);

	// Initialize pointers to NULL
	this->inputVolume			= NULL;
	this->unitVectors			= NULL;
	this->trianglesArray		= NULL;
	this->scalarVolume			= NULL;

	// Set default options
	this->numberOfAngles		= 0;
	this->nMethod				= NM_MinMax;
	this->nScope				= NS_WholeImage;
	this->scale					= 1.0;
	this->sharpenExponent		= 1.0;
	this->enableSharpening		= true;
	this->glyphType				= GGT_Mesh;
	this->colorMethod			= CM_Direction;
	this->normalizeScalars		= true;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkGeometryGlyphFromSHBuilder::~vtkGeometryGlyphFromSHBuilder()
{
	// Delete all angle pairs from the angles array
	for (std::vector<double *>::iterator i = this->anglesArray.begin(); i != this->anglesArray.end(); ++i)
	{
		double * currentAngles = (*i);
		delete currentAngles;
	}

	// Clear the angles array itself
	this->anglesArray.clear();

	// Delete the unit vector array
	if (this->unitVectors)
	{
		for (int i = 0; i < this->numberOfAngles; ++i)
		{
			delete[] (this->unitVectors[i]);
		}

		delete[] this->unitVectors;
		this->unitVectors = NULL;
	}

	// Delete the triangles array
	if (this->trianglesArray)
	{
		this->trianglesArray->Delete();
		this->trianglesArray = NULL;
	}
}


//-------------------------------[ Execute ]-------------------------------\\

void vtkGeometryGlyphFromSHBuilder::Execute()
{
	// We can't do anything if the geometry has not yet been constructed
	if (this->unitVectors == NULL || this->anglesArray.size() == 0 || this->trianglesArray == NULL)
		return;

	// Set normalization scope to local, because other scopes are not suppored for SH glyphs
	if (this->nScope == NS_WholeImage || this->nScope == NS_SeedPoints)
		this->nScope = NS_Local;

	// Get the seed points
	vtkPointSet * seeds = vtkPointSet::SafeDownCast(this->GetInput(0));

	if (!seeds)
	{
		vtkErrorMacro(<< "Seed points have not been set!");
		return;
	}

	// Check if we've got any seed points
	if (seeds->GetNumberOfPoints() <= 0)
		return;

	// Check if the input volume has been set
	if (!(this->inputVolume))
	{
		vtkErrorMacro(<< "Input volume has not been set!");
		return;
	}

	// Get the scalar array from the input volume
	vtkPointData * shPD = this->inputVolume->GetPointData();

	if (!shPD)
	{
		vtkErrorMacro(<< "Input volume does not contain point data!");
		return;
	}

	vtkDataArray * shCoefficients = shPD->GetScalars();

	if (!shCoefficients)
	{
		vtkErrorMacro(<< "Input volume does not contain a scalar array!");
		return;
	}

	int shOrder;

	// Get the SH order, based on the number of coefficients
	switch(shCoefficients->GetNumberOfComponents())
	{
		case 1:		shOrder = 0;	break;
		case 6:		shOrder = 2;	break;
		case 15:	shOrder = 4;	break;
		case 28:	shOrder = 6;	break;
		case 45:	shOrder = 8;	break;

		default:
			vtkErrorMacro(<< "Number of SH coefficients is not supported!");
			return;
	}

	// Initialize progress bar
	this->UpdateProgress(0.0);
	this->SetProgressText("Initializing...");

	// Minimum and maximum for normalization
	double minNorm = 0.0;
	double maxNorm = 1.0;

	// Get the output of the filter
	vtkPolyData * output = this->GetOutput(0);

	if (!output)
	{
		vtkErrorMacro(<< "Output has not been set!");
		return;
	}

	// Scalar array and its range
	vtkDataArray * scalars = NULL;
	double scalarRange[2] = {0.0, 1.0};

	// If we need the scalar volume for coloring...
	if ((this->colorMethod == CM_WDirection || this->colorMethod == CM_Scalar) && this->scalarVolume)
	{
		// ...check if it contains point data...
		if (this->scalarVolume->GetPointData())
		{
			// ...and if so, select the scalar array.
			scalars = this->scalarVolume->GetPointData()->GetScalars();

			scalars->GetRange(scalarRange);

			// If the range is invalid, set it to 0-1 instead
			if (scalarRange[1] - scalarRange[0] <= 0.0)
			{
				scalarRange[0] = 0.0;
				scalarRange[1] = 1.0;
			}
		}
	}

	// Create a new set of points
	vtkPoints * newPoints = vtkPoints::New();
	newPoints->SetDataTypeToDouble();

	// Create an array for the glyph colors
	vtkDataArray * colors;

	// If we're using direction-based coloring, we want a 3-component array
	if (this->colorMethod == CM_Direction || this->colorMethod == CM_WDirection)
	{
		colors = vtkDataArray::CreateDataArray(VTK_UNSIGNED_CHAR);
		colors->SetNumberOfComponents(3);
	}
	// Otherwise, we use one component per point
	else
	{
		colors = vtkDataArray::CreateDataArray(VTK_DOUBLE);
		colors->SetNumberOfComponents(1);
	}

	// Create a cell array for the topology, which will consist either of lines
	// (from the glyph center to the vertices), or polygons (from the triangle array).

	vtkCellArray * newTopology = vtkCellArray::New();

	// Array containing SH coefficients of the current voxel
	double * coeffs = new double[shCoefficients->GetNumberOfComponents()];

	// Step size for the progress bar
	int progressStepSize = seeds->GetNumberOfPoints() / 25;
	progressStepSize += (progressStepSize > 0) ? 0 : 1;
	this->SetProgressText("Constructing geometry glyphs...");

	// Loop through all seed points
	for (int pointId = 0; pointId < seeds->GetNumberOfPoints(); ++pointId)
	{
		// Update progress bar
		if ((pointId % progressStepSize) == 0)
		{
			this->UpdateProgress((double) pointId / (double) seeds->GetNumberOfPoints());
		}

		// Get the seed point coordinates (glyph center)
		double * p = seeds->GetPoint(pointId);

		// Find the corresponding voxel
		vtkIdType imagePointId = this->inputVolume->FindPoint(p[0], p[1], p[2]);

		// Check if the seed point lies inside the image
		if (imagePointId == -1)
			continue;

		// Get the current SH coefficients
		shCoefficients->GetTuple(imagePointId, coeffs);

		// Use these coefficients, in combination with the array containing the
		// angles of the tessellation points, to compute a deformator, which contains
		// the radius per tessellation point.

		std::vector<double> radii = HARDITransformationManager::CalculateDeformator(coeffs, &(this->anglesArray), shOrder);

		// Check if we succeeded
		if (radii.size() != this->numberOfAngles)
			continue;

		// Compute the minimum and maximum radius 
		this->computeMinMaxRadii(&radii, minNorm, maxNorm);

		// Create a new list for the glyph points
		vtkIdList * glyphPoints = vtkIdList::New();

		// Loop through all unit vectors
		for (int vectorId = 0; vectorId < this->numberOfAngles; ++vectorId)
		{
			// Get the current unit vector
			double * v = this->unitVectors[vectorId];

			// Get the radius at the current voxel for this vector
			double r = radii.at(vectorId);

			// Apply Min/Max normalization. If the normalization method has been
			// set to "NM_Maximum", "minNorm" will be zero, and we will simply
			// divide the radius by "maxNorm".

			r = (r - minNorm) / (maxNorm - minNorm);

			// Sharpen the glyph by computing the power of the radii.
			if (this->enableSharpening)
				r = pow(r, this->sharpenExponent);

			// If we're using radius-based coloring, add the radius to the color array
			if (this->colorMethod == CM_Radius)
				colors->InsertNextTuple1(r);

			// Multiply the final radius (in the range 0-1) by the overall scale
			r *= this->scale;

			// Add the unit vector multiplied by the radius to the glyph center
			vtkIdType newPointId = newPoints->InsertNextPoint(p[0] + v[0] * r, p[1] + v[1] * r, p[2] + v[2] * r);

			// Add the unit vector to the color array
			if (this->colorMethod == CM_Direction || (this->colorMethod == CM_WDirection && scalars == NULL))
			{
				colors->InsertNextTuple3(	abs(v[0] * 255.0), 
											abs(v[1] * 255.0),
											abs(v[2] * 255.0));
			}

			// If we're using weighted direction-based coloring, first compute the weight
			else if (this->colorMethod == CM_WDirection && scalars)
			{
				vtkIdType scalarPointId = this->scalarVolume->FindPoint(p);

				if (scalarPointId > -1)
				{
					double currentScalar = scalars->GetTuple1(scalarPointId);
					double w = (currentScalar - scalarRange[0]) / (scalarRange[1] - scalarRange[0]);

					colors->InsertNextTuple3(	w * abs(v[0] * 255.0), 
												w * abs(v[1] * 255.0),
												w * abs(v[2] * 255.0));
				}
				else
				{
					// If the seed point is located outside of the scalar volume, 
					// we make the glyph black.

					colors->InsertNextTuple3(0.0, 0.0, 0.0);
				}
			}

			// If we're using scalar coloring, simply copy the scalar volume
			else if (this->colorMethod == CM_Scalar)
			{
				if (scalars)
				{
					vtkIdType scalarPointId = this->scalarVolume->FindPoint(p);

					if (scalarPointId > -1)
					{
						double tempScalar = scalars->GetTuple1(scalarPointId);

						if (this->normalizeScalars)
							colors->InsertNextTuple1((tempScalar - scalarRange[0]) / (scalarRange[1] - scalarRange[0]));
						else
							colors->InsertNextTuple1(tempScalar);
					}
					else
					{
						// If the seed point is located outside of the scalar volume, 
						// we insert the scalar value 0.0.

						colors->InsertNextTuple1(0.0);
					}
				}
				else
					// If we don't have a scalar array (for example because the scalar
					// volume combo box is set to "None"), we set all colors to 0.0.
					colors->InsertNextTuple1(0.0);
			}


			// Add the new point ID to the list
			glyphPoints->InsertNextId(newPointId);
		}

		radii.clear();
		// If we don't have triangles, we use lines from the glyph center instead,
		// so we need to add the glyph center as an extra point.

		if (this->trianglesArray == NULL || this->glyphType == GGT_Star)
		{
			vtkIdType newPointId = newPoints->InsertNextPoint(p[0], p[1], p[2]);
			glyphPoints->InsertNextId(newPointId);
		}

		// List used for creating the topology cells.
		vtkIdList * topologyList = vtkIdList::New();

		// Polygonal mesh
		if (this->trianglesArray && this->glyphType == GGT_Mesh)
		{
			// Loop through all triangles (sets of three point indices)
			for (int triangleId = 0; triangleId < this->trianglesArray->GetNumberOfTuples(); ++triangleId)
			{
				int IDs[3];

				// Get the indices of the points that define this triangle
				this->trianglesArray->GetTupleValue(triangleId, IDs);

				// Get the point IDs for these indices, and add them to the list
				topologyList->InsertNextId(glyphPoints->GetId(IDs[0]));
				topologyList->InsertNextId(glyphPoints->GetId(IDs[1]));
				topologyList->InsertNextId(glyphPoints->GetId(IDs[2]));

				// Add the list of three points to the topology, and reset it
				newTopology->InsertNextCell(topologyList);
				topologyList->Reset();
			}
		}
		// Lines to the glyph center
		else
		{
			// Get the ID of the center point
			vtkIdType centerPointId = glyphPoints->GetId(glyphPoints->GetNumberOfIds() - 1);

			// For each angle, add a line between the center point and one of the vertices
			for (int angleId = 0; angleId < (glyphPoints->GetNumberOfIds() - 1); ++angleId)
			{
				topologyList->InsertNextId(centerPointId);
				topologyList->InsertNextId(glyphPoints->GetId(angleId));
				newTopology->InsertNextCell(topologyList);
				topologyList->Reset();
			}
		}

		glyphPoints->Delete();
		glyphPoints = NULL;

		topologyList->Delete();
		topologyList = NULL;

	} // for [All seed points]

	// Delete the coefficient array
	delete[] coeffs;

	// Set the points
	output->SetPoints(newPoints);
	newPoints->Delete();

	// If we're drawing polygons, attach the topology array to the polydata as 
	// its POLYGON array, and add the color array to the point data (one color
	// per vertex). Clear the cell data scalar array, so that the mapper will 
	// automatically use the point data scalars for coloring.

	if (this->trianglesArray && this->glyphType == GGT_Mesh)
	{
		output->SetPolys(newTopology);
		output->GetPointData()->SetScalars(colors);
		output->GetCellData()->SetScalars(NULL);
	}

	// If we're drawing lines, attach the topology array to the polydata as 
	// its LINES array, and add the color array to the cell data (one color
	// per line). Clear the point data scalar array, so that the mapper will 
	// automatically use the cell data scalars for coloring.

	else
	{
		output->SetLines(newTopology);
		output->GetCellData()->SetScalars(colors);
		output->GetPointData()->SetScalars(NULL);
	}

	// Finalize progress bar
	this->UpdateProgress(1.0);

	// Done, delete temporary object
	newTopology->Delete();
	colors->Delete();
}


//---------------------------[ computeGeometry ]---------------------------\\

bool vtkGeometryGlyphFromSHBuilder::computeGeometry(int tessOrder)
{
	// Check if the input volume has been set
	if (!(this->inputVolume))
	{
		vtkErrorMacro(<< "Input volume has not been set!");
		return false;
	}

	// Delete existing triangle array
	if (this->trianglesArray)
	{
		this->trianglesArray->Delete();
		this->trianglesArray = NULL;
	}

	// Delete the unit vector array
	if (this->unitVectors)
	{
		for (int i = 0; i < this->numberOfAngles; ++i)
		{
			delete[] (this->unitVectors[i]);
		}

		delete[] this->unitVectors;
		this->unitVectors = NULL;
	}

	// Create a sphere tessellator, using an icosahedron as the starting sphere
	Visualization::sphereTesselator<double> * tess = new Visualization::sphereTesselator<double>(Visualization::icosahedron);
	
	// Tessellate the sphere and get the result
	tess->tesselate(tessOrder);
	vtkPolyData * tessellatedSphere = vtkPolyData::New();
	tess->getvtkTesselation(true, tessellatedSphere);
	delete tess;

	// Get the tessellation points
	vtkPoints * tessellationPoints = tessellatedSphere->GetPoints();
	int numberOfTessPoints = tessellationPoints->GetNumberOfPoints();

	// Allocate new array for the unit vectors
	this->unitVectors = new double*[numberOfTessPoints];

	double p[3];

	// Loop through all tessellation points
	for (int i = 0; i < numberOfTessPoints; ++i)
	{
		this->unitVectors[i] = new double[3];

		// Get the current point
		tessellationPoints->GetPoint(i, p);

		// Normalize the point, and store it in the unit vector array
		vtkMath::Normalize(p);

		this->unitVectors[i][0] = p[0];
		this->unitVectors[i][1] = p[1];
		this->unitVectors[i][2] = p[2];
	}

	// Triangulate the tessellated sphere
	this->trianglesArray = vtkIntArray::New();
	SphereTriangulator * triangulator = new SphereTriangulator;
	triangulator->triangulateFromUnitVectors(tessellationPoints, this->trianglesArray);
	delete triangulator;

	// Delete existing angles array
	for (std::vector<double *>::iterator i = this->anglesArray.begin(); i != this->anglesArray.end(); ++i)
	{
		double * currentAngles = (*i);
		delete currentAngles;
	}

	this->anglesArray.clear();

	// Store the number of tessellation points (angles)
	this->numberOfAngles = numberOfTessPoints;

	// Convert each point to spherical coordinates, and add it to the array
	for (int i = 0; i < numberOfTessPoints; ++i)
	{
		double * sc = new double[2];
		sc[0] = acos(this->unitVectors[i][2]);
		sc[1] = atan2(this->unitVectors[i][1], this->unitVectors[i][0]);
		this->anglesArray.push_back(sc);
	}

	return true;
}


//--------------------------[ computeMinMaxRadii ]-------------------------\\

void vtkGeometryGlyphFromSHBuilder::computeMinMaxRadii(std::vector<double> * radii, double & rMin, double & rMax)
{
	// Return the range 0-1 if normalization is turned off
	if (this->nMethod == NM_None)
	{
		rMin = 0.0;
		rMax = 1.0;
		return;
	}

	double tempMin = VTK_DOUBLE_MAX;
	double tempMax = VTK_DOUBLE_MIN;

	std::vector<double>::iterator i;

	// Find the minimum and maximum radius
	for (i = radii->begin(); i != radii->end(); ++i)
	{
		if ((*i) < tempMin)		tempMin = (*i);
		if ((*i) > tempMax)		tempMax = (*i);
	}

	// If the maximum is not larger than the minimum, return the range 0-1.
	if (tempMax <= tempMin)
	{
		rMin = 0.0;
		rMax = 1.0;
		return;
	}

	// Store the maximum
	rMax = tempMax;

	// If we're using min/max normalization, store the minimum, otherwise use 0.0.
	rMin = (this->nMethod == NM_MinMax) ? tempMin : 0.0;
}


} // namespace bmia
