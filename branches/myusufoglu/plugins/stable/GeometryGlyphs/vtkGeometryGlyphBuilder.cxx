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
 * vtkGeometryGlyphBuilder.cxx
 *
 * 2011-04-11	Evert van Aart
 * - First version. 
 * 
 * 2011-04-20	Evert van Aart
 * - Added support for changing the glyph type. 
 *
 * 2011-05-04	Evert van Aart
 * - Increased support for coloring glyphs.
 * 
 * 2011-05-09	Evert van Aart
 * - Made the "Execute" and "computeGeometry" functions virtual, for use with the
 *   derived class for building SH glyphs.
 * 
 * 2011-08-05	Evert van Aart
 * - Fixed a major error in the computation of the unit vectors.
 *
 */
 

 /** Includes */

#include "vtkGeometryGlyphBuilder.h"


namespace bmia {


vtkStandardNewMacro(vtkGeometryGlyphBuilder);


//-----------------------------[ Constructor ]-----------------------------\\

vtkGeometryGlyphBuilder::vtkGeometryGlyphBuilder()
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

vtkGeometryGlyphBuilder::~vtkGeometryGlyphBuilder()
{
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

	// Unregister the input volume
	if (this->inputVolume)
		this->inputVolume->UnRegister(this);
}


//-------------------------------[ Execute ]-------------------------------\\

void vtkGeometryGlyphBuilder::Execute()
{
	// We can't do anything if the geometry has not yet been constructed
	if (this->unitVectors == NULL)
		return;

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

	// Get the "Vectors" array from the input volume
	vtkPointData * radiiPD = this->inputVolume->GetPointData();

	if (!radiiPD)
	{
		vtkErrorMacro(<< "Input volume does not contain point data!");
		return;
	}

	vtkDoubleArray * radii = vtkDoubleArray::SafeDownCast(radiiPD->GetArray("Vectors"));

	if (!radii)
	{
		vtkErrorMacro(<< "Input volume does not contain a 'Vectors' array!");
		return;
	}

	// Minimum and maximum for normalization
	double minNorm = 0.0;
	double maxNorm = 1.0;

	// Initialize progress bar
	this->UpdateProgress(0.0);
	this->SetProgressText("Initializing...");

	// Compute normalization minimum/maximum for whole image...
	if (this->nScope == NS_WholeImage && this->nMethod != NM_None)
	{
		this->computeNormalizationFactorsForWholeImage(radii, &minNorm, &maxNorm);
	}
	// ...or only for the current seed points.
	else if (this->nScope == NS_SeedPoints && this->nMethod != NM_None)
	{
		this->computeNormalizationFactorsForROI(radii, seeds, &minNorm, &maxNorm);
	}

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

		// If we're using local normalization, compute the minimum and maximum radius
		if (this->nScope == NS_Local && this->nMethod != NM_None)
			this->computeNormalizationFactorsForVoxel(radii, imagePointId, &minNorm, &maxNorm);

		// Create a new list for the glyph points
		vtkIdList * glyphPoints = vtkIdList::New();

		// Loop through all unit vectors around the point
		for (int vectorId = 0; vectorId < this->numberOfAngles; ++vectorId)
		{
			// Get the current unit vector
			double * v = this->unitVectors[vectorId];

			// Get the radius at the current voxel for this vector
			double r = radii->GetComponent(imagePointId, vectorId);

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
		} //vectors

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

bool vtkGeometryGlyphBuilder::computeGeometry(int tessOrder)
{
	// Get the image
	vtkImageData * image = this->inputVolume;

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

	this->trianglesArray = NULL;

	if (!image)
		return false;

	vtkPointData * imagePD = image->GetPointData();

	if (!imagePD)
		return false;

	// Get the array containing the glyphs radii
	vtkDoubleArray * radiiArray = vtkDoubleArray::SafeDownCast(imagePD->GetArray("Vectors"));

	if (!radiiArray)
		return false;

	// Get the array containing the angles for each glyph vertex
	vtkDoubleArray * anglesArray = vtkDoubleArray::SafeDownCast(imagePD->GetArray("Spherical Directions"));

	if (!anglesArray)
		return false;

	// Angles array should have two components. Furthermore, the number of sets of
	// angles should match the number of radii.

	if (anglesArray->GetNumberOfComponents() != 2 || anglesArray->GetNumberOfTuples() != radiiArray->GetNumberOfComponents())
		return false;

	this->numberOfAngles = anglesArray->GetNumberOfTuples();

	// Allocate new array for the unit vectors
	this->unitVectors = new double*[this->numberOfAngles];

	// Loop through all angles
	for (int i = 0; i < this->numberOfAngles; ++i)
	{
		this->unitVectors[i] = new double[3];

		// Get the two angles (azimuth and zenith)
		double * angles = anglesArray->GetTuple2(i);

		// Compute the 3D coordinates for these angles on the unit sphere
		this->unitVectors[i][0] = sinf(angles[0]) * cosf(angles[1]);
		this->unitVectors[i][1] = sinf(angles[0]) * sinf(angles[1]);
		this->unitVectors[i][2] = cosf(angles[0]);
	}

	// Try to get the triangles array from the volume. If this fails, we will use
	// lines instead of polygons.

	this->trianglesArray = vtkIntArray::SafeDownCast(imagePD->GetArray("Triangles"));

	return true;
}


//-----------------------[ setInputVolume ]-----------------------\\

void vtkGeometryGlyphBuilder::setInputVolume(vtkImageData * image)
{
	// Do nothing if the image hasn't changed
	if (this->inputVolume == image)
		return;

	// Unregister the previous image
	if (this->inputVolume)
		this->inputVolume->UnRegister((vtkObjectBase *) this);

	// Store the pointer
	this->inputVolume = image;

	// Register the new image
	image->Register((vtkObjectBase *) this);
}


//--------------[ computeNormalizationFactorsForWholeImage ]---------------\\

void vtkGeometryGlyphBuilder::computeNormalizationFactorsForWholeImage(vtkDoubleArray * radii, double * min, double * max)
{
	// Initialize minimum and maximum to maximum and minimum double values, respectively
	double tempMin = VTK_DOUBLE_MAX;
	double tempMax = VTK_DOUBLE_MIN;

	// Loop through all angles
	for (int angleId = 0; angleId < radii->GetNumberOfComponents(); ++angleId)
	{
		// Get the range for the current angle
		double range[2];
		radii->GetRange(range, angleId);

		// If one of the ranges is invalid, we use 0-1 instead
		if ((range[1] - range[0]) <= 0.0)
		{
			(*max) = 1.0;
			(*min) = 0.0;
			return;
		}

		if (range[0] < tempMin)		tempMin = range[0];
		if (range[1] > tempMax)		tempMax = range[1];
	}

	// Store the maximum
	(*max) = tempMax;

	// Store the minimum, depending on the normalization method
	     if (this->nMethod == NM_MinMax)	(*min) = tempMin;
	else if (this->nMethod == NM_Maximum)	(*min) = 0.0;

	// Final check if the range is correct
	if ((*max) - (*min) <= 0.0 || (*max) < 0.0 || (*min) < 0.0)
	{
		(*max) = 1.0;
		(*min) = 0.0;
	}
}


//---------------[ computeNormalizationFactorsForWholeRoi ]----------------\\

void vtkGeometryGlyphBuilder::computeNormalizationFactorsForROI(vtkDoubleArray * radii, vtkPointSet * seeds, double * min, double * max)
{
	double tempMin = VTK_DOUBLE_MAX;
	double tempMax = VTK_DOUBLE_MIN;

	// Loop through all seed points
	for (int pointId = 0; pointId < seeds->GetNumberOfPoints(); ++pointId)
	{
		// Get the seed point coordinates (glyph center)
		double * p = seeds->GetPoint(pointId);

		// Find the corresponding voxel
		vtkIdType imagePointId = this->inputVolume->FindPoint(p[0], p[1], p[2]);

		// Check if the seed point lies inside the image
		if (imagePointId == -1)
			continue;

		// Loop through all angles
		for (int angleId = 0; angleId < radii->GetNumberOfComponents(); ++angleId)
		{
			double r = radii->GetComponent(imagePointId, angleId);

			if (r < tempMin)	tempMin = r;
			if (r > tempMax)	tempMax = r;
		}
	}

	(*max) = tempMax;

	     if (this->nMethod == NM_MinMax)	(*min) = tempMin;
	else if (this->nMethod == NM_Maximum)	(*min) = 0.0;

	if ((*max) - (*min) <= 0.0 || (*max) < 0.0 || (*min) < 0.0)
	{
		(*max) = 1.0;
		(*min) = 0.0;
	}
}


//--------------[ computeNormalizationFactorsForWholeVoxel ]---------------\\

void vtkGeometryGlyphBuilder::computeNormalizationFactorsForVoxel(vtkDoubleArray * radii, vtkIdType ptId, double * min, double * max)
{
	double tempMin = VTK_DOUBLE_MAX;
	double tempMax = VTK_DOUBLE_MIN;

	// Loop through all angles
	for (int angleId = 0; angleId < radii->GetNumberOfComponents(); ++angleId)
	{
		double r = radii->GetComponent(ptId, angleId);

		if (r < tempMin)	tempMin = r;
		if (r > tempMax)	tempMax = r;
	}

	(*max) = tempMax;

	     if (this->nMethod == NM_MinMax)	(*min) = tempMin;
	else if (this->nMethod == NM_Maximum)	(*min) = 0.0;

	if ((*max) - (*min) <= 0.0 || (*max) < 0.0 || (*min) < 0.0)
	{
		(*max) = 1.0;
		(*min) = 0.0;
	}
}


//-----------------------------[ setGlyphType ]----------------------------\\

bool vtkGeometryGlyphBuilder::setGlyphType(GeometryGlyphType rType)
{
	// Switching to star-shaped glyphs is no problem
	if (rType == GGT_Star)
	{
		this->glyphType = GGT_Star;
		return true;
	}

	// If we want to switch to meshes, we need to make sure that the triangles array has been set
	if (rType == GGT_Mesh)
	{
		if (this->trianglesArray == NULL)
		{
			// If it hasn't, show an error message, and return false. The Geometry Glyphs
			// plugin class can use this return value to set the combo box (back) to
			// star-shaped glyphs.

			QMessageBox::warning(NULL, "Geometry Glyphs", "Cannot draw 3D mesh: Input discrete sphere function does not have triangulation data!");
			this->glyphType = GGT_Star;
			return false;
		}

		this->glyphType = GGT_Mesh;
		return true;
	}

	// This should never happen
	QMessageBox::warning(NULL, "Geometry Glyphs", "Unknown glyph type!");
	this->glyphType = GGT_Star;
	return false;
}


//---------------------------[ setScalarVolume ]---------------------------\\

void vtkGeometryGlyphBuilder::setScalarVolume(vtkImageData * rVolume)
{
	// Store the pointer
	this->scalarVolume = rVolume;
}


//-----------------------[ FillInputPortInformation ]----------------------\\

int vtkGeometryGlyphBuilder::FillInputPortInformation(int port, vtkInformation * info)
{
	// Input should be of type "vtkPointSet"
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPointSet");

	return 1;	
}


} // namespace bmia
