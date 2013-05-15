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
 * vtkCHGlyphMapper.cxx
 *
 * 2008-11-20	Tim Peeters
 * - First version
 *
 * 2011-01-07	Evert van Aart
 * - Made compatible with DTITool3, refactored some code.
 *
 * 2011-06-20	Evert van Aart
 * - Added support for LUTs.
 * - Removed support for measure range clamping and square rooting, as both things
 *   can be achieved with the Transfer Functions now.
 * - Measure values are pre-computed once now, which should smoothen camera movements
 *   when rendering SH glyphs.
 *
 * 2011-07-07	Evert van Aart
 * - After rendering the glyphs, put GL options that were disabled/enabled during
 *   rendering back to their original state. In particular, failing to re-enable
 *   blending cause problems with text rendering.
 *
 */


/** Includes */

#include "vtkCHGlyphMapper.h"


namespace bmia {


vtkStandardNewMacro(vtkCHGlyphMapper);


//-----------------------------[ Constructor ]-----------------------------\\

vtkCHGlyphMapper::vtkCHGlyphMapper()
{
	// Set default values of processing parameters
	this->MaxA0				= 0.0;
	this->MaxRadius			= 0.0;
	this->AverageA0			= 0.0;
	this->AverageRadius		= 0.0;
	this->LocalScaling		= true;
	this->FiberODF			= false;
	this->MinMaxNormalize	= false;
	this->StepSize			= 0.01;
	this->ZRotationAngle	= 0.0;
	this->YRotationAngle	= 0.0;
	this->B					= 2000.0;
	this->Eigenval[0]		= 5;
	this->Eigenval[1]		= 2;
	this->ColoringMethod	= SHCM_Direction;
	this->coloringMeasure	= HARDIMeasures::GA;

	// Set pointers to NULL
	this->UStepSize			= NULL;
	this->UNumRefineSteps	= NULL;
	this->UColoring			= NULL;
	this->URadiusThreshold	= NULL;
	this->MinMax			= NULL;

	// Read the shader program file
	this->ReadShaderProgram();

	// Setup the shader uniforms 
	this->SetupShaderUniforms();

	// Remove the shader uniforms that are used in "SHGlyphMapper" but not
	// here from the shader program to avoid warning messages
  
	this->SP->RemoveShaderUniform(this->ULightPosition);
}


//--------------------------[ ReadShaderProgram ]--------------------------\\

void vtkCHGlyphMapper::ReadShaderProgram()
{
	// Create shader program reader
	vtkMyShaderProgramReader* reader1 = vtkMyShaderProgramReader::New();

	// Set the filename of the shader file
	reader1->SetFileName("shaders/CHglyph.prog");

	// Execute the reader to read the shader file
	reader1->Execute();

	// Get the shader program and register it
	this->SP = reader1->GetOutput();
	this->SP->Register(this);

	// Done, delete the reader
	reader1->Delete();
}


//------------------------------[ Destructor ]-----------------------------\\

vtkCHGlyphMapper::~vtkCHGlyphMapper()
{
	// Delete the shader program
	if (this->SP)  
	{
		this->SP->Delete(); 
		this->SP = NULL;
	}
}


//------------------------------[ DrawPoints ]-----------------------------\\

void vtkCHGlyphMapper::DrawPoints()
{
	// If these conditions do not hold, we cannot draw the points
	if (!(this->MaxA0 > 0.0)			|| 
		!(this->MaxRadius > 0.0)		|| 
		!(this->AverageA0 > 0.0)		|| 
		!(this->AverageRadius > 0.0)	)
	{
		return;
	}

	// Get the input image data
	vtkImageData * inputImage = this->GetInput();

	// Check if the input has been set
	if (!inputImage)
	{
		vtkErrorMacro(<< "No input set!");
		return;
	}

	// Force the input to update
	inputImage->GetProducerPort()->GetProducer()->Update();

	// Get the point data of the input
	vtkPointData * inputPD = inputImage->GetPointData();

	// Check if the input has been set
	if (!inputPD)
	{
		vtkErrorMacro(<< "Input does not contain point data!");
		return;
	}

	// Get the spherical harmonics coefficients
	vtkDataArray * SHCoefficientsArray = inputPD->GetScalars();

	// Check if the input has been set
	if (!SHCoefficientsArray)
	{
		vtkErrorMacro(<< "Input does not contain an array with SH coefficients!");
		return;
	}

	// Check if we've got a LUT
	if (this->ColoringMethod == SHCM_Measure && this->lut == NULL)
	{
		vtkErrorMacro(<< "LUT has not been set!");
		return;
	}

	// Try to get the "minmax" array from the input. If it does not exist, the
	// pointer will be NULL; we check for that further down the line.

	vtkDataArray * minMaxArray = inputPD->GetArray("minmax");

	// Double-check that the Shader Program exists
	assert(this->SP);

	// Get the addresses of the variables on GPU
	GLint coeff_array1_attrib = vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients1");
	GLint coeff_array2_attrib = vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients2");
	GLint coeff_array3_attrib = vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients3");
	GLint coeff_array4_attrib = vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients4");
	GLint camera_z_attrib     = vtkgl::GetAttribLocation(this->SP->GetHandle(), "CameraZ");
	GLint minmax_attrib       = vtkgl::GetAttribLocation(this->SP->GetHandle(), "MinMaxRadius");


	// Get the seed points
	vtkPointSet * pointset = this->GetSeedPoints();

	// Check if the seed points have been set
	if (!pointset)
	{
		vtkErrorMacro(<<"No point set to render!");
		return;
	}

	// Force the seed point set to update
	pointset->Update();

	// Get the number of seed points
	vtkIdType numberOfPoints = pointset->GetNumberOfPoints();

	// If we use measure-based coloring, and the scalar array does not yet exist,
	// compute the measure values for the current seed points here.

	if (this->ColoringMethod == SHCM_Measure && this->scalarArray == NULL)
	{
		this->computeScalars(pointset, inputImage, SHCoefficientsArray);
	}

	// Some GL features will be enabled or disabled; store their current state now,
	// so that we can restore them at the end of the render pass.

	bool lightingWasEnabled		= glIsEnabled(GL_LIGHTING);
	bool blendingWasEnabled		= glIsEnabled(GL_BLEND);
	bool tex2dWasEnabled		= glIsEnabled(GL_TEXTURE_2D);
	bool cullFaceWasDisabled	= glIsEnabled(GL_CULL_FACE);

	// Further OpenGL initialization
	glDisable(GL_TEXTURE_2D);
	glPointSize(2.0);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);

	// Point coordinates
	double p[3];

	// Radius of the bounding sphere
	double r; 

	// Spherical Harmonics coefficients
	double * coefficients;

	// Point index
	vtkIdType pointId;

	// If the input does not contain a min/max array, or if we use
	// spherical deconvolution fiber ODF to sharpen the glyphs, we 
	// create a new array for the minimum and maximum.

	if ((!minMaxArray) || this->FiberODF)
	{
		this->MinMax = new double[2];

		// In this case, the minimum is always zero
		this->MinMax[0] = 0.0; 
	}

	// General index
	int j;

	// Coordinates of a covering square
	double CS[4][3]; 
	double CSr[4][2];

	// Get the up direction of the camera
	double viewUp[3];
	this->CurrentCamera->GetViewUp(viewUp);

	// Coordinate transformations
	double eX[3];
	double eY[3]; 
	double eZ[3];
	double alpha;
	double beta; 
	double gamma;
	
	// Glyphs radii
	double allRadii[4];

	// Start rendering quads
	glBegin(GL_QUADS);


	// Loop through all seed points
	for (vtkIdType i = 0; i < numberOfPoints; ++i)
	{
		// Get the current seed point
		pointset->GetPoint(i, p);

		// Find the point ID of the seed point in the input image
		pointId = inputImage->FindPoint(p);

		// Do nothing if the point was not found
		if (pointId == -1) 
		{
			continue;
		}

		// Get the SH coefficient of the point
		coefficients  = SHCoefficientsArray->GetTuple(pointId);    

		// Check if the first coefficient is positive
		if (coefficients[0] <= 0.0)
		{
			continue;
		} 

		// If this condition holds, we can get the minimum and maximum from the array
		if (minMaxArray && !this->FiberODF)
		{
			this->MinMax = minMaxArray->GetTuple(pointId);

			// Input minima and maxima are estimated by discrete polygonalization of 
			// the glyph, which is not completely accurate.

			this->MinMax[0] *= 0.95;
			this->MinMax[1] *= 1.05;

			// The check against "r" below is not strictly needed, but if the data in the 
			// "minMaxArray" is incorrect, this will prevent very bed performance.

			r = this->BoundingSphereRadius(coefficients);

			if (r < this->MinMax[1]) 
			{
				this->MinMax[1] = r;
			}

		} // if [minMaxArray && !this->FiberODF]

		// Set minimum to zero, and compute the maximum radius
		else
		{
			this->MinMax[0] = 0.0;
			this->MinMax[1] = this->BoundingSphereRadius(coefficients);
		}

		// Transpose such that the glyph center is the origin, and the eye 
		// position stays the same relatively.
      
		double eyeposT[3];

		for (j = 0; j < 3; ++j) 
		{
			eyeposT[j] = this->EyePosition[j] - p[j];
		}

		// Determine the axes of an observer coordinate system
      	for (j = 0; j < 3; ++j) 
		{
			eZ[j] = eyeposT[j];
		}

		vtkMath::Normalize(eZ);

		// We assume that "viewUp" is never parallel to "eZ"; with a normal camera, 
		// this doesn't happen, because then "p" would not be in the viewport).

		vtkMath::Cross(viewUp, eZ, eX); 
		vtkMath::Normalize(eX);
		vtkMath::Cross(eZ, eX, eY); 
		vtkMath::Normalize(eY);

		// Compute the rotation angles to go from world coordinates to observer coordinates
		gamma = atan2(eY[2], -1.0 * eX[2]);
		beta  = atan2(sqrt(eX[2] * eX[2] + eY[2] * eY[2]), eZ[2]);
		alpha = atan2(eZ[1], eZ[0]);

		double measure = 0.0;
		double rgb[3] = {1.0, 1.0, 1.0};

		// Color the glyphs using a HARDI measure
		if (this->ColoringMethod == SHCM_Measure && this->scalarArray)
		{
			// Get the measure value from the pre-computed array
			measure = this->scalarArray->GetTuple1(i);

			// Translate the scalar value to an RGB tuple
			this->lut->GetColor(measure, rgb);

			// Apply the color to the glyph
			glColor3d(rgb[0], rgb[1], rgb[2]);
		}

		// Update the SH coefficients
		this->UpdateSHCoefficients(coefficients);

		// Set the bounding radius
		r = this->MinMax[1];

		double e[3];
      
		// Rotate around Z-axis.
		for (j = 0; j < 2; ++j)
		{
			e[j] = eyeposT[j];
		}

		eyeposT[0] = e[0] * cos(alpha) + e[1] * sin(alpha);
		eyeposT[1] = e[1] * cos(alpha) - e[0] * sin(alpha);

		// Rotate around Y-axis
		for (j = 0; j < 3; ++j) 
		{
			e[j] = eyeposT[j];
		}

		eyeposT[0] = e[0] * cos(beta) - e[2] * sin(beta);
		eyeposT[2] = e[2] * cos(beta) + e[0] * sin(beta);

		// Rotate around z-axis
		for (j = 0; j < 3; ++j) 
		{
			e[j] = eyeposT[j];
		}

		eyeposT[0] = e[0] * cos(gamma) + e[1] * sin(gamma);
		eyeposT[1] = e[1] * cos(gamma) - e[0] * sin(gamma);

		// "EyeposT" should now be (0, 0, Z);
      
		// Copy the camera position to the GPU
		vtkgl::VertexAttrib1d(camera_z_attrib, eyeposT[2]);

		// Do reverse operation on SH coefficients
		this->RealWignerZRotation(coefficients, alpha);
		this->RealWignerYRotation(coefficients, beta);
		this->RealWignerZRotation(coefficients, gamma);

		// Compute radii of bounding cylinder
		double cylRhoMax = vtkCHGlyphMapper::CylinderRMax(coefficients);
		double cylZMax   = vtkCHGlyphMapper::CylinderZMax(coefficients);
    
		allRadii[0] = this->MinMax[0];
		allRadii[1] = this->MinMax[1];
		allRadii[2] = cylRhoMax; 
		allRadii[3] = cylZMax; 

		if (allRadii[2] < r)
		{
			r = allRadii[2];
		}

		glNormal3dv(p);

		// Set values on the GPU
		vtkgl::VertexAttrib4dv(coeff_array1_attrib, &(coefficients[0]));
		vtkgl::VertexAttrib4dv(coeff_array2_attrib, &(coefficients[4]));
		vtkgl::VertexAttrib4dv(coeff_array3_attrib, &(coefficients[8]));
		vtkgl::VertexAttrib3dv(coeff_array4_attrib, &(coefficients[12]));
		vtkgl::VertexAttrib4dv(minmax_attrib, allRadii);

		// Define the coordinates of a covering square
		for (j = 0; j < 3; ++j)
        {
			CS[0][j] = p[j] - r * eX[j] - r * eY[j] + cylZMax * eZ[j];
			CS[1][j] = p[j] + r * eX[j] - r * eY[j] + cylZMax * eZ[j];
			CS[2][j] = p[j] + r * eX[j] + r * eY[j] + cylZMax * eZ[j];
			CS[3][j] = p[j] - r * eX[j] + r * eY[j] + cylZMax * eZ[j];
        }

		// Relative quad coords in observer coordinate system. "z = cylZMax" for all
		CSr[0][0] = -1.0 * r;	CSr[0][1] = -1.0 * r;
		CSr[1][0] =  1.0 * r;	CSr[1][1] = -1.0 * r;
		CSr[2][0] =  1.0 * r;	CSr[2][1] =  1.0 * r;
		CSr[3][0] = -1.0 * r;	CSr[3][1] =  1.0 * r;

		// Render the covering square
		for (j = 0; j < 4; ++j)
		{
			glColor4d(CSr[j][0], CSr[j][1], cylZMax, eyeposT[2]);
			glTexCoord3f(rgb[0], rgb[1], rgb[2]);
			glVertex3dv(CS[j]);
		}

	} // for [numberOfPoints]

	// Done rendering
	glEnd();

	// Delete the min/max array
	if ((!minMaxArray) || this->FiberODF)
	{
		delete[] this->MinMax;
	}

	this->MinMax = NULL;

	// Set options that were disabled/enabled back to how they were
	if (lightingWasEnabled)		glEnable(GL_LIGHTING);
	if (blendingWasEnabled)		glEnable(GL_BLEND);
	if (tex2dWasEnabled)		glEnable(GL_TEXTURE_2D);
	if (cullFaceWasDisabled)	glDisable(GL_CULL_FACE);
}


//----------------------------[ CylinderRhoMax ]---------------------------\\

float vtkCHGlyphMapper::CylinderRhoMax(int l, int m)
{
	// Output value
	float result = 0.0;
  
	// Hard-coded results for all combinations of "l" and "m"
	     if ((l == 0) && (m ==  0))		result = 0.282095;
	else if ((l == 2) && (m == -2))		result = 0.386274;
	else if ((l == 2) && (m == -1))		result = 0.297354;
	else if ((l == 2) && (m ==  0))		result = 0.315392;
	else if ((l == 2) && (m ==  1))		result = 0.297354;
	else if ((l == 2) && (m ==  2))		result = 0.386274;
	else if ((l == 4) && (m == -4))		result = 0.442533;
	else if ((l == 4) && (m == -3))		result = 0.358249;
	else if ((l == 4) && (m == -2))		result = 0.334523;
	else if ((l == 4) && (m == -1))		result = 0.311653;
	else if ((l == 4) && (m ==  0))		result = 0.317357;
	else if ((l == 4) && (m ==  1))		result = 0.311653;
	else if ((l == 4) && (m ==  2))		result = 0.334523;
	else if ((l == 4) && (m ==  3))		result = 0.358249;
	else if ((l == 4) && (m ==  4))		result = 0.442533;

	return result;
}


//-----------------------------[ CylinderZMax ]----------------------------\\

float vtkCHGlyphMapper::CylinderZMax(int l, int m)
{
	// Output value
	float result = 0.0;

	// Hard-coded results for all combinations of "l" and "m"
	     if ((l == 0) && (m ==  0))		result = 0.282095;
	else if ((l == 2) && (m == -2))		result = 0.148677;
	else if ((l == 2) && (m == -1))		result = 0.297354;
	else if ((l == 2) && (m ==  0))		result = 0.630783;
	else if ((l == 2) && (m ==  1))		result = 0.297354;
	else if ((l == 2) && (m ==  2))		result = 0.148677;
	else if ((l == 4) && (m == -4))		result = 0.126660;
	else if ((l == 4) && (m == -3))		result = 0.232690;
	else if ((l == 4) && (m == -2))		result = 0.335275;
	else if ((l == 4) && (m == -1))		result = 0.459798;
	else if ((l == 4) && (m ==  0))		result = 0.846284;
	else if ((l == 4) && (m ==  1))		result = 0.459798;
	else if ((l == 4) && (m ==  2))		result = 0.335275;
	else if ((l == 4) && (m ==  3))		result = 0.232690;
	else if ((l == 4) && (m ==  4))		result = 0.126660;
  
	return result;
}


//-----------------------------[ CylinderRMax ]----------------------------\\

float vtkCHGlyphMapper::CylinderRMax(double * coefficients)
{
	assert(coefficients != NULL);
  
	// Output radius
	double sumL = 0.0;

	// Sum for one order
	double sumM = 0.0;

	// Coefficient index
	int j = 0;

	// Loop through the orders
	for (int l = 0; l <= 4; l += 2)
	{
		// Reset the sum for this order
		sumM = 0.0;

		// Process (2*L+1) coefficients
		for (int m = -l; m <= l; m++)
		{
			sumM += fabs(coefficients[j]) * vtkCHGlyphMapper::CylinderRhoMax(l, m);
			j++;
		}

		sumL += sumM;
	}
  
	return sumL; 
}


//-----------------------------[ CylinderZMax ]----------------------------\\

float vtkCHGlyphMapper::CylinderZMax(double * coefficients)
{
	assert(coefficients != NULL);

	// Output radius
	double sumL = 0.0;

	// Sum for one order
	double sumM = 0.0;

	// Coefficient index
	int j = 0;

	// Loop through the orders
	for (int l = 0; l <= 4; l += 2)
	{
		// Reset the sum for this order
		sumM = 0.0;

		// Process (2*L+1) coefficients
		for (int m = -l; m <= l; m++)
		{
			sumM += fabs(coefficients[j]) * vtkCHGlyphMapper::CylinderZMax(l, m);
			j++;
		}

		sumL += sumM;
	}

	return sumL; 
}


} // namespace bmia
