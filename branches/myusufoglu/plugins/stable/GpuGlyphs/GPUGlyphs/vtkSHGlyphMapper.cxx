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
 * vtkSHGlyphMapper.cxx
 *
 * 2008-09-16	Tim Peeters
 * - First version
 *
 * 2010-12-01	Evert van Aart
 * - Made compatible with DTITool3, refactored some code.
 *
 * 2010-01-10	Evert van Aart
 * - Added the "initODF" function.
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

#include "vtkSHGlyphMapper.h"


namespace bmia {


vtkStandardNewMacro(vtkSHGlyphMapper);


//-----------------------------[ Constructor ]-----------------------------\\

vtkSHGlyphMapper::vtkSHGlyphMapper()
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
	this->scalarArray		= NULL;

	// Read the shader program file
	this->ReadShaderProgram();

	// Setup the shader uniforms 
	this->SetupShaderUniforms();
}


//--------------------------[ ReadShaderProgram ]--------------------------\\

void vtkSHGlyphMapper::ReadShaderProgram()
{
	// Create shader program reader
	vtkMyShaderProgramReader * reader1 = vtkMyShaderProgramReader::New();

	// Set the filename of the shader file
	reader1->SetFileName("shaders/SHglyph.prog");

	// Execute the reader to read the shader file
	reader1->Execute();

	// Get the shader program and register it
	this->SP = reader1->GetOutput();
	this->SP->Register(this);

	// Done, delete the reader
	reader1->Delete();
}


//-------------------------[ SetupShaderUniforms ]-------------------------\\

void vtkSHGlyphMapper::SetupShaderUniforms()
{
	// First, call the setup function of the parent class
	this->vtkGlyphMapperVA::SetupShaderUniforms();

	// Delete existing variables
	if (this->UStepSize)			this->UStepSize->Delete();
	if (this->UNumRefineSteps)		this->UNumRefineSteps->Delete();
	if (this->UColoring)			this->UColoring->Delete();
	if (this->URadiusThreshold)		this->URadiusThreshold->Delete();

	// Create new variable for the step size
	this->UStepSize = vtkUniformFloat::New();
	this->UStepSize->SetName("StepSize");
	this->UStepSize->SetValue(this->StepSize);

	// Create new variable for the number of refine steps
	this->UNumRefineSteps = vtkUniformInt::New();
	this->UNumRefineSteps->SetName("NumRefineSteps");
	this->UNumRefineSteps->SetValue(0);

	// Create new variable for the coloring options
	this->UColoring = vtkUniformInt::New();
	this->UColoring->SetName("Coloring");
	this->UColoring->SetValue(this->ColoringMethod);

	// Create new variable for the radius threshold
	this->URadiusThreshold = vtkUniformFloat::New();
	this->URadiusThreshold->SetName("RadiusThreshold");
	this->URadiusThreshold->SetValue(0.5);

	// Add variables to the shader program
	if (this->SP) 
	{
		this->SP->AddShaderUniform(this->UStepSize);
		this->SP->AddShaderUniform(this->UNumRefineSteps);
		this->SP->AddShaderUniform(this->UColoring);
		this->SP->AddShaderUniform(this->URadiusThreshold);
	}
}


//------------------------------[ Destructor ]-----------------------------\\

vtkSHGlyphMapper::~vtkSHGlyphMapper()
{
	// Delete existing variables
	if (this->UStepSize)			this->UStepSize->Delete();
	if (this->UNumRefineSteps)		this->UNumRefineSteps->Delete();
	if (this->UColoring)			this->UColoring->Delete();
	if (this->URadiusThreshold)		this->URadiusThreshold->Delete();

	// Delete the shader program
	if (this->SP)  
	{
		this->SP->Delete(); 
		this->SP = NULL;
	}

	// Delete the scalar array
	if (this->scalarArray)
	{
		this->scalarArray->Delete();
		this->scalarArray = NULL;
	}
}


//------------------------------[ DrawPoints ]-----------------------------\\

void vtkSHGlyphMapper::DrawPoints()
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
	GLint saturation_attrib		= vtkgl::GetAttribLocation(this->SP->GetHandle(), "Saturation");
	GLint pos_attrib			= vtkgl::GetAttribLocation(this->SP->GetHandle(), "GlyphPosition");
	GLint coeff_array1_attrib	= vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients1");
	GLint coeff_array2_attrib	= vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients2");
	GLint coeff_array3_attrib	= vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients3");
	GLint coeff_array4_attrib	= vtkgl::GetAttribLocation(this->SP->GetHandle(), "SHCoefficients4");
	GLint minmax_attrib			= vtkgl::GetAttribLocation(this->SP->GetHandle(), "MinMaxRadius");

	// so that we can restore them at the end of the render pass.

	bool lightingWasEnabled		= glIsEnabled(GL_LIGHTING);
	bool blendingWasEnabled		= glIsEnabled(GL_BLEND);
	bool tex2dWasEnabled		= glIsEnabled(GL_TEXTURE_2D);
	bool cullFaceWasDisabled	= glIsEnabled(GL_CULL_FACE);

	// Further OpenGL initialization
	glDisable(GL_TEXTURE_2D);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);
	glPointSize(2.0);

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

	// Point coordinates
	double p[3];

	// Bounding Box matrix
	double BB[8][3];

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

		// Transform the point
		this->transformPoint(p, true);

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

		// Update the SH coefficients
		this->UpdateSHCoefficients(coefficients);

		// Set the bounding radius
		r = this->MinMax[1];

		// Set the vertex attributes
		vtkgl::VertexAttrib3dv(pos_attrib, p);
		vtkgl::VertexAttrib4dv(coeff_array1_attrib, &(coefficients[0]));
		vtkgl::VertexAttrib4dv(coeff_array2_attrib, &(coefficients[4]));
		vtkgl::VertexAttrib4dv(coeff_array3_attrib, &(coefficients[8]));
		vtkgl::VertexAttrib3dv(coeff_array4_attrib, &(coefficients[12]));
		vtkgl::VertexAttrib2dv(minmax_attrib, this->MinMax); 

		// Color the glyphs using a HARDI measure
		if (this->ColoringMethod == SHCM_Measure && this->scalarArray)
		{
			// Get the measure value from the pre-computed array
			double measure = this->scalarArray->GetTuple1(i);

			double rgb[3];

			// Translate the scalar value to an RGB tuple
			this->lut->GetColor(measure, rgb);

			// Apply the color to the glyph
			glColor3d(rgb[0], rgb[1], rgb[2]);
		}

		// Define the coordinates of the bounding box around the glyph
		BB[0][0] = p[0] - r;	BB[0][1] = p[1] - r;	BB[0][2] = p[2] - r;
		BB[1][0] = p[0] + r;	BB[1][1] = p[1] - r;	BB[1][2] = p[2] - r;
		BB[2][0] = p[0] + r;	BB[2][1] = p[1] + r;	BB[2][2] = p[2] - r;
		BB[3][0] = p[0] - r;	BB[3][1] = p[1] + r;	BB[3][2] = p[2] - r;
		BB[4][0] = p[0] - r;	BB[4][1] = p[1] - r;	BB[4][2] = p[2] + r;
		BB[5][0] = p[0] + r;	BB[5][1] = p[1] - r;	BB[5][2] = p[2] + r;
		BB[6][0] = p[0] + r;	BB[6][1] = p[1] + r;	BB[6][2] = p[2] + r;
		BB[7][0] = p[0] - r;	BB[7][1] = p[1] + r;	BB[7][2] = p[2] + r;

		// Set point coordinates
		glTexCoord3dv(p);
      
		// Render the bounding box
		glVertex3dv(BB[0]); glVertex3dv(BB[3]); glVertex3dv(BB[2]); glVertex3dv(BB[1]);		// Front
		glVertex3dv(BB[0]); glVertex3dv(BB[4]); glVertex3dv(BB[7]); glVertex3dv(BB[3]);		// Left
		glVertex3dv(BB[0]); glVertex3dv(BB[1]); glVertex3dv(BB[5]); glVertex3dv(BB[4]);		// Bottom
		glVertex3dv(BB[1]); glVertex3dv(BB[2]); glVertex3dv(BB[6]); glVertex3dv(BB[5]);		// Right
		glVertex3dv(BB[3]); glVertex3dv(BB[7]); glVertex3dv(BB[6]); glVertex3dv(BB[2]);		// Top
		glVertex3dv(BB[5]); glVertex3dv(BB[6]); glVertex3dv(BB[7]); glVertex3dv(BB[4]);		// Back

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


//-------------------------------[ SetInput ]------------------------------\\

void vtkSHGlyphMapper::SetInput(vtkImageData * input)
{
	// Store the input image
	vtkGlyphMapperVA::SetInput(input);
	this->InputUpdated();
}


//-------------------------------[ SetInput ]------------------------------\\

void vtkSHGlyphMapper::SetInput(vtkDataSet * input)
{
	// Store the input data set
	vtkGlyphMapperVA::SetInput(input);
	this->InputUpdated();
}


//--------------------------[ SetInputConnection ]-------------------------\\

void vtkSHGlyphMapper::SetInputConnection(vtkAlgorithmOutput * input)
{
	// Store the input connection
	vtkGlyphMapperVA::SetInputConnection(input);
	this->InputUpdated();
}


//-----------------------------[ InputUpdated ]----------------------------\\

void vtkSHGlyphMapper::InputUpdated()
{
	// Get the input
	vtkImageData * inputImage = this->GetInput();

	// Check if the input has been set
	if (!inputImage)
	{
		vtkErrorMacro(<< "No input set!");
		return;
	}

	// Force the input to update itself
	inputImage->Update();

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

	// Get the number of points in the input image
	vtkIdType numberOfPoints = SHCoefficientsArray->GetNumberOfTuples();

	// Used for computing the maximum and average A0 and radius
	double maxA0	= 0.0;
	double maxR		= 0.0;
	double sumA0	= 0.0;
	double sumR		= 0.0;

	// Radius of bounding sphere
	double r;

	// SH Coefficients
	double * coefficients;

	// Loop through all points in the input image
	for (vtkIdType i = 0; i < numberOfPoints; ++i)
	{
		// Get the SH coefficients
		coefficients  = SHCoefficientsArray->GetTuple(i);    

		// Check if the coefficients exist
		if (!coefficients)
		{
			return;
		}

		// Do nothing if the first coefficient is not positive
		if (coefficients[0] <= 0.0)
		{
			continue;
		}

		// Keep track of the maximum first coefficient
		if (coefficients[0] > maxA0)
		{
			maxA0 = coefficients[0];
		}
		
		// Compute the radius of the bounding sphere
		r = this->BoundingSphereRadius(coefficients);

		// Keep track of the maximum radius
		if (r > maxR) 
		{
			maxR = r;
		}

		// Keep track of the sum of radius and first coefficient
		sumA0 += coefficients[0];
		sumR  += r;

    } // for [numberOfPoints]

	// Store maxima and averages
	this->MaxA0				= maxA0;
	this->MaxRadius			= maxR;
	this->AverageA0			= sumA0 / (double) numberOfPoints;
	this->AverageRadius		= sumR /  (double) numberOfPoints;

	// Delete the scalar array
	if (this->scalarArray)
	{
		this->scalarArray->Delete();
		this->scalarArray = NULL;
	}
}


//----------------------------[ computeScalars ]---------------------------\\

void vtkSHGlyphMapper::computeScalars(vtkPointSet * seeds, vtkImageData * image, vtkDataArray * coeffArray)
{
	// Create and setup the scalar array
	this->scalarArray = vtkDoubleArray::New();
	this->scalarArray->SetNumberOfComponents(1);
	this->scalarArray->SetNumberOfTuples(seeds->GetNumberOfPoints());

	// Get the number of seed points
	vtkIdType numberOfPoints = seeds->GetNumberOfPoints();

	// Point coordinates
	double p[3];

	// Spherical Harmonics coefficients
	double * coefficients;

	// Point index
	vtkIdType pointId;

	// Create an object for computing HARDI measures
	HARDIMeasures * HMeasures = new HARDIMeasures;

	// Create a progress dialog for tracking progress (for computationally expensive measures)
	QProgressDialog progress("(Re)computing HARDI measure values for current seed points...", QString(), 0, numberOfPoints, NULL);
	progress.setMinimumDuration(1000);
	progress.setWindowTitle("GPU Glyphs");
	progress.setValue(0);

	// Compute a step size for updating the progress dialog
	int progressStepSize = numberOfPoints / 100;
	progressStepSize += (progressStepSize == 0) ? (1) : (0);

	// Loop through all seed points
	for (vtkIdType i = 0; i < numberOfPoints; ++i)
	{
		// Get the current seed point
		seeds->GetPoint(i, p);

		// Find the point ID of the seed point in the input image
		pointId = image->FindPoint(p);

		// Do nothing if the point was not found
		if (pointId == -1) 
		{
			this->scalarArray->SetTuple1(i, 0.0);
			continue;
		}

		// Get the SH coefficient of the point
		coefficients  = coeffArray->GetTuple(pointId);    

		// Check if the first coefficient is positive
		if (coefficients[0] <= 0.0)
		{
			this->scalarArray->SetTuple1(i, 0.0);
			continue;
		} 

		// Get the selected HARDI measure for the current voxel
		double measure = HMeasures->HARDIMeasure((int) this->coloringMeasure, coefficients, 4);

		// Add the measure value to the array
		this->scalarArray->SetTuple1(i, measure);

		// Update the progress bar
		if (i % progressStepSize == 0)
		{
			progress.setValue(i);
		}

	} // for [numberOfPoints]

	progress.setValue(numberOfPoints);

	// Delete the measures object
	delete HMeasures;
}


//-------------------------[ BoundingSphereRadius ]------------------------\\

double vtkSHGlyphMapper::BoundingSphereRadius(double * coefficients)
{
	// Output radius
	double sumL = 0.0;

	// Sum for one order
	double sumM = 0.0;

	// Coefficient index
	int	j = 0;
	
	// Loop through the orders
	for (int l = 0; l <= 4; l += 2)
	{
		// Reset the sum for this order
		sumM = 0.0;
    
		// Process (2*L+1) coefficients
		for (int m = -l; m <= l; m++)
		{
			// Compute sum of squared coefficients
			sumM += coefficients[j] * coefficients[j];
			j++;
		}
    
		// Update output radius
		sumM *= (2.0 * (double) l + 1.0) / (4.0 * vtkMath::Pi());
		sumL += sumM;
    }

	// Compute output
	sumL = sqrtf(3) * sqrtf(sumL);

	return sumL; 
}


//-----------------------------[ SetStepSize ]-----------------------------\\

void vtkSHGlyphMapper::SetStepSize(float step)
{
	// Store the step size if it positive
	if (step > 0.0)
	{
		this->StepSize = step;

		if (this->UStepSize)
		{
			this->UStepSize->SetValue(this->StepSize);
		}
	}
}


//--------------------------[ SetNumRefineSteps ]--------------------------\\

void vtkSHGlyphMapper::SetNumRefineSteps(int num)
{
	// Store the number of refinement steps
	if (this->UNumRefineSteps)
	{
		this->UNumRefineSteps->SetValue(num);
	}
}


//-------------------------[ RealWignerZRotation ]-------------------------\\

void vtkSHGlyphMapper::RealWignerZRotation(double * coefficients, double angle)
{
	// Do nothing if the angle is zero
	if (angle == 0.0) 
		return;
  
	// Compute new coefficient values
	this->RealWignerZRotation0(&coefficients[0], angle);
	this->RealWignerZRotation2(&coefficients[1], angle);
	this->RealWignerZRotation4(&coefficients[6], angle);
}


//-------------------------[ RealWignerZRotation0 ]------------------------\\

void vtkSHGlyphMapper::RealWignerZRotation0(double * coefficients, double angle)
{
	// Nothing to do for 0th order SH.
	return;
}


//-------------------------[ RealWignerZRotation2 ]------------------------\\

void vtkSHGlyphMapper::RealWignerZRotation2(double * coefficients, double angle)
{
	// Local copy of second-order coefficients
	double a[5];
  
	// Copy coefficients
	for (int i = 0; i < 5; i++) 
	{
		a[i] = coefficients[i];
	}

	// Pre-compute sines and cosines
	double c1 = cos(angle);
	double c2 = cos(2.0 * angle);
	double s1 = sin(angle);
	double s2 = sin(2.0 * angle);

	// Compute output coefficients
	coefficients[0] = a[0] * c2 + a[4] * s2;
	coefficients[1] = a[1] * c1 - a[3] * s1;
	coefficients[2] = a[2];
	coefficients[3] = a[3] * c1 + a[1] * s1;
	coefficients[4] = a[4] * c2 - a[0] * s2;
}


//-------------------------[ RealWignerZRotation4 ]------------------------\\

void vtkSHGlyphMapper::RealWignerZRotation4(double * coefficients, double angle)
{
	// Local copy of fourth-order coefficients
	double a[9];

	// Copy coefficients
	for (int i = 0; i < 9; i++) 
	{
		a[i] = coefficients[i];
	}

	// Pre-compute sines and cosines
	double c1 = cos(angle);
	double c2 = cos(2.0 * angle);
	double c3 = cos(3.0 * angle);
	double c4 = cos(4.0 * angle);
	double s1 = sin(angle);
	double s2 = sin(2.0 * angle);
	double s3 = sin(3.0 * angle);
	double s4 = sin(4.0 * angle);

	// Compute output coefficients
	coefficients[0] = a[0] * c4 + a[8] * s4;
	coefficients[1] = a[1] * c3 - a[7] * s3;
	coefficients[2] = a[2] * c2 + a[6] * s2;
	coefficients[3] = a[3] * c1 - a[5] * s1;
	coefficients[4] = a[4];
	coefficients[5] = a[5] * c1 + a[3] * s1;
	coefficients[6] = a[6] * c2 - a[2] * s2;
	coefficients[7] = a[7] * c3 + a[1] * s3;
	coefficients[8] = a[8] * c4 - a[0] * s4;
}


//-------------------------[ RealWignerYRotation ]-------------------------\\

void vtkSHGlyphMapper::RealWignerYRotation(double * coefficients, double angle)
{
	// Do nothing if the angle is zero
	if (angle == 0.0) 
		return;

	// Compute new coefficient values
	this->RealWignerYRotation0(&coefficients[0], angle);
	this->RealWignerYRotation2(&coefficients[1], angle);
	this->RealWignerYRotation4(&coefficients[6], angle);
}


//-------------------------[ RealWignerYRotation0 ]------------------------\\

void vtkSHGlyphMapper::RealWignerYRotation0(double * coefficients, double angle)
{
	// Nothing to do for 0th order SH.
	return;
}


//-------------------------[ RealWignerYRotation2 ]------------------------\\

void vtkSHGlyphMapper::RealWignerYRotation2(double * coefficients, double angle)
{
	// Local copy of second-order coefficients
	double a[5];

	// Copy coefficients
	for (int i = 0; i < 5; i++) 
	{
		a[i] = coefficients[i];
	}

	// Pre-compute sines and cosines
	double c1 = cos(angle);
	double c2 = cos(2.0 * angle);
	double s1 = sin(angle);
	double s2 = sin(2.0 * angle);

	// Compute output coefficients
	coefficients[0] = (3 * a[0] + sqrtf(3) * a[2] + a[0] * c2 - sqrtf(3) * a[2] * c2 - 2 * a[1] * s2) / 4.0;
	coefficients[1] = (2 * a[1] * c2 + a[0] * s2 - sqrtf(3) * a[2] * s2) / 2.0;
	coefficients[2] = (sqrtf(3) * a[0] + a[2] - sqrtf(3) * a[0] * c2 + 3 * a[2] * c2 + 2 * sqrtf(3) * a[1] * s2) / 4.0;
	coefficients[3] = a[3] * c1 - a[4] * s1;
	coefficients[4] = a[4] * c1 + a[3] * s1;
}


//-------------------------[ RealWignerYRotation4 ]------------------------\\

void vtkSHGlyphMapper::RealWignerYRotation4(double * coefficients, double angle)
{
	// Local copy of fourth-order coefficients
	double a[9];

	// Copy coefficients
	for (int i = 0; i < 9; i++) 
	{
		a[i] = coefficients[i];
	}

	// Pre-compute sines and cosines
	double c1 = cos(angle);
	double c2 = cos(2.0 * angle);
	double c3 = cos(3.0 * angle);
	double c4 = cos(4.0 * angle);
	double s1 = sin(angle);
	double s2 = sin(2.0 * angle);
	double s3 = sin(3.0 * angle);
	double s4 = sin(4.0 * angle);

	// Compute output coefficients
	coefficients[0] = (35*a[0] + 10*sqrtf(7)*a[2] + 3*sqrtf(35)*a[4] + 28*a[0]*c2 - 8*sqrtf(7)*a[2]*c2 - 4*sqrtf(35)*a[4]*c2 + a[0]*c4 - 2*sqrtf(7)*a[2]*c4 + sqrtf(35)*a[4]*c4 - 28*sqrtf(2)*a[1]*s2 - 4*sqrtf(14)*a[3]*s2 - 2*sqrtf(2)*a[1]*s4 + 2*sqrtf(14)*a[3]*s4)/64.0;
	coefficients[1] = (28*a[1]*c2 + 4*sqrtf(7)*a[3]*c2 + 4*a[1]*c4 - 4*sqrtf(7)*a[3]*c4 + 14*sqrtf(2)*a[0]*s2 - 4*sqrtf(14)*a[2]*s2 - 2*sqrtf(70)*a[4]*s2 + sqrtf(2)*a[0]*s4 - 2*sqrtf(14)*a[2]*s4 + sqrtf(70)*a[4]*s4)/32.0;
	coefficients[2] = (5*sqrtf(7)*a[0] + 10*a[2] + 3*sqrtf(5)*a[4] - 4*sqrtf(7)*a[0]*c2 + 8*a[2]*c2 + 4*sqrtf(5)*a[4]*c2 - sqrtf(7)*a[0]*c4 + 14*a[2]*c4 - 7*sqrtf(5)*a[4]*c4 + 4*sqrtf(14)*a[1]*s2 + 4*sqrtf(2)*a[3]*s2 + 2*sqrtf(14)*a[1]*s4 - 14*sqrtf(2)*a[3]*s4)/32.0;
	coefficients[3] = (4*sqrtf(7)*a[1]*c2 + 4*a[3]*c2 - 4*sqrtf(7)*a[1]*c4 + 28*a[3]*c4 + 2*sqrtf(14)*a[0]*s2 - 4*sqrtf(2)*a[2]*s2 - 2*sqrtf(10)*a[4]*s2 - sqrtf(14)*a[0]*s4 + 14*sqrtf(2)*a[2]*s4 - 7*sqrtf(10)*a[4]*s4)/32.0;
	coefficients[4] = (3*sqrtf(35)*a[0] + 6*sqrtf(5)*a[2] + 9*a[4] - 4*sqrtf(35)*a[0]*c2 + 8*sqrtf(5)*a[2]*c2 + 20*a[4]*c2 + sqrtf(35)*a[0]*c4 - 14*sqrtf(5)*a[2]*c4 + 35*a[4]*c4 + 4*sqrtf(70)*a[1]*s2 + 4*sqrtf(10)*a[3]*s2 - 2*sqrtf(70)*a[1]*s4 + 14*sqrtf(10)*a[3]*s4)/64.0;
	coefficients[5] = (9*a[5]*c1 + 3*sqrtf(7)*a[7]*c1 + 7*a[5]*c3 - 3*sqrtf(7)*a[7]*c3 - 3*sqrtf(2)*a[6]*s1 - 3*sqrtf(14)*a[8]*s1 - 7*sqrtf(2)*a[6]*s3 + sqrtf(14)*a[8]*s3)/16.0;
	coefficients[6] = (2*a[6]*c1 + 2*sqrtf(7)*a[8]*c1 + 14*a[6]*c3 - 2*sqrtf(7)*a[8]*c3 + 3*sqrtf(2)*a[5]*s1 + sqrtf(14)*a[7]*s1 + 7*sqrtf(2)*a[5]*s3 - 3*sqrtf(14)*a[7]*s3)/16.0;
	coefficients[7] = (3*sqrtf(7)*a[5]*c1 + 7*a[7]*c1 - 3*sqrtf(7)*a[5]*c3 + 9*a[7]*c3 - sqrtf(14)*a[6]*s1 - 7*sqrtf(2)*a[8]*s1 + 3*sqrtf(14)*a[6]*s3 - 3*sqrtf(2)*a[8]*s3)/16.0;
	coefficients[8] = (2*sqrtf(7)*a[6]*c1 + 14*a[8]*c1 - 2*sqrtf(7)*a[6]*c3 + 2*a[8]*c3 + 3*sqrtf(14)*a[5]*s1 + 7*sqrtf(2)*a[7]*s1 - sqrtf(14)*a[5]*s3 + 3*sqrtf(2)*a[7]*s3)/16.0;

	return;
}


//-------------------------[ UpdateSHCoefficients ]------------------------\\

void vtkSHGlyphMapper::UpdateSHCoefficients(double * coefficients)
{
	// Check if the min/max array exists
	bool mm = (this->MinMax != NULL);

	// Rotate the coefficients if necessary
	this->RealWignerZRotation(coefficients, this->ZRotationAngle);
	this->RealWignerYRotation(coefficients, this->YRotationAngle);

	// Use spherical deconvolution fiber ODF to sharpen the glyphs
	if (this->FiberODF)
	{
		this->DoFiberODF(coefficients);

		// Update the maximum radius
		if (mm) 
		{
			this->MinMax[1] = this->BoundingSphereRadius(coefficients);
		}
    }
	// Normalize the minimum and maximum if needed
	else if (mm && this->MinMaxNormalize)
	{
		this->DoMinMax(coefficients);
		this->MinMax[0] = 0.0;
		this->MinMax[1] = 1.0;
	}

	// Scale the glyphs
	this->ScaleGlyphs(coefficients);
}


//-----------------------------[ ScaleGlyphs ]-----------------------------\\

void vtkSHGlyphMapper::ScaleGlyphs(double * coefficients)
{
	// Check if the min/max array exists
	bool mm = (this->MinMax != NULL);

	// Get the required scale
	double scale = this->GetGlyphScaling();
  
	// If "LocalScaling" is on, normalize using the maximum radius
	if (mm && this->LocalScaling) 
	{
		scale /= (2.0 * this->MinMax[1]);
	}
	// Otherwise, use the global maximum of the first SH coefficients
	else 
	{
		scale /= this->MaxA0;
	}

	// Scale all coefficients
	for (int k = 0; k < 15; k++) 
	{
		coefficients[k] *= scale;
	}

	// Scale the minimum/maximum radius
	if (mm) 
	{
		for (int k = 0; k < 2; k++) 
		{
			this->MinMax[k] *= scale;
		}
	}
}


//-----------------------------[ SetColoring ]-----------------------------\\

void vtkSHGlyphMapper::SetColoring(int col)
{
	// Store the coloring type
	this->ColoringMethod = (SHColoringMethod) col;

	if (this->UColoring)
	{
		this->UColoring->SetValue(col);
	}
}


//--------------------------[ SetRadiusThreshold ]-------------------------\\

void vtkSHGlyphMapper::SetRadiusThreshold(float threshold)
{
	// Store the threshold of the radius
	if (this->URadiusThreshold)
	{
		this->URadiusThreshold->SetValue(threshold);
	}
}


//------------------------------[ DoFiberODF ]-----------------------------\\

void vtkSHGlyphMapper::DoFiberODF(double * coefficients)
{
	HARDITransformationManager::FiberODF(4, this->B, this->Eigenval, coefficients, coefficients);
}


//-------------------------------[ DoMinMax ]------------------------------\\

void vtkSHGlyphMapper::DoMinMax(double * coefficients)
{
	// The maximum should of course be higher than the minimum
	assert(this->MinMax[1] > this->MinMax[0]);

	// Scale first coefficients
	coefficients[0] = (coefficients[0] - 2.0 * sqrt(vtkMath::DoublePi()) * MinMax[0]) / (MinMax[1] - MinMax[0]);

	// Scale the other coefficients
	for (int i = 1; i < 15; i++) 
	{
		coefficients[i] /= (MinMax[1] - MinMax[0]);
	}
}


//-------------------------------[ initODF ]-------------------------------\\

void vtkSHGlyphMapper::initODF(vtkImageData *eigen)
{
	// Get the seed points
	vtkPointSet * points = this->GetSeedPoints();

	// Check if the seed points have been set
	if (!points)
	{
		vtkErrorMacro(<< "Called initODF before setting the seed points!");
		return;
	}

	// Get number of seed points
	vtkIdType numberOfPoints = points->GetNumberOfPoints();

	// Check if the number of seed points is positive
	if (numberOfPoints <= 0)
	{
		vtkErrorMacro(<< "Number of seed points is not positive!");
		return;
	}

	// Check if the eigensystem image has been set
	if (!eigen)
	{
		vtkErrorMacro(<< "Eigensystem image not set!");
		return;
	}

	// Get the point data of the eigensystem image
	vtkPointData * eigenPD = eigen->GetPointData();

	// Check if the point data exists
	if (!eigenPD)
	{
		vtkErrorMacro(<< "Eigensystem image does not contain point data!");
		return;
	}

	// Get the first two eigenvalue arrays
	vtkDataArray * EVal1Array = eigenPD->GetArray("Eigenvalue 1");
	vtkDataArray * EVal2Array = eigenPD->GetArray("Eigenvalue 2");

	// Check if the arrays exist
	if (!EVal1Array || !EVal2Array)
	{
		vtkErrorMacro(<< "Eigenvalue array(s) missing from eigensystem image!");
		return;
	}

	// Temporary point coordinates
	double p[3];

	// ID of current point
	vtkIdType pointId;

	// Number of valid eigenvalues (negative values are not valid)
	int numberOfValidEVals = 0; 
	
	// First and second eigenvalues of current seed point
	double eval1; 
	double eval2;

	// Sum of first and second eigenvalues
	double eval1Sum = 0.0; 
	double eval2Sum = 0.0;

	// Loop through all seed points
	for (vtkIdType i = 0; i < numberOfPoints; i++)
	{
		// Get the coordinates of the seed point
		points->GetPoint(i, p);

		// Find the seed point in the eigensystem image
		pointId = eigen->FindPoint(p);

		// Do nothing if the point was not found
		if (pointId != -1)
		{
			// Get the first and second eigenvalues
			eval1 = EVal1Array->GetTuple1(pointId);
			eval2 = EVal2Array->GetTuple1(pointId);

			// If both eigenvalues are positive, add them to the running sum
			if (eval1 > 0.0 && eval2)
			{
				eval1Sum += eval1;
				eval2Sum += eval2;
				numberOfValidEVals++;
			}

		} // if [pointId != -1]

	} // for [every seed point]

	// Check if the sum values are valid
	if (	numberOfValidEVals == 0	||
			eval1Sum <= 0.0			|| 
			eval2Sum <= 0.0			||
			eval2Sum > eval1Sum		) 
	{
		vtkErrorMacro(<< "Could not compute mean eigenvalues!");
		return;
	}

	// Divide by the number of valid eigenvalues to get the average
	eval1Sum /= (double) numberOfValidEVals;
	eval2Sum /= (double) numberOfValidEVals;

	// Set average eigenvalues for fODF sharpening
	this->SetEigenval1(eval1Sum);
	this->SetEigenval2(eval2Sum);
}


} // namespace bmia
