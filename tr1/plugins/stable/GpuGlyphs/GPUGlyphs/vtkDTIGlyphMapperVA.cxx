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
 * vtkDTIGlyphMapperVA.cxx
 *
 * 2008-09-01	Tim Peeters
 * - First version
 *
 * 2011-02-08	Evert van Aart
 * - Added support for coloring the glyphs using LUT or weighted RGB.
 * - Cleaned up the code, added comments.
 *
 * 2011-07-07	Evert van Aart
 * - After rendering the glyphs, put GL options that were disabled/enabled during
 *   rendering back to their original state. In particular, failing to re-enable
 *   blending cause problems with text rendering.
 *
 */

#include "vtkDTIGlyphMapperVA.h"


namespace bmia {


vtkStandardNewMacro(vtkDTIGlyphMapperVA);


//-----------------------------[ Constructor ]-----------------------------\\

vtkDTIGlyphMapperVA::vtkDTIGlyphMapperVA()
{
	// Create a reader for the shader program
	vtkMyShaderProgramReader * spReader = vtkMyShaderProgramReader::New();
	spReader->SetFileName("shaders/tensorglyph.prog");
	spReader->Execute();

	// Get the shader program
	this->SP = spReader->GetOutput();
	this->SP->Register(this);
  
	// Delete the reader
	spReader->Delete();

	// Setup the uniforms
	this->SetupShaderUniforms();

	// Add the glyph scaling uniform to the shader
	if (this->SP) 
		this->SP->AddShaderUniform(this->UGlyphScaling);

	// Coloring method is RGB by default
	this->currentColoringMethod = vtkDTIGlyphMapperVA::CM_RGB;

	// Hard-coded value for the square root of one third.
	this->SQRT13 = 0.57735026918962576450914878050196;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkDTIGlyphMapperVA::~vtkDTIGlyphMapperVA()
{
	// Everything is done in the superclass
}


//------------------------------[ DrawPoints ]-----------------------------\\

void vtkDTIGlyphMapperVA::DrawPoints()
{
	// Get the input image
	vtkImageData * input = this->GetInput();

	if (!input)
	{
		vtkErrorMacro(<< "No input for the DTI GPU Glyphs!");
		return;
	}

	// Get the point data for the input image
	vtkPointData * inputPD = input->GetPointData();

	if (!inputPD)
	{
		vtkErrorMacro(<< "Input image does not contain point data!");
		return;
	}

	// Get the eigenvectors and -scalars
	vtkDataArray * eigenVectors1 = inputPD->GetVectors("Eigenvector 1");
	vtkDataArray * eigenVectors2 = inputPD->GetVectors("Eigenvector 2");
	vtkDataArray * eigenVectors3 = inputPD->GetVectors("Eigenvector 3");
	vtkDataArray * eigenValues1  = inputPD->GetScalars("Eigenvalue 1");
	vtkDataArray * eigenValues2  = inputPD->GetScalars("Eigenvalue 2");
	vtkDataArray * eigenValues3  = inputPD->GetScalars("Eigenvalue 3");

	if (!(eigenVectors1 && eigenVectors2 && eigenVectors3 && eigenValues1 && eigenValues2 && eigenValues3))
	{
		vtkErrorMacro(<< "Missing eigenvector and/or eigenvalue arrays!");
		return;
	}

	// Check if the seed points have been correctly set
	vtkPointSet * pointSet = this->GetSeedPoints();

	if (!pointSet)
	{
		vtkErrorMacro(<<"No point set to render!");
		return;
	}

	vtkIdType numberOfSeeds = pointSet->GetNumberOfPoints();

	// Do nothing if there are no seed points
	if (numberOfSeeds == 0)
		return;

	// Check if the shader program has been loaded
	if (!(this->SP))
	{
		vtkErrorMacro(<< "Shader program has not yet been loaded!");
		return;
	}

	// Initialize the attributes of the shader program
	GLint eigenVector1Att = vtkgl::GetAttribLocation(this->SP->GetHandle(), "Eigenvector1");
	GLint eigenVector2Att = vtkgl::GetAttribLocation(this->SP->GetHandle(), "Eigenvector2");
	GLint eigenVector3Att = vtkgl::GetAttribLocation(this->SP->GetHandle(), "Eigenvector3");
	GLint eigenValuesAtt  = vtkgl::GetAttribLocation(this->SP->GetHandle(), "Eigenvalues");
	GLint positionAtt     = vtkgl::GetAttribLocation(this->SP->GetHandle(), "GlyphPosition");
	GLint colorAtt        = vtkgl::GetAttribLocation(this->SP->GetHandle(), "Color");

	// Scalar array of the AI image
	vtkDataArray * scalars = NULL;

	// Range of the scalars
	double scalarRange[2];

	// We only need the scalars if the coloring method is LUT or weighted RGB
	if (	this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_LUT   || 
			this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_WRGBA ||
			this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_WRGBB )
	{
		// Check if the input is correct
		if (!(this->aiImage))
		{
			vtkErrorMacro(<< "Anisotropy Image not set!");
			return;
		}

		if (!(this->aiImage->GetPointData()))
		{
			vtkErrorMacro(<< "Anisotropy Image does not contain point data!");
			return;
		}

		scalars = this->aiImage->GetPointData()->GetScalars();

		if (!scalars)
		{
			vtkErrorMacro(<< "Anisotropy Image does not contain scalar values!");
			return;
		}

		// Get the range of the scalars
		scalars->GetRange(scalarRange);

		// Avoid division by zero
		if (scalarRange[0] == scalarRange[1])
			scalarRange[0] -= 1.0;

		// Check if the LUT has been set
		if (this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_LUT && this->lut == NULL)
		{
			vtkErrorMacro(<< "Look-Up Table not set!");
			return;
		}
	}
	// Update the input image
	input->Update();

	// Some GL features will be enabled or disabled; store their current state now,
	// so that we can restore them at the end of the render pass.

	bool lightingWasEnabled		= glIsEnabled(GL_LIGHTING);
	bool blendingWasEnabled		= glIsEnabled(GL_BLEND);
	bool tex2dWasEnabled		= glIsEnabled(GL_TEXTURE_2D);
	bool cullFaceWasDisabled	= glIsEnabled(GL_CULL_FACE);
		
	// Setup OpenGL
	glDisable(GL_TEXTURE_2D);
	glPointSize(2.0);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);

	// Position of the glyph
	double p[3];

	// Bounding box of the glyph
	double BB[8][3];

	// Maximum glyph radius
	double r = this->MaxGlyphRadius;

	// Point ID in the input image
	vtkIdType pointId;

	// Start drawing
	glBegin(GL_QUADS);

	// Loop through all seed points
	for (vtkIdType seedPointID = 0; seedPointID < numberOfSeeds; seedPointID++)
	{
		// Get the coordinates of the current point
		pointSet->GetPoint(seedPointID, p);

		// Find the corresponding point in the input image
		pointId = input->FindPoint(p);

		// Check if the point was found
		if (pointId == -1)
			continue;
		
		// Check if the point has a positive first eigenvalue
		if (eigenValues1->GetTuple1(pointId) < 0.0)
			continue;

		// Transform the glyph position, and copy it to the attributes
		this->transformPoint(p, true);
		vtkgl::VertexAttrib3dv(positionAtt, p);

		// Color weight
		double w = 1.0;

		if (scalars)
		{
			// Get the current scalar
			w = scalars->GetTuple1(pointId);

			// For weighted RGB, normalize the scalar to the range 0-1
			if (this->currentColoringMethod != vtkDTIGlyphMapperVA::CM_LUT)
				w = (w - scalarRange[0]) / (scalarRange[1] - scalarRange[0]);
		}
		
		// Temporary eigenvector array
		double tempEVec[3];

		// Get the first eigenvector
		eigenVectors1->GetTuple(pointId, &tempEVec[0]);

		// Color of the glyph
		double c[3];

		// RGB: Use the absolute eigenvector elements
		if (this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_RGB)
		{
			c[0] = fabs(tempEVec[0]);
			c[1] = fabs(tempEVec[1]);
			c[2] = fabs(tempEVec[2]);
		}
		// Weighted RGB (Lightness): Multiply absolute eigenvector values by "w".
		else if (this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_WRGBB)
		{
			c[0] = fabs(tempEVec[0]) * w;
			c[1] = fabs(tempEVec[1]) * w;
			c[2] = fabs(tempEVec[2]) * w;
		}
		// Weighted RGB (Saturation): Decrease saturation based on "w".
		else if (this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_WRGBA)
		{
			c[0] = fabs(tempEVec[0]) * w + (1.0 - w) * this->SQRT13;
			c[1] = fabs(tempEVec[1]) * w + (1.0 - w) * this->SQRT13;
			c[2] = fabs(tempEVec[2]) * w + (1.0 - w) * this->SQRT13;
		}
		// LUT: Map "w" through the current LUT
		else if (this->currentColoringMethod == vtkDTIGlyphMapperVA::CM_LUT)
		{
			this->lut->GetColor(w, c);
		}

		// Copy the color to the attributes
		vtkgl::VertexAttrib3dv(colorAtt, c);

		// Transform and store the first eigenvector
		this->transformPoint(tempEVec, false);
		vtkgl::VertexAttrib3dv(eigenVector1Att, &tempEVec[0]);
	
		// Get, transform and store the second eigenvector
		eigenVectors2->GetTuple(pointId, &tempEVec[0]);
		this->transformPoint(tempEVec, false);
		vtkgl::VertexAttrib3dv(eigenVector2Att, &tempEVec[0]);

		// Get, transform and store the third eigenvector
		eigenVectors3->GetTuple(pointId, &tempEVec[0]);
		this->transformPoint(tempEVec, false);
		vtkgl::VertexAttrib3dv(eigenVector3Att, &tempEVec[0]);

		// Copy the eigenvalues to the attributes
		vtkgl::VertexAttrib3d(eigenValuesAtt,	eigenValues1->GetTuple1(pointId), 
												eigenValues2->GetTuple1(pointId), 
												eigenValues3->GetTuple1(pointId));

		// Compute the bounding box
		BB[0][0] = p[0] - r; BB[0][1] = p[1] - r; BB[0][2] = p[2] - r;
		BB[1][0] = p[0] + r; BB[1][1] = p[1] - r; BB[1][2] = p[2] - r;
		BB[2][0] = p[0] + r; BB[2][1] = p[1] + r; BB[2][2] = p[2] - r;
		BB[3][0] = p[0] - r; BB[3][1] = p[1] + r; BB[3][2] = p[2] - r;
		BB[4][0] = p[0] - r; BB[4][1] = p[1] - r; BB[4][2] = p[2] + r;
		BB[5][0] = p[0] + r; BB[5][1] = p[1] - r; BB[5][2] = p[2] + r;
		BB[6][0] = p[0] + r; BB[6][1] = p[1] + r; BB[6][2] = p[2] + r;
		BB[7][0] = p[0] - r; BB[7][1] = p[1] + r; BB[7][2] = p[2] + r;

		// Set the bounding box
		glTexCoord3dv(p);
		glVertex3dv(BB[0]); glVertex3dv(BB[3]); glVertex3dv(BB[2]); glVertex3dv(BB[1]);
		glVertex3dv(BB[0]); glVertex3dv(BB[4]); glVertex3dv(BB[7]); glVertex3dv(BB[3]);
		glVertex3dv(BB[0]); glVertex3dv(BB[1]); glVertex3dv(BB[5]); glVertex3dv(BB[4]);
		glVertex3dv(BB[1]); glVertex3dv(BB[2]); glVertex3dv(BB[6]); glVertex3dv(BB[5]);
		glVertex3dv(BB[3]); glVertex3dv(BB[7]); glVertex3dv(BB[6]); glVertex3dv(BB[2]);
		glVertex3dv(BB[5]); glVertex3dv(BB[6]); glVertex3dv(BB[7]); glVertex3dv(BB[4]);
	}

	// Stop drawing quads
	glEnd();

	// Set options that were disabled/enabled back to how they were
	if (lightingWasEnabled)		glEnable(GL_LIGHTING);
	if (blendingWasEnabled)		glEnable(GL_BLEND);
	if (tex2dWasEnabled)		glEnable(GL_TEXTURE_2D);
	if (cullFaceWasDisabled)	glDisable(GL_CULL_FACE);
}


} // namespace bmia
