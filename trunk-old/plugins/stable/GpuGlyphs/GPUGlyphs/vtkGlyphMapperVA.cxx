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
 * vtkGlyphMapperVA.cxx
 *
 * 2008-02-27	Tim Peeters
 * - First version of vtkGlyphMapper.cxx.
 *
 * 2008-09-03	Tim Peeters
 * - First version of vtkGlyphMapperVA.cxx based on vtkGlyphMapper.cxx.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for transforming and coloring tensors.
 * - Cleaned up code, added comments.
 *
 * 2011-02-18	Evert van Aart
 * - Improved error handling for non-supported GPUs.
 *
 */


/** Includes */

#include "vtkGlyphMapperVA.h"


namespace bmia {


vtkCxxSetObjectMacro(vtkGlyphMapperVA, SeedPoints, vtkPointSet);


//-----------------------------[ Constructor ]-----------------------------\\

vtkGlyphMapperVA::vtkGlyphMapperVA()
{
	// Set default values
	this->Initialized			= false;
	this->SP					= NULL;
	this->SeedPoints			= NULL;
	this->MaxGlyphRadius		= 0.5;
	this->GlyphScaling			= 1.0;
	this->EyePosition			= NULL;
	this->UEyePosition			= NULL;
	this->ULightPosition		= NULL;
	this->UMaxGlyphRadius		= NULL;
	this->UGlyphScaling			= NULL;
	this->CurrentCamera			= NULL;
	this->transformationMatrix	= NULL;
	this->aiImage				= NULL;
	this->lut					= NULL;
	this->notSupported			= false;
}


//-------------------------[ SetupShaderUniforms ]-------------------------\\

void vtkGlyphMapperVA::SetupShaderUniforms()
{
	// Delete existing uniforms
	if (this->UEyePosition != NULL)
		this->UEyePosition->Delete();
	if (this->ULightPosition != NULL) 
		this->ULightPosition->Delete();
	if (this->UMaxGlyphRadius != NULL) 
		this->UMaxGlyphRadius->Delete();
	if (this->UGlyphScaling != NULL) 
		this->UGlyphScaling->Delete();

	// Create new uniforms
	this->UEyePosition = vtkUniformVec3::New();
	this->UEyePosition->SetName("EyePosition");
  
	this->ULightPosition = vtkUniformVec3::New();
	this->ULightPosition->SetName("LightPosition");

	this->UMaxGlyphRadius = vtkUniformFloat::New();
	this->UMaxGlyphRadius->SetName("MaxGlyphRadius");

	this->UGlyphScaling = vtkUniformFloat::New();
	this->UGlyphScaling->SetName("GlyphScaling");
	this->UGlyphScaling->SetValue(this->GlyphScaling);

	// Add the uniforms to the shader program. The uniforms that are
	// not included here are added in the relevant subclasses.

	if (this->SP) 
	{
		this->SP->AddShaderUniform(this->UEyePosition);
		this->SP->AddShaderUniform(this->ULightPosition);	
	}
  
	this->CurrentCamera = NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkGlyphMapperVA::~vtkGlyphMapperVA()
{
	if (this->SP)
	{
		this->SP->Delete();
	}

	// Delete existing uniforms
	if (this->UEyePosition != NULL)
		this->UEyePosition->Delete();
	if (this->ULightPosition != NULL) 
		this->ULightPosition->Delete();
	if (this->UMaxGlyphRadius != NULL) 
		this->UMaxGlyphRadius->Delete();
	if (this->UGlyphScaling != NULL) 
		this->UGlyphScaling->Delete();
}


//------------------------------[ Initialize ]-----------------------------\\

void vtkGlyphMapperVA::Initialize(vtkRenderWindow * renwin)
{
	// Nothing to do
	if (this->Initialized)
		return;

	// Create a new extension manager
	vtkOpenGLExtensionManager * extensions = vtkOpenGLExtensionManager::New();
	extensions->SetRenderWindow(renwin);

	// Check if version 2.0 of OpenGL is supported
	int supports_GL_VERSION_2_0 = extensions->ExtensionSupported("GL_VERSION_2_0");

	// If so, load this extension
	if (supports_GL_VERSION_2_0)
	{
		extensions->LoadExtension("GL_VERSION_2_0");
	}
	else
	{
		// Otherwise, print a warning and return
		vtkWarningMacro(<<"GL_VERSION_2_0 is not supported!");

		QMessageBox::warning(NULL, "GPU Glyph Mapper",	QString("Your GPU does not support version 2.0 of OpenGL.") +
														QString("\nAs a result, the GPU Glyphs plugin will not work.") +
														QString("\nPlease update the drivers of your video card and try again."));
		
		this->notSupported = true;
		return;
	}

	extensions->Delete();

	// Successfully initialized
	this->Initialized = true;
}


//--------------------------[ SetMaxGlyphRadius ]--------------------------\\

void vtkGlyphMapperVA::SetMaxGlyphRadius(float r)
{
	// Store the new radius in the local variable and in the shader uniform
	if (this->MaxGlyphRadius != r)
	{
		this->MaxGlyphRadius = r;
		this->UMaxGlyphRadius->SetValue(this->MaxGlyphRadius);
		this->Modified();
	}
}


//--------------------------------[ Render ]-------------------------------\\

void vtkGlyphMapperVA::Render(vtkRenderer * ren, vtkVolume * vol)
{
	// Do nothing if this mapper is not supported
	if (this->notSupported)
		return;

	// Give the current render window focus
	ren->GetRenderWindow()->MakeCurrent();

	// Initialize if necessary
	if (!this->Initialized)
	{
		this->Initialize(ren->GetRenderWindow());

		// Do nothing if this mapper is not supported
		if (this->notSupported)
			return;
	}

	// Get the current camera, copy its position to the "EyePosition" vectors
	this->CurrentCamera = ren->GetActiveCamera();
	this->EyePosition = this->CurrentCamera->GetPosition(); 
	this->UEyePosition->SetValue(	(float) this->EyePosition[0], 
									(float) this->EyePosition[1], 
									(float) this->EyePosition[2]);

	// Set the light position to the eye position
	this->ULightPosition->SetValue(	(float) this->EyePosition[0], 
									(float) this->EyePosition[1], 
									(float) this->EyePosition[2]);

	// Activate the shader program
	if (this->SP) 
		this->SP->Activate();

	// Draw the glyphs
	this->DrawPoints();

	// Deactivate the shader program
	if (this->SP) 
		this->SP->Deactivate();
  
	// Clear the eye position pointer
	this->EyePosition = NULL;
}


//---------------------------[ SetGlyphScaling ]---------------------------\\

void vtkGlyphMapperVA::SetGlyphScaling(float scale)
{
	// Store the new glyph scale
	this->GlyphScaling = scale;
	this->UGlyphScaling->SetValue(this->GlyphScaling);
	this->SetMaxGlyphRadius(this->GlyphScaling);
}


//---------------------------[ transformPoint ]----------------------------\\

void vtkGlyphMapperVA::transformPoint(double * p, bool translate)
{
	// Do nothing if there's no transformation matrix
	if (!this->transformationMatrix)
		return;

	// Copy 3D vector into a 4D vector
	double p4d[4];

	p4d[0] = p[0]; 
	p4d[1] = p[1]; 
	p4d[2] = p[2]; 

	// If the last element is one, we apply a translation (fourth column of
	// the transformation matrix); if it is zero, we do not translate, but we
	// do perform the other transformations.

	p4d[3] = translate ? 1.0 : 0.0;

	// Transform the point
	this->transformationMatrix->MultiplyPoint(p4d, p4d);

	// Copy the result back to the 3D vector
	p[0] = p4d[0];
	p[1] = p4d[1];
	p[2] = p4d[2];
}


} // namespace bmia
