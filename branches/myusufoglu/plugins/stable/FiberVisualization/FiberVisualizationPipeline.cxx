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
 * FiberVisualizationPipeline.cxx
 *
 * 2010-10-05	Evert van Aart
 * - First Version.
 * 
 * 2010-10-22	Evert van Aart
 * - Added support for coloring using the "CellData" array.
 * 
 * 2011-02-01	Evert van Aart
 * - Added support for bypassing the simplification filter.
 *
 * 2011-04-18	Evert van Aart
 * - Moved the simplifcation filter to the "Helpers" library.
 *
 * 2011-04-26	Evert van Aart
 * - Improved progress reporting.
 *
 */


/** Includes */

#include "FiberVisualizationPipeline.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberVisualizationPipeline::FiberVisualizationPipeline(UserOutput * rUserOut)
{
	// Set default parameter values

	this->SimplifyStepSize		= 1.0f;
	this->SimplifyEnabled		= true;
	this->ColorType				= FiberVisualizationPlugin::FC_SingleColor;
	this->ColorShiftValues		= false;
	this->ColorUseAIWeighting	= false;
	this->ShapeType				= FiberVisualizationPlugin::FS_Streamlines; 
	this->ShapeNumberOfSides	= 6;
	this->ShapeRadius			= 0.5f;
	this->ShapeHyperScale		= 1.0f;
	this->LightingEnable		= false;
	this->LightingAmbient		=  0.2f;
	this->LightingDiffuse		=  0.6f;
	this->LightingSpecular		=  0.4f;
	this->LightingSpecularPower = 30.0f;
	this->ShadowsEnable			= false;
	this->ShadowsAmbient		= 0.1f;
	this->ShadowsDiffuse		= 0.3f;
	this->ShadowsWidth			= 4.0f;
	this->aiImageIndex			= -1;
	this->eigenImageIndex		= -1;
	this->lutIndex				= -1;
	this->updateMapper			= false;
	this->rebuildPipeline		= true;

	// Create the simplification filter. Since we always use this filter,
	// we can create it here.

	this->simplifyFilter = vtkStreamlineToSimplifiedStreamline::New();
	this->simplifyFilter->SetStepLength(this->SimplifyStepSize);

	// Set all other pointers to NULL

	this->colorFilterMEV		= NULL;
	this->colorFilterDirection	= NULL;
	this->colorFilterAI			= NULL;
	this->shapeFilter			= NULL;
	this->mapper				= NULL;
	this->eigenImageData		= NULL;
	this->aiImageData			= NULL;
	this->lut					= NULL;
	this->userOut				= rUserOut;
}


//------------------------------[ Destructor ]-----------------------------\\

FiberVisualizationPipeline::~FiberVisualizationPipeline()
{
	// Delete the simplification filter
	this->simplifyFilter->Delete();

	// Delete the shape filter
	if (this->shapeFilter)
	{
		this->userOut->deleteProgressBarForAlgorithm(this->shapeFilter);
		this->shapeFilter->Delete();
	}

	// Delete any existing color filter
	if (this->colorFilterMEV)
		this->colorFilterMEV->Delete();
	if (this->colorFilterDirection)
		this->colorFilterDirection->Delete();
	if (this->colorFilterAI)
		this->colorFilterAI->Delete();
}


//-------------------------[ setupSimplifyFilter ]-------------------------\\

void FiberVisualizationPipeline::setupSimplifyFilter(float rStepSize, bool rDoSimplify)
{
	// Don't do anything if the value hasn't changed
	if (rStepSize == this->SimplifyStepSize && rDoSimplify == this->SimplifyEnabled)
		return;

	// Set the new values
	this->SimplifyStepSize = rStepSize;
	this->SimplifyEnabled = rDoSimplify;
	this->simplifyFilter->SetStepLength(rStepSize);
	this->simplifyFilter->setDoBypass(!rDoSimplify);
	this->simplifyFilter->Modified();
}


//---------------------------[ setupColorFilter ]--------------------------\\

void FiberVisualizationPipeline::setupColorFilter(FiberVisualizationPlugin::FiberColor rColor, bool rShiftValues, bool rAIWeighting)
{
	// If the current coloring type is "MEV" and the new one isn't...
	if (this->colorFilterMEV && rColor != FiberVisualizationPlugin::FC_MEV)
	{
		// ...delete the existing MEV coloring filter...
		this->colorFilterMEV->Delete();
		this->colorFilterMEV = NULL;

		// ...and signal that the pipeline needs to be rebuilt.
		this->rebuildPipeline = true;
	}
	// Likewise, but for the "Direction" coloring type
	if (this->colorFilterDirection && rColor != FiberVisualizationPlugin::FC_Direction)
	{
		this->colorFilterDirection->Delete();
		this->colorFilterDirection = NULL;
		this->rebuildPipeline = true;
	}
	// Likewise, but for the "AI" coloring type
	if (this->colorFilterAI && rColor != FiberVisualizationPlugin::FC_AI)
	{
		this->colorFilterAI->Delete();
		this->colorFilterAI = NULL;
		this->rebuildPipeline = true;
	}

	// Main Eigenvector coloring
	if (rColor == FiberVisualizationPlugin::FC_MEV)
	{
		// If necessary, create a new MEV coloring filter
		if (!this->colorFilterMEV)
			this->colorFilterMEV = vtkMEVToStreamlineColor::New();

		// Set the filter parameters and pointers
		this->colorFilterMEV->setEigenImageData(this->eigenImageData);
		this->colorFilterMEV->setAIImageData(this->aiImageData);
		this->colorFilterMEV->setUseAIWeighting(rAIWeighting);
		this->colorFilterMEV->setShiftValues(rShiftValues);
		this->colorFilterMEV->Modified();

		// Update the mapper if necessary
		this->updateMapper = true;
	}

	// Fiber Direction coloring
	if (rColor == FiberVisualizationPlugin::FC_Direction)
	{
		// If necessary, create a new Direction coloring filter
		if (!this->colorFilterDirection)
			this->colorFilterDirection = vtkDirectionToStreamlineColor::New();

		// Set the filter parameters and pointers
		this->colorFilterDirection->setAIImageData(this->aiImageData);
		this->colorFilterDirection->setUseAIWeighting(rAIWeighting);
		this->colorFilterDirection->setShiftValues(rShiftValues);
		this->colorFilterDirection->Modified();

		// Update the mapper if necessary
		this->updateMapper = true;
	}

	// Anisotropy Index coloring
	if (rColor == FiberVisualizationPlugin::FC_AI)
	{
		// If necessary, create a new AI coloring filter
		if (!this->colorFilterAI)
			this->colorFilterAI = vtkProbeFilter::New();

		// Set the filter parameters and pointers
		this->colorFilterAI->SetSource(this->aiImageData);
		this->colorFilterAI->Modified();

		// Update the mapper if necessary
		this->updateMapper = true;
	}

	// If we change to one of the coloring types that do not need a color
	// filter, we still need to update the mapper, to correctly set the
	// color rendering options.

	if ((rColor == FiberVisualizationPlugin::FC_CellData    && this->ColorType != FiberVisualizationPlugin::FC_CellData   ) ||
		(rColor == FiberVisualizationPlugin::FC_FiberData   && this->ColorType != FiberVisualizationPlugin::FC_FiberData  ) ||
		(rColor == FiberVisualizationPlugin::FC_SingleColor && this->ColorType != FiberVisualizationPlugin::FC_SingleColor) )
	{
		this->updateMapper = true;		
	}

	// Store coloring options
	this->ColorType = rColor;
	this->ColorUseAIWeighting = rAIWeighting;
	this->ColorShiftValues = rShiftValues;
}


//---------------------------[ setupShapeFilter ]--------------------------\\

void FiberVisualizationPipeline::setupShapeFilter(FiberVisualizationPlugin::FiberShape rType, int rNumberOfSides, float rRadius, float rHyperScale)
{
	// Streamlines
	if (rType == FiberVisualizationPlugin::FS_Streamlines)
	{
		// If the shape type hasn't changed, only update the shape options
		if (this->ShapeType == FiberVisualizationPlugin::FS_Streamlines)
		{
			this->ShapeNumberOfSides = rNumberOfSides;
			this->ShapeRadius = rRadius;
			this->ShapeHyperScale = rHyperScale;
			this->updateMapper = true;
			return;
		}

		// Delete existing shape filters (since Streamlines do not require a filter)
		if (this->shapeFilter)
		{
			this->userOut->deleteProgressBarForAlgorithm(this->shapeFilter);
			this->shapeFilter->Delete();
			this->shapeFilter = NULL;
		}
	}

	// Streamtubes
	else if (rType == FiberVisualizationPlugin::FS_Streamtubes)
	{
		// If the current shape type is also "Streamtubes"...
		if (this->ShapeType == FiberVisualizationPlugin::FS_Streamtubes)
		{
			// ... set the new shape options...
			vtkStreamlineToStreamTube * filterCast = (vtkStreamlineToStreamTube *) this->shapeFilter;
			filterCast->SetNumberOfSides(rNumberOfSides);
			filterCast->SetRadius(rRadius);
			filterCast->Modified();

			// ...and store the options in this class.
			this->ShapeNumberOfSides = rNumberOfSides;
			this->ShapeRadius = rRadius;
			this->ShapeHyperScale = rHyperScale;

			return;		
		}

		// Delete existing shape filter
		if (this->shapeFilter)
		{
			this->userOut->deleteProgressBarForAlgorithm(this->shapeFilter);
			this->shapeFilter->Delete();
			this->shapeFilter = NULL;
		}

		// Create a new Streamtubes filter
		vtkStreamlineToStreamTube * filterCast = vtkStreamlineToStreamTube::New();

		// Set filter variables
		filterCast->SetNumberOfSides(rNumberOfSides);
		filterCast->SetRadius(rRadius);

		// Store filter pointer
		this->shapeFilter = (vtkStreamlineToStreamGeneralCylinder *) filterCast;
		this->userOut->createProgressBarForAlgorithm(this->shapeFilter, "Fiber Visualization");
	}

	// Hyperstreamtubes
	else if (rType == FiberVisualizationPlugin::FS_Hyperstreamtubes)
	{
		// If the current shape type is also "Hyperstreamtubes"...
		if (this->ShapeType == FiberVisualizationPlugin::FS_Hyperstreamtubes)
		{
			// ... set the new shape options...
			vtkStreamlineToHyperStreamline * filterCast = (vtkStreamlineToHyperStreamline *) this->shapeFilter;
			filterCast->SetNumberOfSides(rNumberOfSides);
			filterCast->SetHyperScale(rHyperScale);
			filterCast->Modified();

			// ...and store the options in this class.
			this->ShapeNumberOfSides = rNumberOfSides;
			this->ShapeRadius = rRadius;
			this->ShapeHyperScale = rHyperScale;

			return;		
		}

		// Delete existing shape filter
		if (this->shapeFilter)
		{
			this->userOut->deleteProgressBarForAlgorithm(this->shapeFilter);
			this->shapeFilter->Delete();
			this->shapeFilter = NULL;
		}

		// Create a new Hyperstreamtubes filter
		vtkStreamlineToHyperStreamline * filterCast = vtkStreamlineToHyperStreamline::New();

		// Set filter variables
		filterCast->SetNumberOfSides(rNumberOfSides);
		filterCast->SetHyperScale(rHyperScale);
		filterCast->SetEigenData(this->eigenImageData);

		// Store filter pointer
		this->shapeFilter = (vtkStreamlineToStreamGeneralCylinder *) filterCast;
		this->userOut->createProgressBarForAlgorithm(this->shapeFilter, "Fiber Visualization");
	}

	// Hyperstreamprisms and Streamribbons (both use the same filter)
	else if (	rType == FiberVisualizationPlugin::FS_Hyperstreamprisms || 
				rType == FiberVisualizationPlugin::FS_Streamribbons)
	{
		// If the current shape type is also "Hyperstreamprisms" or "Streamribbons"...
		if (this->ShapeType == FiberVisualizationPlugin::FS_Hyperstreamprisms || 
			this->ShapeType == FiberVisualizationPlugin::FS_Streamribbons)
		{
			// ... set the new shape options...
			vtkStreamlineToHyperStreamPrisma * filterCast = (vtkStreamlineToHyperStreamPrisma *) this->shapeFilter;
			filterCast->SetNumberOfSides(rNumberOfSides);
			filterCast->SetHyperScale(rHyperScale);
			filterCast->SetTubeNotRibbons(rType == FiberVisualizationPlugin::FS_Hyperstreamprisms);
			filterCast->Modified();

			// ...and store the options in this class.
			this->ShapeNumberOfSides = rNumberOfSides;
			this->ShapeRadius = rRadius;
			this->ShapeHyperScale = rHyperScale;
	
			this->ShapeType = rType;

			return;		
		}

		// Delete existing shape filter
		if (this->shapeFilter)
		{
			this->userOut->deleteProgressBarForAlgorithm(this->shapeFilter);
			this->shapeFilter->Delete();
			this->shapeFilter = NULL;
		}

		// Create a new Hyperstreamprisms filter
		vtkStreamlineToHyperStreamPrisma * filterCast = vtkStreamlineToHyperStreamPrisma::New();

		// Set filter variables
		filterCast->SetNumberOfSides(rNumberOfSides);
		filterCast->SetHyperScale(rHyperScale);
		filterCast->SetEigenData(this->eigenImageData);
		filterCast->SetTubeNotRibbons(rType == FiberVisualizationPlugin::FS_Hyperstreamprisms);

		// Store filter pointer
		this->shapeFilter = (vtkStreamlineToStreamGeneralCylinder *) filterCast;
		this->userOut->createProgressBarForAlgorithm(this->shapeFilter, "Fiber Visualization");
	}

	// Store shape options
	this->ShapeType = rType;
	this->ShapeNumberOfSides = rNumberOfSides;
	this->ShapeRadius = rRadius;
	this->ShapeHyperScale = rHyperScale;

	// Update the mapper if necessary, and rebuild the pipeline

	this->updateMapper = true;	
	this->rebuildPipeline = true;

}


//-----------------------------[ setupMapper ]-----------------------------\\

void FiberVisualizationPipeline::setupMapper()
{
	// Only continue if a change in one of the filters requires the mapper
	// to be updated.

	if (!this->updateMapper)
		return;

	// Delete existing mapper

	if (this->mapper)
	{
		this->mapper->Delete();
		this->mapper = NULL;
	}

	// Create a GPU mapper for Streamlines
	if (ShapeType == FiberVisualizationPlugin::FS_Streamlines)
	{
		// Create the mapper
		vtkFiberMapper * newMapper = vtkFiberMapper::New();

		// Store mapper pointer
		this->mapper = (vtkPolyDataMapper *) newMapper;
	}
	// Create a general polydata mapper
	else
	{
		// Create the mapper
		vtkPolyDataMapper * newMapper = vtkPolyDataMapper::New();

		// Store mapper pointer
		this->mapper = newMapper;
	}

	// Use look-up tables for AI coloring...
	if (this->ColorType == FiberVisualizationPlugin::FC_AI || 
		this->ColorType == FiberVisualizationPlugin::FC_FiberData)
	{
		this->mapper->SetColorModeToMapScalars();
		this->mapper->SetLookupTable(this->lut);
	}
	// ...and direct RGB coloring otherwise
	else
	{
		this->mapper->SetColorModeToDefault();
	}

	// Set whether or not to color the fibers using scalar data
	if (this->ColorType == FiberVisualizationPlugin::FC_SingleColor)
	{
		this->mapper->ScalarVisibilityOff();
	}
	else
	{
		this->mapper->ScalarVisibilityOn();
	}

	// Set whether to use cell data or point data for coloring
	if (this->ColorType == FiberVisualizationPlugin::FC_CellData)
	{
		this->mapper->SetScalarModeToUseCellData();
	}
	else if (this->ColorType == FiberVisualizationPlugin::FC_AI			||
			 this->ColorType == FiberVisualizationPlugin::FC_Direction	||
			 this->ColorType == FiberVisualizationPlugin::FC_FiberData	||
			 this->ColorType == FiberVisualizationPlugin::FC_MEV		)
	{
		 this->mapper->SetScalarModeToUsePointData(); 
	}

	// Update is done, reset update flag
	this->updateMapper = false;

	// Signal that the pipeline needs to be rebuilt
	this->rebuildPipeline = true;
}


//----------------------------[ setupLighting ]----------------------------\\

void FiberVisualizationPipeline::setupLighting(bool enable, float amb, float dif, float spec, float specpow)
{
	// Store lighting options
	this->LightingEnable		= enable;
	this->LightingAmbient		= amb;
	this->LightingDiffuse		= dif;
	this->LightingSpecular		= spec;
	this->LightingSpecularPower = specpow;

	// If we're using Streamlines, copy the lighting options to the mapper
	if (this->ShapeType == FiberVisualizationPlugin::FS_Streamlines)
	{
		vtkFiberMapper * mapperCast = (vtkFiberMapper *) this->mapper;
		mapperCast->SetLighting(enable);
		mapperCast->SetAmbientContribution(amb);
		mapperCast->SetDiffuseContribution(dif);
		mapperCast->SetSpecularContribution(spec);
		mapperCast->SetSpecularPower(specpow);
	}
}


//-----------------------------[ setupShadows ]----------------------------\\

void FiberVisualizationPipeline::setupShadows(bool enable, float amb, float dif, float width)
{
	// Store lighting options
	this->ShadowsEnable		= enable;
	this->ShadowsAmbient	= amb;
	this->ShadowsDiffuse	= dif;
	this->ShadowsWidth		= width;

	// If we're using Streamlines, copy the lighting options to the mapper
	if (this->ShapeType == FiberVisualizationPlugin::FS_Streamlines)
	{
		vtkFiberMapper * mapperCast = (vtkFiberMapper *) this->mapper;
		mapperCast->SetShadowing(enable);
		mapperCast->SetAmbientContributionShadow(amb);
		mapperCast->SetDiffuseContributionShadow(dif);
		mapperCast->SetShadowLineWidth(width);
	}
}


//----------------------------[ setupPipeline ]----------------------------\\

void FiberVisualizationPipeline::setupPipeline(vtkPolyData * input, vtkActor * actor)
{
	// Don't do anything if we don't need to rebuild the pipeline
	if (!this->rebuildPipeline)
		return;
	
	// We need a mapper to build the pipeline
	if (!this->mapper)
		return;

	// First stage: Simplification filter
	this->simplifyFilter->SetInput(input);

	// The "currentOutput" pointer allows us to skip the two optional
	// filters (coloring and shape) if necessary.

	vtkAlgorithmOutput * currentOutput = this->simplifyFilter->GetOutputPort();

	// Main Eigenvector coloring
	if (this->colorFilterMEV)
	{
		// Set input to output of the simplification filter
		this->colorFilterMEV->SetInputConnection(0, currentOutput);

		// Update current output pointer
		currentOutput = this->colorFilterMEV->GetOutputPort(0);
	}
	// Fiber Direction coloring
	else if (this->colorFilterDirection)
	{
		// Set input to output of the simplification filter
		this->colorFilterDirection->SetInputConnection(0, currentOutput);

		// Update current output pointer
		currentOutput = this->colorFilterDirection->GetOutputPort(0);
	}
	// Anisotropy Index coloring
	else if (this->colorFilterAI)
	{
		// Set input to output of the simplification filter
		this->colorFilterAI->SetInputConnection(0, currentOutput);

		// Update current output pointer
		currentOutput = this->colorFilterAI->GetOutputPort(0);
	}

	// The shape filter is only NULL for Streamlines
	if (this->shapeFilter)
	{
		// Set input to the current output pointer
		this->shapeFilter->SetInputConnection(0, currentOutput);


		// Update current output pointer
		currentOutput = this->shapeFilter->GetOutputPort(0);
	}

	// Set mapper input to current output pointer
	this->mapper->SetInputConnection(0, currentOutput);

	// Store the mapper in the actor
	actor->SetMapper(this->mapper);

	// Done rebuilding
	this->rebuildPipeline = false;
}


//----------------------------[ modifiedInput ]----------------------------\\

void FiberVisualizationPipeline::modifiedInput()
{
	this->simplifyFilter->Modified();
}


} // namespace bmia