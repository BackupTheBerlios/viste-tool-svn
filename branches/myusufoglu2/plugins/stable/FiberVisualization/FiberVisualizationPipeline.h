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
 * FiberVisualizationPipeline.h
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


/** ToDo List for "FiberVisualizationPipeline"
	Last updated 05-10-2010 by Evert van Aart

    - Add support for color look-up tables.
*/


#ifndef bmia_FiberVisualizationPipeline_h
#define bmia_FiberVisualizationPipeline_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "FiberVisualizationPlugin.h"
#include "vtkMEVToStreamlineColor.h"
#include "vtkDirectionToStreamlineColor.h"
#include "Helpers/vtkStreamlineToSimplifiedStreamline.h"
#include "vtkStreamlineToStreamGeneralCylinder.h"
#include "vtkStreamlineToStreamTube.h"
#include "vtkStreamlineToHyperStreamline.h"
#include "vtkStreamlineToHyperStreamPrisma.h"
#include "HWShading/vtkFiberMapper.h"
#include "core/UserOutput.h"

/** Includes - VTK */

#include <vtkPolyData.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkImageData.h>
#include <vtkProbeFilter.h>
#include <vtkScalarsToColors.h>


namespace bmia {


/** This class contains the visualization pipeline for fiber data sets. Each
	set of fibers will have one instantiation of this class associated with it,
	containing a mapper, one or more processing filters, and all relevant para-
	meters. Currently, the pipeline contains four stages, two of which are
	optional: a fiber simplification filter to reduce the number of fiber 
	points and smooth the fibers, a coloring filter (optional), a shape 
	filter (optional), and a mapper. Pipelines are controlled and maintained
	by the "FiberVisualizationPlugin" class.
*/

class FiberVisualizationPipeline
{
	public:

		/** Constructor */

		FiberVisualizationPipeline(UserOutput * rUserOut);

		/** Destructor */

		~FiberVisualizationPipeline();

		/** Set options for the simplification filter.
			@param rStepSize		New step size. 
			@param rDoSimplify		Whether or not the simplification filter needs to execute. */

		void setupSimplifyFilter(float rStepSize, bool rDoSimplify);

		/** Set options for the coloring filter. If necessary, delete the existing
			color filter and create a new one. Doing so will rebuild the pipeline.
			@param rColor			New coloring type.
			@param rShiftValues		Shift vector values.
			@param rAIWeighting		Apply AI weighting. */

		void setupColorFilter(FiberVisualizationPlugin::FiberColor rColor, bool rShiftValues, bool rAIWeighting);

		/** Set options for the shape filter. If necessary, delete the existing
			shape filter and create a new one. Doing so will rebuild the pipeline.
			@param rType			New shape type.
			@param rNumberOfSides	Number of sides of the 3D tubes.
			@param rRadius			Radius of the tubes.
			@param rHyperScale		Hyper scale for HyperStreamlines. */

		void setupShapeFilter(FiberVisualizationPlugin::FiberShape rType, int rNumberOfSides, float rRadius, float rHyperScale);

		/** Set options of the mapper. This function should always be called _after_
			the previous two functions have been called; depending on what happened
			in these functions, it may be necessary to create a new mapper or change
			the settings of the existing one. */

		void setupMapper();

		/** Set lighting options for use with the GPU mapper. 
			@param enable			Enable lighting.
			@param amb				Ambient contribution.
			@param dif				Diffusion contribution.
			@param spec				Specular contribution.
			@param specpow			Specular power. */

		void setupLighting(bool enable, float amb, float dif, float spec, float specpow);

		/** Set shadows options for use with the GPU mapper. 
			@param enable			Enable shadows.
			@param amb				Ambient contribution.
			@param dif				Diffusion contribution.
			@param width			Shadow line width. */

		void setupShadows(bool enable, float amb, float dif, float width);

		/** (Re-)build the visualization pipeline.
			@param input			Input fiber set.
			@param actor			Output actor. */

		void setupPipeline(vtkPolyData * input, vtkActor * actor);

		/** Call when the input poly data has been modified, but the pipeline itself
			has not changed. This forces a re-execution of the simplification filter,
			after which the other filters and the mapper will also update. */

		void modifiedInput();

		/** Image data sets containing eigensystem and AI values, respectively. */
		
		vtkImageData * eigenImageData;
		vtkImageData * aiImageData;

		/** Index of AI and eigensystem data images in the UI combo box */

		int aiImageIndex;
		int eigenImageIndex;

		/** Simplification Filter Options */

		float SimplifyStepSize;								// Step size
		bool SimplifyEnabled;								// Enables bypassing

		/** Color Filter Options */	

		FiberVisualizationPlugin::FiberColor ColorType;		// Coloring type
		bool ColorUseAIWeighting;							// Weight RGB values using AI values
		bool ColorShiftValues;								// Shift vector values

		/** Shape Filter Options */

		FiberVisualizationPlugin::FiberShape ShapeType;		// Shape type
		int ShapeNumberOfSides;								// Number of sides of the 3D tubes
		float ShapeRadius;									// Radius of the tubes
		float ShapeHyperScale;								// Hyper scale for HyperStreamlines

		/** Lighting Options */

		bool LightingEnable;								// Enable lighting
		float LightingAmbient;								// Ambient contribution
		float LightingDiffuse;								// Diffusion contribution
		float LightingSpecular;								// Specular contribution
		float LightingSpecularPower;						// Specular power

		/** Shadow Options */

		bool ShadowsEnable;									// Enable shadows
		float ShadowsAmbient;								// Ambient contribution
		float ShadowsDiffuse;								// Diffusion contribution
		float ShadowsWidth;									// Shadow line width

		/** Color LUT */

		vtkScalarsToColors * lut;							// LUT used for scalar coloring
		int lutIndex;										// Index of the LUT in the GUI combo box

		/** If "true", the visualization pipeline needs to be rebuilt, which
			is usually because one of the filters or the mapper has changed. */

		bool rebuildPipeline;

	private:

		/** Filter Pointers */

		vtkPolyDataMapper * mapper;								// Mapper
		vtkStreamlineToSimplifiedStreamline * simplifyFilter;	// Simplification filter
		vtkStreamlineToStreamGeneralCylinder * shapeFilter;		// Shape filter
		vtkMEVToStreamlineColor * colorFilterMEV;				// Color filter (MEV)
		vtkDirectionToStreamlineColor * colorFilterDirection;	// Color filter (Direction)
		vtkProbeFilter * colorFilterAI;							// Color filter (AI)
		UserOutput * userOut;									// User output (for progress bars)

		/** If "true", the mapper needs to be updated. Can only be changed 
			in "setupColorFilter" and "setupShapeFilter", as the mapper options
			and type can depend on the settings of these filters. */

		bool updateMapper;

}; // class FiberVisualizationPipeline


} // namespace bmia


#endif // bmia_FiberVisualizationPipeline_h