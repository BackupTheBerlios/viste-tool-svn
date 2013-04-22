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
 * vtkSHGlyphMapper.h
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
 */


#ifndef bmia_vtkSHGlyphMapper_h
#define bmia_vtkSHGlyphMapper_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes = Custom Files */

#include "vtkGlyphMapperVA.h"
#include "HWShading/vtkUniformSampler.h"
#include "HWShading/vtkMyShaderProgram.h"
#include "HWShading/vtkMyShaderProgramReader.h"
#include "HWShading/vtkUniformFloat.h"
#include "HWShading/vtkUniformInt.h"
#include "HWShading/vtkUniformBool.h"
#include "HARDI/HARDIMeasures.h"


/** Includes - VTK */

#include <vtkgl.h>
#include <vtkObjectFactory.h>
#include <vtkPointSet.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkMath.h>
#include <vtkAlgorithmOutput.h>
#include <vtkDoubleArray.h>
#include <vtkScalarsToColors.h>

/** Includes - Qt */

#include <QProgressDialog>

namespace bmia {


/** Mapper for GPU rendering of glyphs for spherical harmonics. Its inputs are:
	1) a "vtkImageData" object containing at least a scalar array with the 
	   spherical harmonic coefficients, and optionally an array containing 
	   the minimum and maximum radius for each glyphs.
	2) a set of seed points. which determines where the glyphs should be rendered.
*/

class vtkSHGlyphMapper : public vtkGlyphMapperVA
{
	public:
  
		/** Enumerator for coloring method. */

		enum SHColoringMethod
		{
			SHCM_Direction = 0,
			SHCM_Radius,
			SHCM_Measure
		};

		/** Constructor Call */

		static vtkSHGlyphMapper * New();

		/** Set the input Spherical Harmonics data.
			@param input	Input SH image data. */

		virtual void SetInput(vtkImageData * input);

		/** Set the input Spherical Harmonics data.
			@param input	Input SH data, stored as a "vtkDataSet". */

		virtual void SetInput(vtkDataSet * input);

		/** Set the input connection of the mapper. Used for construction
			of the rendering pipeline.
			@param input	Output port of VTK filter. */

		virtual void SetInputConnection(vtkAlgorithmOutput * input);

		/** Set/Get local scaling. With local scaling on, the glyphs will be normalized. */
  
		vtkSetMacro(LocalScaling, bool);
		vtkBooleanMacro(LocalScaling, bool);
		vtkGetMacro(LocalScaling, bool);

		/** Set/Get the step size for the linear search in the first part 
			of the ray casting. */
  
		void SetStepSize(float step);
		vtkGetMacro(StepSize, float); 

		/** Set the number of refinement steps. */

		void SetNumRefineSteps(int num);

		/** Rotate all glyphs around the Z-axis. The rotation functions are 
			for testing purposes only. */

		vtkSetMacro(ZRotationAngle, double);
		vtkGetMacro(ZRotationAngle, double);
		vtkSetMacro(YRotationAngle, double);
		vtkGetMacro(YRotationAngle, double);


		/** Set the coloring method for the glyphs.
			@param coloring		Desired coloring method. */
  
		void SetColoring(int coloring);
 
		/** Set the threshold for the radius.
			@param threshold	Desired threshold. */
  
		void SetRadiusThreshold(float threshold);

		/** Set whether or not to normalize the coefficients using
			the minimum and maximum radius. */
  
		vtkSetMacro(MinMaxNormalize, bool);
		vtkBooleanMacro(MinMaxNormalize, bool);

		/** Set whether or not to use spherical deconvolution fiber 
			ODF to sharpen the glyphs. */
  
		vtkSetMacro(FiberODF, bool);
		vtkBooleanMacro(FiberODF, bool);

		/** Set eigenvalues for fiber ODF. 
			@param v	First eigenvalue. */
  
		void SetEigenval1(double v)
		{
			this->Eigenval[0] = v;
		}
  
		/** Set eigenvalues for fiber ODF. 
			@param v	Second eigenvalue. */
  
		void SetEigenval2(double v)
		{
			this->Eigenval[1] = v;
		}

		/** Set/Get B-value (for fiber ODF). */
  
		vtkSetMacro(B, double);
		vtkGetMacro(B, double);

		/** Set the coloring measure.
			@param rMeasure		Required measure. */

		void setColoringMeasure(int rMeasure)
		{
			coloringMeasure = (HARDIMeasures::HARDIMeasureType) rMeasure;
		}

		/** Initialize the fiber ODF normalization method by computing and storing 
			the average values of the first and second DTI eigenvalues at the seed
			point locations. During execution, these values are used to normalize
			the glyphs. 
			@param eigen		Current eigensystem image. */

		void initODF(vtkImageData * eigen);

		/** Clear the scalar array, and set its pointer to NULL. This will cause the
			scalar to be recomputed during the next render pass. Can be called, for
			example, when the coloring options are changed. */

		void clearScalars()
		{
			if (scalarArray)
			{
				scalarArray->Delete();
				scalarArray = NULL;
			}
		}

	protected:
		
		/** Constructor */

		vtkSHGlyphMapper();

		/** Destructor */
  
		~vtkSHGlyphMapper();

		/** Read the shader program. */

		virtual void ReadShaderProgram();

		/** Configure the uniforms of the shader program. */
  
		virtual void SetupShaderUniforms();

		/** Draw glyphs at each seed point. */
  
		virtual void DrawPoints();

		/** Called when the input dataset is changed or updated. Recomputes
			maximum and mean values for the radius and the first coefficient. */
  
		virtual void InputUpdated();

		/** Compute the radius of the minimal bounding sphere around the 
			spherical harmonic with the given SH coefficients.
			@param coefficients	SH Coefficients. */
  
		double BoundingSphereRadius(double * coefficients);

		/** Rotate the spherical harmonics around the Z-axis. Rotation of zero-th,
			second and fourth order coefficients is done in separate functions. 
			@param coefficients	SH coefficients.
			@param angle		Rotation angle. */
  
		void RealWignerZRotation (double * coefficients, double angle);
		void RealWignerZRotation0(double * coefficients, double angle);
		void RealWignerZRotation2(double * coefficients, double angle);
		void RealWignerZRotation4(double * coefficients, double angle);

		/** Rotate the spherical harmonics around the Y-axis. Rotation of zero-th,
			second and fourth order coefficients is done in separate functions. 
			@param coefficients	SH coefficients.
			@param angle		Rotation angle. */
  
		void RealWignerYRotation (double * coefficients, double angle);
		void RealWignerYRotation0(double * coefficients, double angle);
		void RealWignerYRotation2(double * coefficients, double angle);
		void RealWignerYRotation4(double * coefficients, double angle);

		/** Manipulate the SH coefficients to scale, sharpen, rotate the glyph etc.
			@param coefficients	SH coefficients. */

		virtual void UpdateSHCoefficients(double * coefficients);

		/** Sharpen the glyph using the fiber ODF.
			@param coefficients	SH coefficients. */

		void DoFiberODF(double * coefficients);

		/** Normalize the glyphs using the local minimum and maximum radius. 
			@param coefficients	SH coefficients. */

		void DoMinMax(double * coefficients);

		/** Scale the glyphs. Also updates the minimum and maximum radius. 
			@param coefficients	SH coefficients. */
  
		void ScaleGlyphs(double * coefficients);

		/** Compute the measure scalar values for the current set of seed points. 
			Called once when rendering the glyphs for the first time after changing
			the input, the seed points, and/or the coloring options. 
			@param seeds		Seed points.
			@param image		Input image.
			@param coeffArray	Array with SH coefficients. */

		void computeScalars(vtkPointSet * seeds, vtkImageData * image, vtkDataArray * coeffArray);

		/** Global maximum and average values of the glyph radius and 
			the first SH coefficient ("A0"). */
  
		double MaxA0;
		double MaxRadius;
		double AverageA0;
		double AverageRadius;

		/** Should we scale the glyphs using local data? */

		bool LocalScaling;

		/** Should we sharper the glyphs using fiber ODF? */
  
		bool FiberODF;

		/** Should we normalize the glyphs using local minimum and maximum radius? */
  
		bool MinMaxNormalize;

		/** Ray casting parameters to be passed to the shaders. */
  
		float			  StepSize;
		vtkUniformFloat * UStepSize;
		vtkUniformInt	* UNumRefineSteps;
		vtkUniformInt	* UColoring;
		vtkUniformFloat	* URadiusThreshold;

		/** Rotation of glyphs around the Y- and Z-axis. */
  
		double ZRotationAngle;
		double YRotationAngle;

		/** B-value of the input SH data. Used for sharpening with the fiber ODF. */
  
		double B;

		/** Eigenvalues. Used for sharpening with the fiber ODF. */
  
		double Eigenval[2];

		/** Temporary array containing minimum and maximum radius for one voxel. */

		double * MinMax;

		/** Coloring method (based on direction, radius, or a selected HARDI measure. */
  
		SHColoringMethod ColoringMethod;

		/** Selected coloring measure. */

		HARDIMeasures::HARDIMeasureType coloringMeasure;

		/** Array used to contain measure values for the current set of seed points. */

		vtkDoubleArray * scalarArray;

}; // class vtkSHGlyphMapper


} // namespace bmia


#endif // bmia_vtkSHGlyphMapper_h
