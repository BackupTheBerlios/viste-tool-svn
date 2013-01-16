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
 * vtkMEVToStreamlineColor.h
 *
 * 2010-10-05	Evert van Aart
 * - First Version.
 *
 * 2010-10-11	Evert van Aart
 * - Changed "abs" to "fabs"; now works under Linux.
 * 
 */


/** ToDo List for "vtkMEVToStreamlineColor"
	Last updated 05-10-2010 by Evert van Aart

	- Should we interpolate DTI tensors, instead of their eigenvectors?
	  Or is eigenvector interpolation accurate enough for visualization
	  purposes?
    - Add progress meter of sorts.
*/


#ifndef bmia__vtkMEVToStreamlineColor_H
#define bmia__vtkMEVToStreamlineColor_H


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include "vtkImageData.h"
#include "vtkPolyDataToPolyDataFilter.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkPoints.h"
#include "vtkFloatArray.h"
#include "vtkCell.h"
#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkPointData.h"


namespace bmia {


/** Converts the Main Eigenvector (MEV) of the DTI tensors along a computed
	fiber to RGB values, which can then be used to color the fibers. RGB
	values are stored as unsigned characters in the "scalars" array of
	the output.
*/

class vtkMEVToStreamlineColor : public vtkPolyDataToPolyDataFilter
{
	public:
	
		/** Constructor Call */

		static vtkMEVToStreamlineColor * New();

		/** Set whether or not the MEV values should be shifted.
			@param rShiftValues		New value. */

		void setShiftValues(bool rShiftValues)
		{
			ShiftValues = rShiftValues;
		}

		/** Set whether or not we should weigh the colors using AI values.
			@param rShiftValues		New value. */

		void setUseAIWeighting(bool rAIWeighting)
		{
			UseAIWeighting = rAIWeighting;
		}

		/** Set the pointer of the AI image. If "UseAIWeighting" is set to 
			"true", we use the values of this image to weigh the RGB values. 
			@param rAI				New image pointer. */

		void setAIImageData(vtkImageData * rAI)
		{
			aiImageData = rAI;
		}

		/** Set the pointer of the eigensystem image. This image contains the
			three eigenvectors and three eigenvalues for each grid point. We only
			use the main eigenvector in this filter. 
			@param rEigen				New image pointer. */

		void setEigenImageData(vtkImageData * rEigen)
		{
			eigenImageData = rEigen;
		}

	protected:

		/** Constructor */

		vtkMEVToStreamlineColor();

		/** Destructor */

		~vtkMEVToStreamlineColor();

		/** Main entry point for the execution of the filter. */

		void Execute();

		/** Image pointers for AI values and eigensystem data. */

		vtkImageData * aiImageData;
		vtkImageData * eigenImageData;

		/** Array containing the main eigenvector of each point. */

		vtkFloatArray * mainEigenVectors;

		/** Array containing the AI scalar values. */

		vtkDataArray * aiScalars;

		/** Arrays containing data of the eight voxels (cell) surrounding
			the current fiber point, used for interpolation. */

		vtkFloatArray * cellMEV;
		vtkDataArray  * cellAI;

	private:

		/** Interpolate the main eigenvector, using the contents of "cellMEV"
			and the interpolation weights obtained from the "FindCell" function.
			@param output		Interpolated eigenvector.
			@param weights		Interpolation weights. */

		void interpolateMEV(double * output, double * weights);

		/** Interpolate the AI scalar value, using the contents of "cellAI"
			and the interpolation weights obtained from the "FindCell" function.
			@param output		Interpolated AI value.
			@param weights		Interpolation weights. */

		void interpolateAI(double * output, double * weights);

		/** If "true", main eigenvector components are mapped from the range
			[-1, 1] to the range [0, 1] by dividing them by two and adding 0.5.
			Otherwise, the absolute values are used. Default value is "false". */

		bool ShiftValues;

		/** If "true", RBG values are weighted (through multiplication) by the
			Anisotropy Index value at the current point. */

		bool UseAIWeighting;

}; // class vtkMEVToStreamlineColor


} // namespace bmia


#endif // bmia__vtkMEVToStreamlineColor_H