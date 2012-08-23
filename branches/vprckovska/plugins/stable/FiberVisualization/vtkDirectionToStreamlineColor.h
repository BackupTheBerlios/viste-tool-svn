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

    - Add progress meter of sorts.
*/


#ifndef bmia__vtkDirectionToStreamlineColor_H
#define bmia__vtkDirectionToStreamlineColor_H


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


/** Converts the local direction vector of a computed fiber to RGB 
	values, which can then be used to color the fibers. RGB	values 
	are stored as unsigned characters in the "scalars" array of	the 
	output. This class differs from "vtkMEVToStreamlineColor" in the
	sense that not the MEV, but the actual direction is used for the
	output color. This allows for more consistent visualization of 
	the direction when the actual direction differs from the local MEV.
*/

class vtkDirectionToStreamlineColor : public vtkPolyDataToPolyDataFilter
{
	public:
	
		/** Constructor Call */

		static vtkDirectionToStreamlineColor * New();

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

	protected:

		/** Constructor */

		vtkDirectionToStreamlineColor();

		/** Destructor */

		~vtkDirectionToStreamlineColor();

		/** Main entry point for the execution of the filter. */

		void Execute();

		/** Image pointer for AI values. */

		vtkImageData * aiImageData;

		/** Array containing the AI scalar values. */

		vtkDataArray * aiScalars;

		/** Arrays containing AI data of the eight voxels (cell) surrounding
			the current fiber point, used for interpolation. */

		vtkDataArray * cellAI;

	private:

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

}; // class vtkDirectionToStreamlineColor


} // namespace bmia


#endif // bmia__vtkDirectionToStreamlineColor_H