/*
 * vtkStreamlineToSimplifiedStreamline.h
 *
 * 2005-10-17	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToSimplifiedStreamline of the DTITool.
 *
 * 2010-09-24	Evert van Aart
 * - First version for the DTITool3. Identical to old versions, save for some code revisions
 *	 and extra comments.
 *
 * 2010-10-19	Evert van Aart
 * - Added support for copying "vtkCellData" from input to output. This is needed because of a new coloring
 *   technique, which applies a single color to one whole fiber. Instead of storing the same color values
 *   for each fiber point, we store the color in the "vtkCellData" field, which is then applied to the
 *   whole fiber.
 *
 * 2011-02-01	Evert van Aart
 * - Added support for bypassing this filter.
 *
 */


/** ToDo List for "vtkStreamlineToSimplifiedStreamline"
	Last updated 24-09-2010 by Evert van Aart

	- In the "Execute" function, we should check whether the input data arrays
	  actually exist, and return an error otherwise.
*/


#ifndef bmia__vtkStreamlineToSimplifiedStreamline_H
#define bmia__vtkStreamlineToSimplifiedStreamline_H


/** Includes - VTK */

#include "vtkPolyDataToPolyDataFilter.h"
#include "vtkDataArray.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"
#include "vtkCellData.h"


namespace bmia {


/** This class implements a filter that simplifies a streamline (polyline),
	which has been generated using one of the fiber tracking methods, to a
	simplified streamline containing less points. The main processing
	variable is "StepLength", which determines the new distance between
	fiber points.
*/


class vtkStreamlineToSimplifiedStreamline : public vtkPolyDataToPolyDataFilter
{
	public:
		
		/** Constructor Call */

		static vtkStreamlineToSimplifiedStreamline *New();

		/** Set the length of the segments composing the fibers. */

		vtkSetClampMacro(StepLength, float, 0.0, VTK_LARGE_FLOAT);
	
		/** Get the length of the segments composing the fibers. */
	
		vtkGetMacro(StepLength, float);

		/** Specify whether or not we need to bypass this filter. 
			@param rB	New bypass setting. */

		void setDoBypass(bool rB)
		{
			doBypass = rB;
		}

	protected:
	
		/** Constructor */

		vtkStreamlineToSimplifiedStreamline();
			
		/** Destructor */
	
		~vtkStreamlineToSimplifiedStreamline() {}
	
		/** Points of the output polylines. */

		vtkPoints * newPts;

		/** Output polylines. */
	
		vtkCellArray * newFiberLine;	
	
		/** List of point IDs of the current polyline */
		vtkIdList * idFiberPoints;  

		/** List of output scalars of the current polyline. Only used 
			if the original polyline had scalars defined. */
	
		vtkDataArray * newScalars;  

		/** Length of the segments composing the fibers. */
	
		float StepLength;

		/** If true, the input if simply copied to the output. */

		bool doBypass;

		/** Update output given current input. */
	
		void Execute();

		/** Add new point to the output.
			@param pt	Point coordinates. */

		void createPoint(double * pt);

};


} // namespace bmia


#endif // bmia__vtkStreamlineToSimplifiedStreamline_H
