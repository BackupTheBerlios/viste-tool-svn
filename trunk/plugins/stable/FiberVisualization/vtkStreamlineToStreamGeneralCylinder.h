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
 * vtkStreamlineToStreamGeneralCylinder.h
 *
 * 2005-10-17	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToStreamGeneralCylinder of the DTITool.
 *
 * 2010-09-24	Evert van Aart
 * - First version for the DTITool3. Identical to old versions, save for some code revisions
 *	 and extra comments.
 *
 * 2011-01-19	Evert van Aart
 * - Improved ERROR checking and -handling.
 * - Added Qt progress bar.
 *
 * 2011-04-16	Evert van Aart
 * - Changed the Qt Progress bar to "updateProgress" calls.
 *
 */


#ifndef bmia__vtkStreamlineToStreamGeneralCylinder_H
#define bmia__vtkStreamlineToStreamGeneralCylinder_H


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include "vtkPolyDataToPolyDataFilter.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"
#include "vtkDataArray.h"
#include "vtkCellData.h"


namespace bmia {


/** This class implements a filter that converts a streamline (polyline),
	which has been generated using one the fiber tracking methods, to a 
	general cylinder arround this streamline. It inherits from "vtkPolyData-
	ToPolyDataFilter". Filters that create streamtubes and hyper streamlines
	inherit from this class.
*/


class vtkStreamlineToStreamGeneralCylinder : public vtkPolyDataToPolyDataFilter
{
	public:
	
		/** Constructor Call */
		static vtkStreamlineToStreamGeneralCylinder * New();

		/** VTK "Get" and "Set" macros for filter variables. */
		vtkSetClampMacro(NumberOfSides, int, 3, VTK_LARGE_INTEGER);
		vtkGetMacro(NumberOfSides, int);
		vtkSetClampMacro(Radius1, float, 0.0, VTK_LARGE_FLOAT);
		vtkSetClampMacro(Radius2, float, 0.0, VTK_LARGE_FLOAT);
		vtkGetMacro(Radius1, float);
		vtkGetMacro(Radius2, float);

		/** Set flag indicating if the scalars are used for scaling 
			the thickness of the cylinder.
			@param flag		Input bool value. */
	
		void useScalarsForThickness(bool flag);

		/** Set flag indicating if the scalars are used for coloring
			the output cylinder. 
			@param flag		Input bool value. */
	
		void useScalarsForColor(bool flag);

protected:

		/** Constructor */
	
		vtkStreamlineToStreamGeneralCylinder();
	
		/** Destructor */
	
		~vtkStreamlineToStreamGeneralCylinder() {}

		/** Points of the general cylinder streamlines. */
	
		vtkPoints * newPts;

		/** Normals of the points of the general cylinder streamlines. */
		
		vtkFloatArray * newNormals;

		/** Strips that define the polyedra of the general cylinder streamlines. */
	
		vtkCellArray * newStrips;

		/** Scalars of the points of the general cylinder streamline. */
	
		vtkDataArray * newScalars;

		/** Scalars of the points of the input fibers. */

		vtkDataArray * oldScalars;

		/** Use the scalars for scaling the thickness? */
		
		bool useScalarsForThicknessFlag;

		/** Use the scalars for colring the streamlines? */
	
		bool useScalarsForColorFlag;

		/** Number of sides of the general cylinder. */

		int NumberOfSides;

		/** Radius of the sections of the general cylinder (ellipse). */

		float Radius1;
		float Radius2;

		/** Number of components in the scalar value array */

		int numberOfScalarComponents;

		/** Cell Data of input and output, used to transfer per-fiber colors. */
		vtkCellData * outCD;
		vtkCellData * inCD;

		/** Update the output given the current input. */
		
		void Execute();

		/** Copy coordinate frames. 
			@param prevFrame	Destination frame.
			@param newFrame		Source frame. */
	
		void copyFrames(double prevFrame[3][3], double newFrame[3][3]);

		/** Creates and stores points and their normal on an ellipse centered
			on the input point. The orientation and size of the ellipse depend
			on the two input vectors and the input radii. The number of points
			create on this ellipse depends on the "NumberOfSides" variable. If
			needed, input scalar values are copied to the output.
			@param pt			Coordinates of central point.
			@param v1			First vector defining the ellipse.
			@param v2			Second vector defining the ellipse.
			@param r1			Radius along "v1".
			@param r2			Radius along "v2".
			@param pointId		ID of input point. */

		virtual void createPoints(double * pt, double * v1, double * v2, double r1, double r2, vtkIdType pointId);

		/** Create strips between two crossections of the general cylinder.
			@param initialId	Point ID of the first point. 
			@param idLine		ID of the current input line. */

		virtual void createPolygons(int initialId, vtkIdType idLine);

		/** Calculates the frame that is used to generate the points around
			the streamline, based on the frame of the previous point, the 
			current and previous fiber point, the vector between these points,
			and the scalar value in the current point. 
			@param prevFrame	Previous coordinate frame.
			@param newFrame		Compute coordinated frame.
			@param currentPoint	Current fiber point.
			@param prevPoint	Previous fiber point.
			@param lineSegment	Vector between previous and current point.
			@param pointId		ID of input point. */

		virtual void calculateCoordinateFramePlane(		double prevFrame[3][3], 
														double newFrame[3][3], 
														double * currentPoint,
														double * prevPoint, 
														double * lineSegment, 
														vtkIdType pointId			);

		/** Calculates the initial frame that is used to generate the points 
			around the streamline. In this case, we simply use "calculateCo-
			ordinateFramePlane", but the children of this class may reimplement
			this function if needed. 
			@param prevFrame	Previous coordinate frame.
			@param newFrame		Compute coordinated frame.
			@param currentPoint	Current fiber point.
			@param prevPoint	Previous fiber point.
			@param lineSegment	Vector between previous and current point.
			@param pointId		ID of input point. */

		virtual void calculateFirstCoordinateFramePlane(double prevFrame[3][3], 
														double newFrame[3][3], 
														double * currentPoint,
														double * prevPoint, 
														double * lineSegment, 
														vtkIdType pointId			);

};


} // namespace bmia


#endif // bmia__vtkStreamlineToStreamGeneralCylinder_H
