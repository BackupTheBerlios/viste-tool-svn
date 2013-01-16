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
 * vtkStreamlineToStreamTube.h
 *
 * 2005-10-17	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToStreamTube of the DTITool.
 *
 * 2010-09-23	Evert van Aart
 * - First version for the DTITool3. Identical to old versions, save for some code revisions
 *	 and extra comments.
 *
 */


#ifndef bmia__vtkStreamlineToStreamTube_H
#define bmia__vtkStreamlineToStreamTube_H


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include "vtkPolyDataToPolyDataFilter.h"
#include "vtkObjectFactory.h"


/** Includes - Custom Files */

#include "vtkStreamlineToStreamGeneralCylinder.h"


namespace bmia {


/** This class implements a filter that converts a streamline (polyline), computed 
	by one of the fiber tracking methods, to a tubular shape arround this streamline. 
	It inherits from "vtkStreamlineToStreamGeneralCylinder". In this class, "radius1"
	will always be equal to "radius2".
*/


class vtkStreamlineToStreamTube : public vtkStreamlineToStreamGeneralCylinder
{
	public:
	
		/** Constructor Call */
		static vtkStreamlineToStreamTube * New();

		/** Set the radius of the tube.
			@param newRadius	Required radius. */
	
		void SetRadius (float newRadius);

		/** Get the radius of the tube. */
	
		float GetRadius ()
		{
			return this->Radius;
		};

	protected:
		
		/** Constructor */
	
		vtkStreamlineToStreamTube();

		/** Destructor */

		~vtkStreamlineToStreamTube() {}

		/** Calculates the frame that is used to generate the points around 
			the streamline. It uses the default implementation of this function
			from "vtkStreamlineToStreamGeneralCylinder", after which it multiplies
			both radii by the "scalar" value. 
			@param prevFrame	Previous coordinate frame.
			@param newFrame		Compute coordinated frame.
			@param currentPoint	Current fiber point.
			@param prevPoint	Previous fiber point.
			@param lineSegment	Vector between previous and current point.
			@param pointId		ID of current input point. */
	
		virtual void calculateCoordinateFramePlane(		double prevFrame[3][3], 
														double newFrame[3][3], 
														double * currentPoint,
														double * prevPoint, 
														double * lineSegment, 
														vtkIdType pointId			);

		/** Calculates the initial frame that is used to generate the points around 
			the streamline. It uses the default implementation of this function
			from "vtkStreamlineToStreamGeneralCylinder", after which it multiplies
			both radii by the "scalar" value. 
			@param prevFrame	Previous coordinate frame.
			@param newFrame		Compute coordinated frame.
			@param currentPoint	Current fiber point.
			@param prevPoint	Previous fiber point.
			@param lineSegment	Vector between previous and current point.
			@param pointId		ID of current input point. */

		virtual void calculateFirstCoordinateFramePlane(double prevFrame[3][3], 
														double newFrame[3][3], 
														double * currentPoint,
														double * prevPoint, 
														double * lineSegment, 
														vtkIdType pointId			);

		/** Radius of the tube. */
	
		float Radius;
};


} // namespace bmia


#endif // bmia__vtkStreamlineToStreamTube_H