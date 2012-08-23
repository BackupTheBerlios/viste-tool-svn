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