/*
 * vtkStreamlineToHyperStreamPrisma.h
 *
 * 2005-11-07	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToHyperStreamPrisma of the DTITool.
 *
 * 2010-09-29	Evert van Aart
 * - First version for the DTITool3. Added some minor code revisions, extended comments. 
 *
 * 2010-10-25	Evert van Aart
 * - Added support for copying "vtkCellData" from input to output. This is needed because of a new coloring
 *   technique, which applies a single color to one whole fiber. Instead of storing the same color values
 *   for each fiber point, we store the color in the "vtkCellData" field, which is then applied to the
 *   whole fiber.
 *
 */


#ifndef bmia__vtkStreamlineToHyperStreamPrisma_H
#define bmia__vtkStreamlineToHyperStreamPrisma_H


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "vtkStreamlineToHyperStreamline.h"
#include "vtkStreamlineToStreamTube.h"

/** Includes - VTK */

#include "vtkImageData.h"
#include "vtkDataArray.h"
#include "vtkObjectFactory.h"
#include "vtkCell.h"
#include "vtkDataArray.h"
#include "vtkPointData.h"
#include "vtkMath.h"

/** This class implements a filter that converts a streamline (polyline), which has
	been computed using one of the fiber tracking methods, to a general prism with 
	four edges, based on the tensor eigenvectors and eigenvalues. There are two 
	differences between this class and "vtkStreamlineToHyperStreamline": The number
	of sides is constant, and it can constuct ribbons instead of tubes. 
*/


namespace bmia {


class vtkStreamlineToHyperStreamPrisma : public vtkStreamlineToHyperStreamline
{
	public:

		/** Constructor Call */

		static vtkStreamlineToHyperStreamPrisma * New();

		/** We do nothing in this case, since we always use four sides.
			@param sides	Required number of sides; not used. */
		
		void SetNumberOfSides(int sides)
		{

		}

		/** Set/Get the type of output, which is either a general prisma 
			(if the boolean values is "true"), or crossing ribbons (if 
			the boolean is "false". */
		
		vtkSetMacro(TubeNotRibbons, bool);
		vtkGetMacro(TubeNotRibbons, bool);

	protected:
		
		/** Constructor */

		vtkStreamlineToHyperStreamPrisma();

		/** Destructor */

		~vtkStreamlineToHyperStreamPrisma() {}

		/** Calculate the polygons as ribbons perpendicular to the streamline 
			or as a prisma, depending on the value of "tubeNotRibbons".
			@param initialId	Array index for first point of the polygon. 
			@param idLine		ID of the current input line. */

		virtual void createPolygons(int initialId, vtkIdType idLine);

		/** Reimplementation of the function that creates the points of the 
			selected 3D shape around the streamline.
			@param pt			Coordinates of central point.
			@param v1			First vector defining the ellipse.
			@param v2			Second vector defining the ellipse.
			@param r1			Radius along "v1".
			@param r2			Radius along "v2".
			@param pointId		ID of current input point. */
	
		virtual void createPoints(double * pt, double * v1, double * v2, double r1, double r2, vtkIdType pointId);
	
		/** The type of output, which is either a general prisma 
			(if the boolean values is "true"), or crossing ribbons (if 
			the boolean is "false". */
	
		bool TubeNotRibbons;

		/** Copy the scalar values of an input point to an output point.
			Used to pass the color of the fiber, which has been computed in
			a previous filter, to the 3D output of this filter.
			@param pointIdIn	Point ID of input point.
			@param pointIdOut	Point ID of output point. */

		void copyScalarValues(vtkIdType pointIdIn, vtkIdType pointIdOut);

}; // vtkStreamlineToHyperStreamPrisma


} // namespace bmia


#endif // bmia__vtkStreamlineToHyperStreamPrisma_H
