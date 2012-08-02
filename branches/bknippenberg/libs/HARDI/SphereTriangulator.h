/*
 * SphereTriangulator.h
 *
 * 2011-04-11	Evert van Aart
 * - First version. Currently uses the "vtkDelaunay3D" filter, which seems to
 *   produce a good triangulation. If a more sophisticated algorithm is needed
 *   further down the road, it can be added to this class. 
 * 
 */
 

#ifndef bmia_SphereTriangulator_h
#define bmia_SphereTriangulator_h


/** Includes - VTK */

#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkMath.h>
#include <vtkDelaunay3D.h>
#include <vtkDelaunay2D.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkDataSetSurfaceFilter.h>


namespace bmia {


/** Class for computing the triangulation of a set of points on a sphere. The input
	spheres should be located on the UNIT sphere, since the algorithms used cannot
	deal with concave hulls. Several different input formats are supported. The output
	will be an integer array with three components and one tuple per triangle. The
	three integers per tuple represent the point IDs of the one triangle. 
*/

class SphereTriangulator
{
	public: 
		
		/** Constructor. */

		SphereTriangulator();

		/** Destructor. */

		~SphereTriangulator();

		/** Compute triangulation from an array of sets of angles defining the
			spherical directions. Each set of angles consists of an azimuth angle
			(-pi to pi), and a zenith angle (0 to pi), both expressed in radians.
			The output is a set of 3-component tuples defining the triangles. 
			Returns true on success. Internally, this function converts the angle
			array to a set of unit vectors, and subsequently calls the 
			"triangulateFromUnitVectors" function. 
			@param anglesArray	Array of sets of angles.
			@param triangles	Output triangle array. */

		bool triangulateFromAnglesArray(vtkDoubleArray * anglesArray, vtkIntArray * triangles);

		/** Compute triangulation from a set of points defining points on the
			unit sphere. The output is a set of 3-component tuples defining the 
			triangles. Returns true on success.
			@param unitVectors	Array of point coordinates on the unit sphere. 
			@param triangles	Output triangle array. */

		bool triangulateFromUnitVectors(vtkPoints * unitVectors, vtkIntArray * triangles);
};


} // namespace bmia


#endif // bmia_SphereTriangulator_h
