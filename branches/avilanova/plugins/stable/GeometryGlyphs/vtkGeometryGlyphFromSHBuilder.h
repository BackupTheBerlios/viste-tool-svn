/*
 * vtkGeometryGlyphFromSHBuilder.h
 *
 * 2011-05-09	Evert van Aart
 * - First version. 
 * 
 * 2011-05-10	Evert van Aart
 * - Fixed a memory allocation bug.
 * 
 */
 

#ifndef bmia_GeometryGlyphsPlugin_vtkGeometryGlyphFromSHBuilder_h
#define bmia_GeometryGlyphsPlugin_vtkGeometryGlyphFromSHBuilder_h


/** Includes - VTK */

#include <vtkPolyDataAlgorithm.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkMath.h>

/** Includes - Custom Files */

#include "vtkGeometryGlyphBuilder.h"
#include "HARDI/sphereTesselator.h"
#include "HARDI/SphereTriangulator.h"
#include "HARDI/HARDIMath.h"
#include "HARDI/HARDITransformationManager.h"

/** Includes - C++ */

#include <vector>
#include <malloc.h>


namespace bmia {


/** Class for building geometry glyphs from spherical harmonics. The parameters
	are largely similar to those of "vtkGeometryGlyphBuilder", which is the base
	class of this filter. The only two functions that differ are "computeGeometry",
	which constructs a tessellated sphere, and "Execute", which uses this sphere
	in combination with the SH coefficients to construct the glyphs. SH data of
	up to the eight order is supported. The input volume should have a scalar array
	with 1, 6, 15, 28 or 25 components.
*/

class vtkGeometryGlyphFromSHBuilder : protected vtkGeometryGlyphBuilder
{
	public:

		/** Constructor Call */

		static vtkGeometryGlyphFromSHBuilder * New();

		/** VTK Macro */

		vtkTypeMacro(vtkGeometryGlyphFromSHBuilder, vtkPolyDataAlgorithm);

		/** Compute the geometry of the glyphs. In this case, this is done by 
			tessellating a sphere with the specified tessellation order. The
			points of this unit sphere are stored in the "unitVectors" array, and
			their angles in spherical coordinates are stored in the "anglesArray"
			vector. The topology of the constructed sphere (i.e., it triangles)
			is stored in the "trianglesArray" array. 
			@param tessOrder	Tessellation order. */

		virtual bool computeGeometry(int tessOrder = 3);

	protected:

		/** Constructor. */

		vtkGeometryGlyphFromSHBuilder();

		/** Destructor. */

		~vtkGeometryGlyphFromSHBuilder();

		/** Execute the filter. */

		virtual void Execute();

	private:

		/** Array containing the angles, in spherical coordinates, of the points
			of the tessellated sphere. Note that, while the parent class uses
			zenith-azimuth spherical coordinates, this class uses 'regular'
			spherical coordinates, the difference being that the first angle
			is the elevation from the XY plane, rather than the angle from the
			positive Z-axis. This was done because the HARDI math functions
			expect spherical coordinates to use this system. */

		std::vector<double *> anglesArray;

		/** Compute the minimum and maximum radius for one glyph. If the normalization
			method is set to "None", the range 0-1 will be returned.
			@param radii	List of radii for one vector. 
			@param rMin		Output minimum radius.
			@param rMax		Output maximum radius. */

		void computeMinMaxRadii(std::vector<double> * radii, double & rMin, double & rMax);

}; // vtkGeometryGlyphFromSHBuilder


} // namespace bmia


#endif // bmia_GeometryGlyphsPlugin_vtkGeometryGlyphFromSHBuilder_h