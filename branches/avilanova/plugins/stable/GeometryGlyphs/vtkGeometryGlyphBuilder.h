/*
 * vtkGeometryGlyphBuilder.h
 *
 * 2011-04-11	Evert van Aart
 * - First version. 
 *
 * 2011-04-20	Evert van Aart
 * - Added support for changing the glyph type. 
 * 
 * 2011-05-04	Evert van Aart
 * - Increased support for coloring glyphs.
 *
 * 2011-05-09	Evert van Aart
 * - Made the "Execute" and "computeGeometry" functions virtual, for use with the
 *   derived class for building SH glyphs.
 * 
 * 2011-08-05	Evert van Aart
 * - Fixed a major error in the computation of the unit vectors.
 *
 */
 

#ifndef bmia_GeometryGlyphsPlugin_vtkGeometryGlyphBuilder_h
#define bmia_GeometryGlyphsPlugin_vtkGeometryGlyphBuilder_h


/** Includes - VTK */

#include <vtkPolyDataAlgorithm.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkMath.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkUnsignedCharArray.h>
#include <vtkCellData.h>

/** Includes - Qt */

#include <QMessageBox>


namespace bmia {


/** This filter builds the glyphs for a discrete sphere function. It has two inputs:
	1) a set of seed points, which specify the glyph positions, and 2) a volume
	containing the discrete sphere function. This volume should contain three 
	point data arrays: A) an N-component array with, for each voxel, the radii 
	for the N sphere directions; B) a 2-component array with N tuples, containing
	sets of angles (in rads) defining the N sphere directions (azimuth in the 
	range -pi to pi, and zenith in the range 0 to pi); C) a 3-component array
	of point indices, in which each set of three indices defines one triangle.
	Array C is optional; if it is not present, we draw lines (from the glyph
	center) instead of polygons. The arrays should be named "Vectors" (A), 
	"Spherical Directions" (B), and "Triangles" (C).

	To use the filter, first set the volume containing the discrete sphere
	function ("setInputVolume"), and then call "computeGeometry". The
	filter includes several options for normalization and sharpening of the glyphs. 
*/

class vtkGeometryGlyphBuilder : public vtkPolyDataAlgorithm
{
	public:

		/** Constructor Call */

		static vtkGeometryGlyphBuilder * New();

		/** VTK Macro */

		vtkTypeMacro(vtkGeometryGlyphBuilder, vtkPolyDataAlgorithm);

		/** Use the "Spherical Directions" array of the input image to pre-compute
			the geometrical template of the glyphs. For each set of input angles,
			a unit vector is constructed, so that the final radius of a glyph in
			a direction can be quickly computed by multiplying the unit vector by
			the local radius for this direction. The function also tries to fetch
			the "Triangles" array from the input volume. Note that you should
			call "setInputVolume" before calling this function. This 
			function only has to re-execute when the input volume changes, not 
			when the seed point change. 
			@param tessOrder	Tessellation order. Not used in this class,
								but may be used in derived classes. */

		virtual bool computeGeometry(int tessOrder = 0);

		/** Set a new input volume.
			@param image		New discrete sphere function volume. */

		void setInputVolume(vtkImageData * image);

		/** Method for normalization. */

		enum NormalizationMethod
		{
			NM_MinMax = 0,		/**< r = (r - min) / (max - min). */
			NM_Maximum,			/**< r = r / max. */
			NM_None				/**< No normalization. */
		};

		/** Scope for normalization, i.e., over which area or volume the maximum
			and minimum values used for normalization are computed. */

		enum NormalizationScope
		{
			NS_WholeImage = 0,	/**< Whole input image. */
			NS_SeedPoints,		/**< Current set of seed points. */
			NS_Local			/**< Current seed point. */
		};

		/** General shape of the glyphs. */

		enum GeometryGlyphType
		{
			GGT_Mesh = 0,		/**< 3D Mesh. Requires a triangulation array. */
			GGT_Star			/**< Start-shape (line to the center). */
		};

		/** Method for coloring the glyphs. */

		enum ColoringMethod
		{
			CM_Direction = 0,	/**< Direction-based coloring (XYZ - RGB). */
			CM_WDirection,		/**< Direction-based color with scalar weighting. */
			CM_Scalar,			/**< LUT-based coloring using a scalar volume. */
			CM_Radius			/**< LUT-based coloring using the radius. */
		};

		/** Set the normalization method.
			@param rMethod		Desired method. */

		void setNormalizationMethod(NormalizationMethod rMethod)
		{
			nMethod = rMethod;
		}

		/** Set the scope of the normalization. 
			@param rScope		Desired scope. */

		void setNormalizationScope(NormalizationScope rScope)
		{
			nScope = rScope;
		}

		/** Set the global scale of the glyphs. 
			@param rScale		Desired scale. */

		void setScale(double rScale)
		{
			scale = rScale;
		}
		
		/** Set the exponent used to sharpen the glyphs. 
			@param rPower		Desired exponent. */

		void setSharpeningExponent(double rPower)
		{
			sharpenExponent = rPower;
		}

		/** Enable or disable sharpening. 
			@param enable		Turn sharpening on or off. */

		void setEnableSharpening(bool enable)
		{
			enableSharpening = enable;
		}

		/** Set the coloring method.
			@param rMethod		Desired coloring method. */

		void setColoringMethod(ColoringMethod rMethod)
		{
			colorMethod = rMethod;
		}

		/** Set the glyph type. Return true if we successfully switched to the 
			required glyph type. If we could not switch, an error message is
			displayed, and false is returned. This can for example happen if
			we want to switch to 3D meshes, but the input data set does not
			have a triangles array.
			@param rType		Desired glyph type. */

		bool setGlyphType(GeometryGlyphType rType);

		/** Set the scalar volume used for coloring. 
			@param rVolume		Desired scalar volume. */

		void setScalarVolume(vtkImageData * rVolume);

		/** Set whether or not to normalize the scalars used for coloring (when
			using "CM_Scalar").
			@param rNorm		Desired setting (normalization on or off). */

		void setNormalizeScalars(bool rNorm)
		{
			normalizeScalars = rNorm;
		}

	protected:

		/** Constructor */

		vtkGeometryGlyphBuilder();

		/** Destructor */

		~vtkGeometryGlyphBuilder();

		/** Execute the filter. */

		virtual void Execute();

		/** Specify the information of the one input port of this filter. 
			Specifically, we say that the input should be of type "vtkPointSet".
			@param port		Input port index.
			@param info		Target input port information. */

		virtual int FillInputPortInformation(int port, vtkInformation * info);

		/** 2D array containing the unit vectors computed from the angles array
			of the current input volume. */

		double ** unitVectors;

		/** Number of spherical directions per glyph. */

		int numberOfAngles;

		/** Triangle array defining the topology of the glyphs. */

		vtkIntArray * trianglesArray;

		/** Input volume, containing an array defining the spherical directions,
			an array with the radius per direction per voxel, and (optionally)
			a triangles array defining the topology of the glyphs. */

		vtkImageData * inputVolume;

		/** Current normalization method. */

		NormalizationMethod nMethod;

		/** Current scope of normalization. */

		NormalizationScope nScope;

		/** Global scale of the glyphs. */

		double scale;

		/** Exponent used for sharpening the glyphs. */

		double sharpenExponent;

		/** Should we apply sharpening? */

		bool enableSharpening;

		/** Current glyph shape. */

		GeometryGlyphType glyphType;

		/** Glyph coloring method. */

		ColoringMethod colorMethod;

		/** Scalar volume used for coloring purposes. */

		vtkImageData * scalarVolume;

		/** If true, scalars will be normalized to the range 0-1 before being added
			to the output. Only relevant if the color method has been set to
			scalar-based coloring using LUTs ("CM_Scalar"). */

		bool normalizeScalars;

		/** Compute the normalization factors (i.e., the minimum and maximum values
			used to normalize the glyphs) for the whole image. In other words,
			find the smallest and largest radius of all voxels in the image.
			@param radii		Array containing all radii of the volume. 
			@param min			Output minimum.
			@param max			Output maximum. */

		void computeNormalizationFactorsForWholeImage(vtkDoubleArray * radii, double * min, double * max);

		/** Compute the normalization factors (i.e., the minimum and maximum values
			used to normalize the glyphs) for the current seed points. In other words,
			find the smallest and largest radius of all seed points.
			@param radii		Array containing all radii of the volume. 
			@param seeds		Current seed points.
			@param min			Output minimum.
			@param max			Output maximum. */

		void computeNormalizationFactorsForROI(vtkDoubleArray * radii, vtkPointSet * seeds, double * min, double * max);

		/** Compute the normalization factors (i.e., the minimum and maximum values
			used to normalize the glyphs) for the current point. In other words,
			find the smallest and largest radius of the current voxel.
			@param radii		Array containing all radii of the volume. 
			@param ptId			Index of the current voxel. 
			@param min			Output minimum.
			@param max			Output maximum. */

		void computeNormalizationFactorsForVoxel(vtkDoubleArray * radii, vtkIdType ptId, double * min, double * max);

}; // class vtkGeometryGlyphBuilder


} // namespace bmia


#endif // bmia_GeometryGlyphsPlugin_vtkGeometryGlyphBuilder_h