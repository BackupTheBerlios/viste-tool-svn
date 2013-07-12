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
 * vtkSphericalHarmonicsToScalarVolumeFilter.h
 *
 * 2011-04-29	Evert van Aart
 * - First version.
 *
 * 2011-05-04	Evert van Aart
 * - Added the volume measure.
 *
 * 2011-08-05	Evert van Aart
 * - Fixed an error in the computation of the unit vectors.
 *
 */


#ifndef bmia_HARDIMeasures_vtkSphericalHarmonicsToScalarVolumeFilter_h
#define bmia_HARDIMeasures_vtkSphericalHarmonicsToScalarVolumeFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkMath.h>

/** Includes - Qt */

#include <QString>
/** Includes - Custom Files */

#include "HARDI/HARDIMeasures.h"

namespace bmia {


/** This filter computes a scalar measure from a discrete sphere function. Its input
	is a volume containing at least an array defining the spherical angles of the
	sample points (see "anglesArray"), and an array containing the radius for each
	sample point per voxel (see "radiiArray"). For some measures, a triangle array
	defining the topology of the sample points is required (see "triangleArray").
	The output is a scalar volume of doubles with one value per voxel. 
*/

class vtkSphericalHarmonicsToScalarVolumeFilter : public vtkSimpleImageToImageFilter
{
	public:

		/** Constructor Call */

		static vtkSphericalHarmonicsToScalarVolumeFilter * New();

		/** Set the triangles array, which defines the topology of the discrete
			sphere function. 
			@param rTriangles	Desired triangle array. */

		void setTrianglesArray(vtkIntArray * rTriangles)
		{
			trianglesArray = rTriangles;
		}

		/** Enumeration of all scalar measures supported by this filter. New measures
			should be added IN FRONT OF "DSPHM_NumberOfMeasures". */

		enum Measure
		{
			DSPHM_SurfaceArea = 0,		/**< Surface area of the 3D glyphs. */
			DSPHM_Volume,				/**< Volume of the 3D glyphs. */
			DSPHM_Average,				/**< Average radius. */
			DSPHM_NumberOfMeasures		/**< Number of measures. Should always be the last item. */
		};

		/** Return the short measure name for the specified scalar measure.
			@param index		Index of the desired measure. */

		QString getShortMeasureName(int index);

		/** Return the long measure name for the specified scalar measure.
			@param index		Index of the desired measure. */

		QString getLongMeasureName(int index);

		/** Set the current measure. If this function is not called before the 
			filter is executed, "DSPHM_SurfaceArea" will be used. 
			@param m			Desired scalar measure. */

		void setCurrentMeasure(int m)
		{
			currentMeasure = (Measure) m;
		}
			/** Measure used to classify the seed points. */

		HARDIMeasures::HARDIMeasureType measure;

	protected:

		/** Constructor. */

		vtkSphericalHarmonicsToScalarVolumeFilter();

		/** Destructor. */

		~vtkSphericalHarmonicsToScalarVolumeFilter();

		/** Execute the filter. Takes a volume containing a discrete sphere function
			as input, and uses it to generate a volume containing a scalar measure.
			Said measure can then be visualized using for example the Plane Visualization
			plugin or the Ray Cast plugin. Called by the VTK pipeline.
			@param input		Input volume (Discrete sphere function). 
			@param output		Output volume (Scalar measure). */

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

	private:

		/** Selected scalar measure. */

		Measure currentMeasure;

		/** Array of triangles defining the topology of the 3D glyphs. The array 
			contains tuples of three integers each; the integers represent indices
			of the sample points of the discrete sphere function. Thus, each tuple
			defines a triangle between three sample points. Note that not all measures
			require this triangle array. */

		vtkIntArray * trianglesArray;

		/** Array of angles. Each set of angles consists of a zenith angle in the
			range [0 - pi] (from the positive Z-axis), and an azimuth angle in the
			range [-pi - pi] (from the positive X-axis). */

		vtkDoubleArray * anglesArray;

		/** Array containing, for each voxel in the volume, a vector of radii for
			the sample points of the discrete sphere function. The angles of these
			sample points (which are the same for all voxels) are stored in the
			angle array; the topological interconnectedness of the points is defined
			in the triangles array. */

		vtkDoubleArray * radiiArray;

		/** 2D Array containing unit vectors, computed from the angles array. The
			array contains one row per sample point, and each row contains the 3D
			vector defining the corresponding point on the unit sphere. */

		double ** unitVectors;

		/** Step size for the progress bar. The progress bar is only updated after
			every "progressStepSize" points. */

		int progressStepSize;

		/** Compute the unit vectors from the angles array. Return true on success,
			and false otherwise. */

		bool computeUnitVectors();

		/** Compute the surface area of the 3D glyphs, which can be used as a scalar
			measure ("DSPHM_SurfaceArea"). Requires a triangle array. 
			@param outArray		Output scalar array. */

		bool computeSurfaceArea(vtkDoubleArray * outArray);

		/** Compute the volume of the 3D glyphs, which can be used as a scalar
			measure ("DSPHM_Volume"). Requires a triangle array. 
			@param outArray		Output scalar array. */

		bool computeVolume(vtkDoubleArray * outArray);

		/** Compute the average radius for each voxel, which can be used as a scalar
			measure ("DSPHM_Average"). Does not require a triangle array. */

		bool computeAverageRadius(vtkDoubleArray * outArray);

}; // vtkSphericalHarmonicsToScalarVolumeFilter


} // namespace bmia


#endif // bmia_HARDIMeasures_vtkSphericalHarmonicsToScalarVolumeFilter_h
