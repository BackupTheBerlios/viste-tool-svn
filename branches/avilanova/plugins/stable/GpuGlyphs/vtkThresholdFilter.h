/*
 * vtkThresholdFilter.h
 *
 * 2011-01-10	Evert van Aart
 * - First version
 *
 */


#ifndef bmia_vtkThresholdFilter_h
#define bmia_vtkThresholdFilter_h


/** Includes - VTK */

#include <vtkPointSetToPointSetFilter.h>
#include <vtkPointSet.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkObjectFactory.h>

/** Includes - Custom Files */

#include "HARDI/HARDIMeasures.h"


namespace bmia {


/** This simple filter has three main inputs: 1) a set of seed points, 2) a scalar threshold
	value, and 3) an image containing Spherical Harmonics. Upon execution, it creates two output 
	point sets: one containing all input seed points for which the HARDI measure, computed from
	the SH image, is less than the threshold, and one with all other seed points. The "invert" 
	variable can be used to exchange these two output data sets (i.e., the first one will contain 
	seed points with scalar values LARGER than the threshold). Used for fused DTI-HARDI glyphs 
	visualization. 
*/

class vtkThresholdFilter : public vtkPointSetToPointSetFilter
{
	public:

		/** Force the filter to execute. Not really a nice way to do this, but the VTK pipeline
			was not cooperating, and this works fine. All it does is call "Execute". */

		void forceExecute();

		/** VTK Macro */

		vtkTypeMacro(vtkThresholdFilter, vtkPointSetToPointSetFilter);

		/** Constructor Call */

		static vtkThresholdFilter * New();

		/** Set the threshold.
			@param rT			Desired threshold. */

		void setThreshold(double rT)
		{
			threshold = rT;
		}

		/** Set the desired HARDI measure. 
			@param rMeasure		Desired HARDI measure. */

		void setMeasure(HARDIMeasures::HARDIMeasureType rMeasure)
		{
			measure = rMeasure;
		}

		/** Set the Spherical Harmonics image. 
			@param rSH			Desired SH image. */

		void setSHImage(vtkImageData * rSH)
		{
			shData = rSH;
		}

		/** Set the invert rule.
			@param rInvert		Desired value for "invert". */

		void setInvert(bool rInvert)
		{
			invert = rInvert;
		}

		/** Set the outputs of the filter. These are created outside of	this filter. 
			@param dtiSeeds		Output seed points for DTI glyphs.
			@param hardiSeeds	Output seed points for HARDI glyphs. */

		void setOutputs(vtkPointSet * dtiSeeds, vtkPointSet * hardiSeeds);

	protected:

		/** Constructor */

		vtkThresholdFilter();

		/** Destructor */

		~vtkThresholdFilter();

		/** Main entry point for the execution of the filter. */

		void Execute();

		/** Measure used to classify the seed points. */

		HARDIMeasures::HARDIMeasureType measure;

		/** Threshold value; Computed HARDI measures are compared against this value. */

		double threshold;

		/** Spherical Harmonics image, used to compute the HARDI measures. */

		vtkImageData * shData;

		/** If false (default), seeds points for which the HARDI measure is less than the
			threshold are sent to the DTI seed point set, and seed points for which the measure
			is equal to or larger than the threshold are sent to the HARDI seed point set. If
			true, this is reversed, so large measure values are used for DTI, and small ones are
			used for HARDI. This can for example be useful for the "R0" measure, which measure
			isotropy rather than anisotropy. */

		bool invert;
};


} // namespace bmia


#endif // bmia_vtkThresholdFilter_h