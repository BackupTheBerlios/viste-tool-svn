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
 * vtkGenericConnectivityMeasureFilter.h
 *
 * 2011-05-12	Evert van Aart
 * - First version.
 *
 * 2011-08-22	Evert van Aart
 * - Added a progress bar.
 *
 */


#ifndef bmia_ConnectivityMeasurePlugin_vtkGenericConnectivityMeasureFilter_h
#define bmia_ConnectivityMeasurePlugin_vtkGenericConnectivityMeasureFilter_h


/** Includes - VTK */

#include <vtkPolyDataToPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkObjectFactory.h>
#include <vtkDataArray.h>
#include <vtkCellArray.h>
#include <vtkMath.h>


namespace bmia {


class vtkGenericConnectivityMeasureFilter : public vtkPolyDataToPolyDataFilter
{
	public:

		/** Constructor Call */

		static vtkGenericConnectivityMeasureFilter * New();

		/** VTK Macro */

		vtkTypeMacro(vtkGenericConnectivityMeasureFilter, vtkPolyDataToPolyDataFilter);

		/** Store the auxiliary image.
			@param rImage		Desired input image. */

		void setAuxImage(vtkImageData * rImage)
		{
			auxImage = rImage;
		}

		/** Turn normalization on or off. 
			@param rNormalize	Desired normalization setting. */

		void setNormalize(bool rNormalize)
		{
			normalize = rNormalize;
		}

	protected:

		/** Auxiliary image, used for data-dependent connectivity measures. */

		vtkImageData * auxImage;

		/** Scalar array of the auxiliary image. */

		vtkDataArray * auxScalars;

		/** If true, normalize the computed connectivity measure values to the range 0-1. */

		bool normalize;

		/** Previous fiber point. */

		double previousPoint[3];

		/** Current fiber point. */

		double currentPoint[3];

		/** Each subclass should set this boolean to true or false in its constructor,
			depending on whether or not it needs an auxiliary image to compute the
			connectivity measure (i.e., whether or not the measure is data-dependent). */

		bool auxImageIsRequired;

		/** Array for the output connectivity measure values. */

		vtkDoubleArray * outScalars;

		/** Entry point for the filter. The default implementation of this filter
			checks the input and output polydata objects, and the auxiliary image
			(through "initAuxImage") if "auxImageIsRequired" is true. Next, it loops
			through all fibers and all points, calling "updateConnectivityMeasure"
			for each point. For most connectivity measures, this should be enough 
			to be able to compute the measure; if not, feel free to overwrite this
			function in your own subclass. */

		virtual void Execute();

		/** Computes the connectivity measure for one point. Should be implemented
			by the subclasses. In this function, subclasses should update the
			previous and current points (i.e., copy "currentPoint" to "previousPoint"
			and read the new "currentPoint" coordinaties using "fiberPointId"), read
			data from the auxiliary image (if required), and use this to compute a
			measure value, which is should then write to the "outScalars" array at
			position "fiberPointId". See "vtkGeodesicConnectionStrength" for an
			example implementation. */

		virtual void updateConnectivityMeasure(vtkIdType fiberPointId, int pointNo);

		/** Initialize the auxiliary image, which consists of setting the pointer
			"auxScalars" and optionally checking if the auxiliary image data is 
			correct. By default, we load the "Scalars" array, and we do not perform
			any additional checking. An example of an additional check may be checking
			if a DTI tensor array has six components. If a test fails, false is returned,
			which will stop the execution of the filter. */

		virtual bool initAuxImage();

		/** Constructor. */

		vtkGenericConnectivityMeasureFilter();

		/** Destructor. */

		~vtkGenericConnectivityMeasureFilter();

}; // class vtkGenericConnectivityMeasureFilter


} // namespace bmia


#endif // bmia_ConnectivityMeasurePlugin_vtkGenericConnectivityMeasureFilter_h