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
 * vtkGeodesicConnectionStrengthFilter.h
 *
 * 2011-05-12	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_ConnectivityMeasurePlugin_vtkGeodesicConnectionStrengthFilter_h
#define bmia_ConnectivityMeasurePlugin_vtkGeodesicConnectionStrengthFilter_h


/** Includes - Custom Files */

#include "vtkGenericConnectivityMeasureFilter.h"
#include "TensorMath/vtkTensorMath.h"

/** Includes - VTK */

#include <vtkCell.h>


namespace bmia {


/** This class computes the Geodesic Connection Strength, which is a connectivity
	measure for DTI fibers. This measure is computed as a fraction: The nominator
	contains the euclidean length of the fiber from its start to the current point,
	while the denominator contains the geodesic length. The geodesic length of 
	line segment "d" is computed as "d * G * d^T", where "G" is the inverse DTI tensor.
*/

class vtkGeodesicConnectionStrengthFilter : public vtkGenericConnectivityMeasureFilter
{
	public:

		/** Constructor Call */

		static vtkGeodesicConnectionStrengthFilter * New();

		/** VTK Macro */

		vtkTypeMacro(vtkGeodesicConnectionStrengthFilter, vtkGenericConnectivityMeasureFilter);

	protected:

		/** Update the connectivity measure for the current point. In this case,
			we increment the euclidean distance (nominator) and the geodesic distance
			(denominator), and then compute the faction to get the measure. Since we
			cannot compute this value for the first fiber point, we copy the value
			of the second point to the first point once we've computed it.
			@param fiberPointId	ID of the current point in the input fiber set.
			@param pointNo		Sequence number of the point in the current fiber. */

		virtual void updateConnectivityMeasure(vtkIdType fiberPointId, int pointNo);

		/** Initialize and check the auxiliary image. In this case, we're using
			a DTI image, so we get its "Tensors" array, and check if it contains
			six components (six unique tensor elements). */

		virtual bool initAuxImage();

		/** Constructor. */

		vtkGeodesicConnectionStrengthFilter();

		/** Destructor. */

		~vtkGeodesicConnectionStrengthFilter();

	private:

		/** ID of the first point of the current fiber. Used to be able to copy the
			measure value of the second point to the first point, since we cannot
			compute an actual value for the first point itself. */

		vtkIdType firstPointId;

		/** Nominator of the fraction used to computed the measure. */

		double nom;

		/** Denominator of the fraction used to computed the measure. */

		double den;

		/** Pointer to the cell containing the current point. Needed for interpolation
			of the DTI tensors at the current position. */

		vtkCell * currentCell;

		/** ID of the cell containing the current point. */

		vtkIdType currentCellId;

}; // class vtkGeodesicConnectionStrengthFilter


} // namespace bmia


#endif // bmia_ConnectivityMeasurePlugin_vtkGeodesicConnectionStrengthFilter_h