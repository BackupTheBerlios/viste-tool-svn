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

/**
 * Invariants.h
 *
 * 2008-05-08	Paulo Rodrigues
 * - First version.
 *
 * 2010-12-17	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_Invariants_h
#define bmia_Invariants_h


/** Includes - C++ */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

/** Includes - Custom Files */

#include "vtkTensorMath.h"
#include "vtkTensorStatistics.h"


namespace bmia {


/** This namespace contains methods for computing invariant measures. All output
	measures can be assumed to be in the range 0-1. The inputs are a measure
	index, corresponding to one of the static integers defined below, and one
	tensor, passed as a double pointer. We assume that the tensor is stored
	as six-element 1D array (i.e., one element per unique element of the 
	symmetric tensor. */

namespace Invariants {

	/** Indices of the different invariant measures. */
	static const int K1 = 0;
	static const int K2 = 1;
	static const int K3 = 2;
	static const int R1 = 3;
	static const int R2 = 4;
	static const int R3 = 5;

	/** Number of invariant measures. */
	static const int numberOfMeasures = 6;

	/** Long names of the distance measures. */
	static const char * longNames[] = 
	{
		"K1",
		"K2",
		"K3",
		"R1",
		"R2",
		"R3"
	};

	/** Short names of the distance measures. */
	static const char * shortNames[] = 
	{
		"K1",
		"K2",
		"K3",
		"Rotation Tangent 1",
		"Rotation Tangent 2",
		"Rotation Tangent 3"
	};

	/** Return the requested invariant measure.
		@param measure		Desired invariant measure.
		@param tensor		Input tensor. */

	double computeInvariant(int measure, double * tensor);

	/** Return the requested invariant difference measure.
		@param measure		Desired invariant measure.
		@param tensorA		First input tensor. 
		@param tensorB		Second input tensor. */

	double computeInvariantDifference(int measure, double * tensorA, double * tensorB);

	/** Return the long/short name of the specified measure.
		@param measure		Desired measure. */

	const char * GetLongName(int measure);
	const char * GetShortName(int measure);

	/** Compute the gradient for the specified index. First uses one of the three
		"gradientK" functions, depending on "i", and then normalizes the result. 
		@param i			Desired gradient (should be 0, 1, or 2).
		@param inTensor		Input tensor, six components.
		@param outTensor	Output gradient, six components. */

	void normalizedGradientK(int i, double * inTensor, double * outTensor);

	/** Compute Gordon's shape invariants. 
		@param inTensor		Input tensor, six components. */

	double invariantK1(double * inTensor);
	double invariantK2(double * inTensor);
	double invariantK3(double * inTensor);

	/** Compute invariant gradient. 
		@param inTensor		Input tensor, six components.
		@param outTensor	Output gradient, six components. */

	void gradientK1(double * inTensor, double * outTensor);
	void gradientK2(double * inTensor, double * outTensor);
	void gradientK3(double * inTensor, double * outTensor);

	/** Compute rotation tangents.
		@param inTensor		Input tensor, six components.
		@param outTensor	Output tangent, six components. */

	void rotationTangent1(double * inTensor, double * outTensor);
	void rotationTangent2(double * inTensor, double * outTensor);
	void rotationTangent3(double * inTensor, double * outTensor);

	/** Compute normalized rotation tangents.
		@param inTensor		Input tensor, six components.
		@param outTensor	Output tangent, six components. */

	void rotationTangent1n(double * inTensor, double * outTensor);
	void rotationTangent2n(double * inTensor, double * outTensor);
	void rotationTangent3n(double * inTensor, double * outTensor);

	/** Performs element-wise multiplication of the tensor elements, and
		returns the sum of all elements in the resulting tensor. 
		@param inTensorA	First input tensor, six components. 
		@param inTensorB	Second input tensor, six components. */

	double doubleContraction(double * inTensorA, double * inTensorB);

	/** Computes the norm of the tensor, used for normalization. 
		@param inTensor		Input tensor, six components. */

	double tensorNorm(double * inTensor);

	/** Computes the product of two 3-element vectors, as [3x1]*[1x3]. The
		result is a 3x3 matrix, stored as a 9-element 1D array.
		@param vecA			First vector.
		@param vecB			Second vector.
		@param outTensor	Output matrix, nine components. */

	void vectorProduct(double * vecA, double * vecB, double * outTensor);

	/** Compute the mean of two tensors. 
		@param inTensorA	First input tensor, six components. 
		@param inTensorB	Second input tensor, six components. 
		@param outTensor	Output tensor, six components. */

	void meanTensor(double * inTensorA, double * inTensorB, double * outTensor);

}; // namespace Invariants


} // namespace bmia


#endif // bmia_Invariants_h
