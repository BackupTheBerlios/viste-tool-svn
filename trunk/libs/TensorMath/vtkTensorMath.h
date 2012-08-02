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
 * vtkTensorMath.h
 * by Tim Peeters
 *
 * 2005-01-27	Tim Peeters
 * - First version
 *
 * 2005-02-16	Tim Peeters
 * - Renamed class from vtkExtractTensorProperties to vtkTensorProperties
 * - Removed non-static filter data.
 *
 * 2005-02-26	Tim Peeters
 * - Added DirectionalDiffusion(t, v)
 *
 * 2005-04-22	Tim Peeters
 * - Renamed class from vtkTensorProperties to vtkTensorMath
 *
 * 2006-03-09	Tim Peeters
 * - Added HelixAngle() function.
 *
 * 2006-12-26	Tim Peeters
 * - Removed anisotropy measures. This is now in AnisotropyMeasures.h
 *
 * 2007-02-13 Anna Vilanova
 * - Improved the efficiency of the EigenSystem calculation. Added several functions for tensor calculation:
 *    EigenSystemToTensor, EigenSystemUnsorted, Substraction, ProductWithScalar, SquareRoot, Square
 *    Also Added a calculation of Min and Max to 0 and then the calculation and computation of the square
 *    (For efficiency is better to have it in one function)
 *
 * 2007-05-15	Tim Peeters
 * - Rename helix angle to out-of-plane component.
 * - rename calculateTensor* to *.
 * - Add Product(..) function.
 *
 * 2007-05-31	Tim Peeters
 * - Added Inverse
 *
 * 2007-06-04	Tim Peeters
 * - Added Log()
 * - Added Trace()
 * - Added Deviatoric()
 * 
 * 2007-06-07 	Paulo Rodrigues
 * - Added GeometricMean()
 * - corrected Inverse()
 * 
 * 2007-06-15	Paulo Rodrigues
 * - Added LogMean()
 * - Added Exp()
 *
 * 2007-07-29	Paulo Rodrigues
 * - Means moved to vtkTensorMeans class
 *
 * 2010-09-02	Tim Peeters
 * - Remove all eigensystem computation functions except one.
 *   Its enough hassle to make sure that these are correct.
 * - Change (symmetric!) tensor representation from 9 to 6 scalar values.
 * - Remove all the functions that are not used in this plugin.
 *   Otherwise I would have to rewrite them for the 6-scalar tensor
 *   representation also.
 */

#ifndef bmia_vtkTensorMath_h
#define bmia_vtkTensorMath_h

//TODO: maybe put these somehwere else? Is there a single .h for these kind of 'constants'?
#define BMIA_INTERPOLATION_NEARESTPOLATION_LINEAR  0
#define BMIA_INTERPOLATION_NEAREST 1

namespace bmia {

/**
 * Class for computing eigensystems, and anisotropy indices of 3D
 * diffusion tensors.
 */
class vtkTensorMath
{
public:
  /**
   * Computes the 3 eigenvectors and eigenvalues of a symmetric tensor.
   * Only works on symmetric tensors. If the tensor is symmetric, the function
   * returns true. If it is not, false is returned.
   * Resulting eigenvalues/vectors are sorted in decreasing order.
   * Eigenvalues are positivised
   * Eigenvectors are normalized. Returns true if the calculation ended
   * successfully.
   */
  static bool EigenSystemSorted(double * tensor6, double * eigenvec, double * eigenval);

	/** Computes the eigenvalues and -vectors of an input tensor. Work exactly like
		"EigenSystemSorted", except for the fact that the output eigensystem is NOT sorted.
		@param inTensor		Input tensor, symmetric, six components. 
		@param V			Output eigenvectors, nine components.
		@param W			Output eigenvalues, nine components. */

  static bool EigenSystemUnsorted(double * tensor6, double * V, double * W);

  /**
   * Returns true if all elements of the tensor are 0.
   */
  static bool IsNullTensor(double * tensor6);

  /**
   * Computes the out-of-plane component. That is, the angle the specified vector
   * makes with the xy-plane.
   *
   * @param vector A unit-length 3D vector, usually the first eigenvector.
   *
   * @return An out-of-plane component in the range [-Pi/2, Pi/2].
   */
  static double OutOfPlaneComponent(double vector[3]);

	/** Computes the inverse of a tensor. Returns 1 after successful computation,
		and 0 if the computation fails.
		@param inTensor			Input tensor, six components.
		@param inverseTensor	Output, inverse of input tensor, six components. */

	static int Inverse(double * inTensor, double * inverseTensor);

	/** Computes the determinant of a tensor.
		@param a				Input tensor, six components. */

	static double Determinant(double * a);

	/** Computes the product of a tensor with a scalar. 
		@param inTensor			Input tensor.
		@param scalar			Input scalar value.
		@param outTensor		Output tensor. 
		@param n				Number of components in input and output tensors. */

	static void ProductWithScalar(double * inTensor, double scalar, double * outTensor, int n);

	/** Computes the product of a tensor with a scalar. 
		@param inTensor			Input tensor.
		@param scalar			Input scalar value.
		@param outTensor		Output tensor. 
		@param n				Number of components in input and output tensors. */

	static void DivisionWithScalar(double * inTensor, double scalar, double * outTensor, int n);

	/** Computes the square root of the tensor. This is done by first getting the
		eigenvalues and -vectors from the tensor, computing the square roots of the
		eigenvalues, and then reconstructing the tensor using the new eigenvalues.
		@param inTensor			Input tensor, six components
		@param outTensor		Output tensor, six components. */

	static void  SquareRoot(double * inTensor, double * outTensor);

	/** Computes the logarithm of the tensor. This is done by first getting the
		eigenvalues and -vectors from the tensor, computing the logarithms of the
		eigenvalues, and then reconstructing the tensor using the new eigenvalues.
		@param inTensor			Input tensor, six components
		@param outTensor		Output tensor, six components. */

	static void Log(double * inTensor, double * outTensor);

	/** Computes the exponent of the tensor. This is done by first getting the
		eigenvalues and -vectors from the tensor, computing the exponents of the
		eigenvalues, and then reconstructing the tensor using the new eigenvalues.
		@param inTensor			Input tensor, six components
		@param outTensor		Output tensor, six components. */

	static void Exp(double * inTensor, double * outTensor);

	/** Compute the deviatoric of the input tensor, which is the input tensor, 
		with its diagonal components decrement by a third of its trace. 
		@param inTensor			Input tensor, six components
		@param outTensor		Output tensor, six components. */

	static void Deviatoric(double * inTensor, double * outTensor);

	/** Converts an eigensystem (values and vectors) to a tensor.
		@param W				Eigenvalues, three components.
		@param V				Eigenvectors, nine components.
		@param outTensor		Output tensor, six components. */

	static void EigenSystemToTensor(double * W, double * V, double * outTensor);    

	/** Compute the product of two 3x3 matrices. Both inputs and the output should 1D arrays
		with nine components each. If you're working with six-component tensors (the
		six unique components of a symmetric tensor), you can use the function
		"Tensor6to9" to convert them to nine-component tensor arrays. 
		@param inTensor1		Input tensor, nine components.
		@param inTensor2		Input tensor, nine components.
		@param outTensor		Output tensor, nine components. */

	static void Product(double * inTensor1, double * inTensor2, double * outTensor);

	/** Compute the element-wise sum of two 1D arrays, and store them in a third
		array. The number of components in the array can be specified. 
		@param inTensor1		Input tensor.
		@param inTensor2		Input tensor.
		@param outTensor		Output tensor.
		@param n				Number of components in each array. */

	static void Sum(double * inTensor1, double * inTensor2, double * outTensor, int n);

	/** Compute the element-wise subtraction of two 1D arrays, and store them in a third
		array. The number of components in the array can be specified. 
		@param inTensor1		Input tensor.
		@param inTensor2		Input tensor.
		@param outTensor		Output tensor, defined as "inTensor1 - inTensor2".
		@param n				Number of components in each array. */

	static void Subtract(double * inTensor1, double * inTensor2, double * outTensor, int n);

	/** Flip a tensor over one axis. Used by the transformation plugin.
		@param inTensor			Input tensor.
		@param outTensor		Output tensor.
		@param axis				Axis (0, 1 or 2) over which the tensor should be flipped. */

	static void Flip(double * inTensor, double * outTensor, int axis);

	/** Compute the trace of the input tensor. 
		@param inTensor			Input tensor, six components. */

	static double Trace(double * inTensor);

	/** Convert 6-element tensor array (containing the six unique elements of a 
		symmetric tensor) to a 9-element array, representing the full 3x3 tensor. 
		@param tensor6			Input tensor, six components. 
		@param tensor9			Output tensor, nine components. */

	static void Tensor6To9(double * tensor6, double * tensor9);

	/** Convert 9-element tensor array (representing the full 3x3 tensor) to a 
		6-element tensor array, containing the six unique elements of a symmetric
		tensor. Note that this assumes symmetry; when used on asymmetric tensors,
		this function will result in a loss of interformation.
		@param tensor9			Input tensor, nine components.
		@param tensor6			Output tensor, six components. */

	static void Tensor9To6(double * tensor9, double * tensor6);

protected:
  /**
   * Sorts arrays of eigenvectors and eigenvalues such that in the end,
   * val[0] >= val[1] >= val[2]. Used by EigenSystem().
   */
  static void Sort(double** vec, double* val);
};

} // namespace bmia

#endif
