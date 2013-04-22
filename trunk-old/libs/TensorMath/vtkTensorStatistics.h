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
 * vtkTensorStatistics.h
 *
 * 2007-07-29	Paulo Rodrigues
 * - First version (taken from the other classes).
 *
 * 2010-12-17	Evert van Aart
 * - First version for the DTITool3. I currently only need the function
 *   "LogEuclidean", so I will leave the rest commented out for now.
 *
 */


#ifndef bmia_vtkTensorStatistics_h
#define bmia_vtkTensorStatistics_h


/** Includes - VTK */

#include <vtkType.h>
#include <vtkImageData.h>
#include <vtkImageData.h>
#include <vtkType.h>
#include <vtkPointData.h>

/** Includes - C++ */

#include <list>
#include <assert.h>

/** Includes - Custom Files */

#include "vtkTensorMath.h"


namespace bmia {


class vtkTensorStatistics 
{
	public:
	
		static const int i_LogEuclidean	= 0;
		static const int i_Geometric	= 1;
		static const int i_Vemuri		= 2;
		static const int num_means		= 3;

		static const char * longNames[];

		vtkTensorStatistics();
		~vtkTensorStatistics();

//		void setTensorSource(vtkImageData * source);
//		void Reset();

	protected:
	
//		vtkImageData * tensorSource;

		/** Mean tensor. */
//	  	double meanTensor[9];

		/** Eigenvalues of the mean tensor. */
// 		double meanEigenValues[3];

		/** Eigenvectors of the mean tensor. */
// 		double meanEigenVectors[9];

		/** Used to update the log euclidean mean. */
//		double sumLogEuclidean[9];

		/** Used to update the Vemuri mean. */
//		double A[9];
//		double B[9];

		/** Number of tensors in the mean. */
//		int	nPointsInMean;

	public:
//		static void Mean(int mean, double *m1, double *m2, double *outTensor);

		/** Computes the logarithm tensors of the input tensors, sums them, 
			multiplies all elements by 0.5, and finally computes the exponent
			tensor of this sum tensor.
			@param m1			First tensor, six components. 
			@param m2			Second tensor, six components. 
			@param outTensor	Output tensor, six components. */

		static void LogEuclidean(double * m1, double * m2, double * outTensor);

//		static void Geometric(double *m1, double *m2, double *outTensor);
//		static void Vemuri(double *m1, double *m2, double *outTensor);
//		static void Vemuri_AB(double *m1, double *m2, double *outTensor);

//		void Mean(int mean, std::list<vtkIdType>* lst);
//		void LogEuclidean(std::list<vtkIdType>* lst);
//		void Geometric(std::list<vtkIdType>* lst);
//		void Vemuri(std::list<vtkIdType>* lst);

//		void Covariance(std::list<vtkIdType>* lst, double* outTensor);
//		double UnbiasMeanSquareDistance(double *covar);

//		void UpdateMean(int mean, double* m2);
//		void Update_LogEuclidean(double *m2);
//		void Update_Geometric(double *m2);
//		void Update_Vemuri(double *m2);

}; // class vtkTensorStatistics 


} // namespace bmia


#endif // bmia_vtkTensorStatistics_h