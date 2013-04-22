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
 * HARDITransformationManager.h
 *
 * 2008-10-18	Vesna Prckovska
 * - First version
 *
 *  2009-03-20	Tim Peeters
 *  - Add FiberODF()
 *
 * 2010-12-07	Evert van Aart
 * - Made compatible with DTITool3. 
 * - Removed dependence on GSL. Visualization of HARDI glyphs depends on this
 *   class, and we want basic visualization to be independent of GSL.
 * - Most functions are disabled at the moment. I will re-enable them when they're
 *   needed by another part of the tool.
 *
 */


#ifndef bmia_HARDITransformationManager_h
#define bmia_HARDITransformationManager_h



#include "HARDIMath.h"
#include <vtkMath.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <strstream>
#include <fstream>
#include <assert.h>
#include <vector>


namespace bmia {

	/**
	* Class for computing the ransformations begore calculating the SH coefficients
	*/
	class HARDITransformationManager
	{
	public:


		HARDITransformationManager();
		~HARDITransformationManager();

		/**
		* Regularization parameter lambda fo Laplace-Beltrami regularization
		*/
//		static double lambda;

		/**
		* Regularization parameter t for ScaleSpace regularization
		*/
//		static double t;

		/**
		* Regularization parameter tikhonov for Tikhonov regularization
		*/
//		static double tikhonov;

		/**
		*  matrix of SH matrix for all directions
		* 
		*/
//		static MatrixDouble BMatrix(MatrixDouble gradients, unsigned int lmax, unsigned int lmin=0);

		static void FiberODF(int l, double b, double * eigenval, double * coefficientsIn, double * coefficientsOut);
		static double FiberODF_A(int l, double a);

		/**
		* FRT matrix
		*/

//		static MatrixDouble LegendreQballTransformation(int l);

		/**
		* FRT matrix for sharp Qball
		*/

//		static MatrixDouble LegendreQballSharpTransformation(int l);

		/**
		*	Least square fit first step for calculating the SH coefficients 
		*/
//		static MatrixDouble LeastSquareFit(int l, MatrixDouble gradientListInSphericalCoordinates);

		/**
		*	FR transform only for Q-ball
		*/
//		static MatrixDouble QballTransform(int l, const MatrixDouble& gradientListInSphericalCoordinates);
				/**
		*	FR transform only for sharp Q-ball
		*/
//		static MatrixDouble QballSharpTransform(int l, const MatrixDouble& gradientListInSphericalCoordinates);

		/**
		*	Return function for the transformation needed for QBall
		*/
//		MatrixDouble GetTransformMatrixQball(int l, const MatrixDouble& gradientListInSphericalCoordinates);
/**
		*	Return function for the transformation needed for sharp QBall
		*/
//		MatrixDouble GetTransformMatrixQballSharp(int l, const MatrixDouble& gradientListInSphericalCoordinates);
		/**
		* Least square fit transformation (used for Qball smoothing on the signal
		*/
//		MatrixDouble GetTransformMatrixDOT(unsigned int l, MatrixDouble gradients);

	/**
	*   This method takes as an input array of coefficients and array of directions                                                                 
	*/
		static std::vector<double> CalculateDeformator(double * SHCoefficients, std::vector<double *> * tessellationPointsInSC, int l);
	
	
	protected:

//		MatrixDouble savedTransformationMatrix;
//		int savedLOrder;


	};
} // namespace bmia

#endif

