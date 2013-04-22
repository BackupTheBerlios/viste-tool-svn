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

