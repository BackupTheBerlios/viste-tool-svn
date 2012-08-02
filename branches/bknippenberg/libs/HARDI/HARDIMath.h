/*
 * HARDIMath.h
 *
 * 2007-12-11	Vesna Prckovska
 * - First version
 *
 * 2010-12-07	Evert van Aart
 * - First version for DTITool3. 
 * - A lot of functions have been disabled. They will be re-enabled when
 *   they're needed by other parts of the tool.
 *
 */


#ifndef bmia_HARDIMath_h
#define bmia_HARDIMath_h


/** Includes - C++ */

#include <stdio.h>
#include <stdlib.h>
#include <complex>


namespace bmia {


class HARDIMath
{
	public:

//		static bool Equals(double a, double b, double precision = 0.005);
		/**
		* erf(x) = 2/sqrt(pi)*integral(exp(-t^2),t,0,x)
		*       = 2/sqrt(pi)*[x - x^3/3 + x^5/5*2! - x^7/7*3! + ...]
		*       = 1-erfc(x)
		*/
//		static double erf(double x);
		/**
		* erfc(x) = 2/sqrt(pi)*integral(exp(-t^2),t,x,inf)
		*        = exp(-x^2)/sqrt(pi) * [1/x+ (1/2)/x+ (2/2)/x+ (3/2)/x+ (4/2)/x+ ...]
		*        = 1-erf(x)
		* expression inside [] is a continued fraction so '+' means add to denominator only
		*/
//		static double erfc(double x);
		/**
		* Creates a spherical function
		*/
		
//		static double Legendre (int l, double x);

		static double AssociatedLegendrePolynomial(int l, int m, double x);
		
		static std::complex<double> SHTTable(int l, int m, double theta, double phi);

		/**
		 * Real SHTransform for the q-ball implementation
		 */
		static double RealSHTransform(int l, int m, double theta, double phi);

		/**
		 * Different regularization schemes
		 */
/*
		static MatrixDouble LaplaceBeltrami(int l, double lambda);
		static MatrixDouble ScaleSpace(int l, double t);
		static MatrixDouble Tikhonov(int l, double tikhonov);
*/			
}; // class HARDIMath


} // namespace bmia


#endif // bmia_HARDIMath_h

