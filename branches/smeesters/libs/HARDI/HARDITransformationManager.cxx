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
 * HARDITransformationManager.cxx
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


#include "HARDITransformationManager.h"


static const double rel_error= 1E-12;


using namespace std;

namespace bmia {
/*
	double HARDITransformationManager::lambda = 0.000;

	double HARDITransformationManager::t = 0.000;

	double HARDITransformationManager::tikhonov = 0.000;
*/
	HARDITransformationManager::HARDITransformationManager()
	{	
//		this->savedLOrder = 0;
	}

	HARDITransformationManager::~HARDITransformationManager()
	{
		// Not needed anymore
		// delete this->savedTransformationMatrix;
	}


void HARDITransformationManager::FiberODF(int l, double b, double * eigenval, double * coefficientsIn, double * coefficientsOut)
{
	// Store input/output coefficient pointers
	double * f = coefficientsOut;
	double * c = coefficientsIn;

	// The order should be 0, 2, 6, or 8
	if (l < 0 || l > 8 || (l % 2) != 0)	
	{
		return;
	}

	// If the second eigenvalue is larger than the first one, we can't do FiberODF sharpening
	if (eigenval[1] > eigenval[0])
	{
		return;
	}

	// Compute the number of coefficients from the order
	int numberOfCoefficients = (l + 1) * (l + 2) / 2;

	// Coefficient index
	int coeffId = 0;

	// Loop through all coefficients
	for (int k = 0; k <= l; k += 2)
	{
		for (int m = -k; m <= k; m++, coeffId++)
		{
			// Compute the output coefficients
			f[coeffId] = (4.0 * b * sqrt(eigenval[0] * eigenval[1]) / (FiberODF_A(k, 1 - (eigenval[1] / eigenval[0])))) * c[coeffId];
		}
	}
	  
	return;
}


double HARDITransformationManager::FiberODF_A(int l, double a)
{
	// Output result
	double r;

	// Switch based on the order
	switch (l) 
	{
		case 0:
			r = 2.0 * asin(sqrtf(a)) / sqrtf(a);
			break;
		
		case 2:
			r = -1.0 * ((-3.0 + 2.0 * a) * asin(sqrtf(a)) 
					+ 3.0 * sqrtf(1.0 - a) * sqrtf(a)) / (2.0 * a * sqrtf(a)) / (2.0 * a * sqrtf(a));
			break;
	  
		case 4:
			r = ((105.0 - 120.0 * a + 24.0 * a * a) * asin(sqrtf(a)) 
					+ (50.0 * a - 105.0) * sqrtf(1.0 - a) * sqrtf(a)) / (32.0 * a * a * sqrtf(a));
			break;
	  
		case 6:
			r = -1.0 * ((-1155.0 + 1890.0 * a - 840.0 * a * a + 80.0 * a * a * a) * asin(sqrtf(a)) 
					+ (1155.0 - 1120.0 * a + 196.0 * a * a) * sqrtf(1.0 - a) * sqrtf(a)) / (128.0 * a * a * a * sqrtf(a));
			break;
	  
		case 8:
			r = ((225225.0 - 480480.0 * a + 332640.0 * a * a - 80640.0 * a * a * a + 4480.0 * a * a * a * a) * asin(sqrtf(a)) 
					+ (-225225.0 + 330330.0 * a - 132440.0 * a * a + 12176.0 * a * a * a) * sqrtf(1.0 - a) * sqrtf(a)) / (8192.0 * a * a * a * a * sqrtf(a));
			break;
	  
		default:
			assert(0);
			break;
	} // switch [l]
	
	return r;
}

/*
	MatrixDouble HARDITransformationManager::GetTransformMatrixQball(int l, const MatrixDouble& gradientListInSphericalCoordinates)
	{
		MatrixDouble Output;
		//XXXXX
		//TODO: Vesna not only l order but also the smoothing!!!!
		//if(this->savedLOrder!=l)
		//{
			this->savedLOrder=l;
			Output = HARDITransformationManager::QballTransform(l, gradientListInSphericalCoordinates);
			this->savedTransformationMatrix = Output;
		//}

		return this->savedTransformationMatrix;

	}
		MatrixDouble HARDITransformationManager::GetTransformMatrixQballSharp(int l, const MatrixDouble& gradientListInSphericalCoordinates)
	{
		MatrixDouble Output;
		//XXXXX
		//TODO: Vesna not only l order but also the smoothing!!!!
		//if(this->savedLOrder!=l)
		//{
			this->savedLOrder=l;
			Output = HARDITransformationManager::QballSharpTransform(l, gradientListInSphericalCoordinates);
			this->savedTransformationMatrix = Output;
		//}

		return this->savedTransformationMatrix;

	}

	MatrixDouble HARDITransformationManager::GetTransformMatrixDOT(unsigned int l, MatrixDouble gradients)
	{
		// If the order is not preprocessed yet, do so and save it
		//if(this->savedLOrder!=l)
		//{
			this->savedLOrder=l;
			this->savedTransformationMatrix = HARDITransformationManager::LeastSquareFit(l, gradients);

		//}

		return this->savedTransformationMatrix;
	}

	MatrixDouble HARDITransformationManager::BMatrix(MatrixDouble gradients, unsigned int lmax, unsigned int lmin)
	{
		// This should not be able to become a non-integer either l+1 or l+2 is even (other is odd), 
		// even times odd is even, divided by two is a round number
		size_t R=((lmax+1)*(lmax+2)-((lmin-1)*(lmin)))/2;

		MatrixDouble B = MatrixDouble( gradients.getNRow(), R );

		for (size_t i=0; i < gradients.getNRow(); i++)
		{
			size_t coeffId=0;
			for (int lOrder=lmin; lOrder <= (int) lmax; lOrder+=2)
			{
				for (int m=-lOrder; m <= lOrder; m++)
				{
					B(i, coeffId) = HARDIMath::RealSHTransform( lOrder, m, gradients.getElement(i, 0), gradients.getElement(i, 1) );
					coeffId++;
				}
			}
		}

		return B;
	}


	MatrixDouble HARDITransformationManager::LegendreQballTransformation(int l)
	{
		double R=(l+1)*(l+2)/2;
		int lOrder;

		MatrixDouble P(R,R);

		for (int i=0; i<R; i++)
		{
			if (i==0) { lOrder=0; }
			else if ((i>0)&&(i<=5)) {lOrder = 2;}
			else if ((i>5)&&(i<=14)){lOrder = 4;}
			else if ((i>14)&&(i<=27)){lOrder = 6;}
			else if ((i>27)&&(i<=44)) {lOrder = 8;}

			P(i,i) = 2*vtkMath::DoublePi()*HARDIMath::Legendre(lOrder,0);
		}
		return P;
	}

		MatrixDouble HARDITransformationManager::LegendreQballSharpTransformation(int l)
	{
		double R=(l+1)*(l+2)/2;
		int lOrder;

		MatrixDouble P(R,R);

		for (int i=0; i<R; i++)
		{
			if (i==0) { lOrder=0; }
			else if ((i>0)&&(i<=5)) {lOrder = 2;}
			else if ((i>5)&&(i<=14)){lOrder = 4;}
			else if ((i>14)&&(i<=27)){lOrder = 6;}
			else if ((i>27)&&(i<=44)) {lOrder = 8;}

			P(i,i) = ((-1)/(8*vtkMath::DoublePi()))*HARDIMath::Legendre(lOrder,0)*lOrder*(lOrder+1);
		}
		return P;
	}

	//(B^TB+L)^-1B^T
	MatrixDouble HARDITransformationManager::LeastSquareFit(int l, MatrixDouble gradients)
	{
		// double R=(l+1)*(l+2)/2;
		MatrixDouble bmatrix = HARDITransformationManager::BMatrix(gradients, l);
		MatrixDouble regMatrix;

		// Depending on the smoothing method get the correct Matrix
		if (HARDITransformationManager::lambda>0)
		{
			regMatrix = HARDIMath::LaplaceBeltrami(l, HARDITransformationManager::lambda);
		} 
		else if (HARDITransformationManager::t>0)
		{
			regMatrix = HARDIMath::ScaleSpace(l, HARDITransformationManager::t);
		}
		else if (HARDITransformationManager::tikhonov>0)
		{
			regMatrix = HARDIMath::Tikhonov(l, HARDITransformationManager::tikhonov);
		}
		else 
		{
			regMatrix = HARDIMath::LaplaceBeltrami(l, 0);
		}

		// Calculate the output
		return (bmatrix.Transposed() * bmatrix + regMatrix).Inverse() * bmatrix.Transposed();
	}

	MatrixDouble HARDITransformationManager::QballTransform(int l, const MatrixDouble& gradientListInSphericalCoordinates)
	{
		MatrixDouble M, P;

		M = HARDITransformationManager::LeastSquareFit(l, gradientListInSphericalCoordinates);
		P = HARDITransformationManager::LegendreQballTransformation(l);

		return P * M;

	}
	MatrixDouble HARDITransformationManager::QballSharpTransform(int l, const MatrixDouble& gradientListInSphericalCoordinates)
	{
		MatrixDouble M, P;

		M = HARDITransformationManager::LeastSquareFit(l, gradientListInSphericalCoordinates);
		P = HARDITransformationManager::LegendreQballSharpTransformation(l);

		return P * M;

	}
*/
std::vector<double> HARDITransformationManager::CalculateDeformator(double * SHCoefficients, std::vector<double *> * tessellationPointsInSC, int l)
{
	// Output deformator vector
	std::vector<double> deformator;

	// Loop through all tessellation points
	for (unsigned int deformatorIndex = 0; deformatorIndex < tessellationPointsInSC->size(); ++deformatorIndex)
	{
		// Index of spherical coordinates
		int coefficientIndex = 0;

		// Current deformator value
		double deformatorValue = 0.0;
			
		// Get spherical coordinates of current tessellation point
		double * tessSC = tessellationPointsInSC->at(deformatorIndex);

		// Loop through all SH coefficients
		for (int i = 0; i <= l; i += 2)
		{
			for (int m = -i; m <= i; ++m, ++coefficientIndex)
			{
				// Compute deformator value
				deformatorValue += SHCoefficients[coefficientIndex] * HARDIMath::RealSHTransform(i, m, tessSC[0], tessSC[1]);
			}
		}

		// Add value to the vector
		deformator.push_back(deformatorValue);
	}

	// Done, return the deformator vector
	return deformator;
}


} // namespace bmia

