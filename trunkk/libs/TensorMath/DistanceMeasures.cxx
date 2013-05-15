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
 * DistanceMeasures.h
 *
 * 2007-03-26	Tim Peeters
 * - First version.
 *
 * 2007-06-07	Tim Peeters
 * - Add a lot of new measures from our overview paper.
 *
 * 2007-07-25	Paulo Rodrigues
 * - TSP corrected, AngularDifference, KullbackLeibner, Bhattacharyya, SimilarityCombined
 *
 * 2008-02-03	Jasper Levink
 * - Changed name of Log2-distance from L2 to L2D
 *
 * 2010-12-16	Evert van Aart
 * - First version for the DTITool3.
 *
 */


/** Includes */

#include "DistanceMeasures.h"


/** Definitions */

#define ERROR_PRECISION 1.0e-20f	// We consider this to be zero


namespace bmia {


namespace Distance {


//---------------------------[ computeDistance ]---------------------------\\

double computeDistance(int measure, double * tensorA, double * tensorB)
{
	// Check if the index is within range
	assert(!(measure < 0 || measure >= Distance::numberOfMeasures));

	// Compute and return the desired measure
	switch (measure)
	{
		case i_Angular:			return Distance::AngularDifference(tensorA, tensorB);
		case i_L2D:				return Distance::L2Distance(tensorA, tensorB);
		case i_Geometric:		return Distance::Geometric(tensorA, tensorB);
		case i_LogEuclidian:	return Distance::LogEuclidian(tensorA, tensorB);
		case i_KullbackLeibner:	return Distance::KullbackLeibner(tensorA, tensorB);
		case i_sp:				return Distance::ScalarProduct(tensorA, tensorB);
		case i_tsp:				return Distance::TensorScalarProduct(tensorA, tensorB);
		case i_ntsp:			return Distance::NormalizedTensorScalarProduct(tensorA, tensorB);
		case i_Bhattacharyya:	return Distance::Bhattacharyya(tensorA, tensorB);
		case i_l:				return Distance::SimilarityLinear(tensorA, tensorB);
		case i_p:				return Distance::SimilarityPlanar(tensorA, tensorB);
		case i_s:				return Distance::SimilaritySpherical(tensorA, tensorB);
		case i_pnl:				return Distance::SimilarityCombined(tensorA, tensorB);
		case i_LI:				return Distance::Lattice(tensorA, tensorB);
		case i_Angular2:		return Distance::AngularDifference(2, tensorA, tensorB);
		case i_Angular3:		return Distance::AngularDifference(3, tensorA, tensorB);

		default:
			// This should never happen
			assert(false);
    }

	return 0.0;
}


//------------------------------[ DotProduct ]-----------------------------\\

double DotProduct(double vec1[3], double vec2[3])
{
	// Return the dot product of the two vectors
	return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}


//--------------------------[ AngularDifference ]--------------------------\\

double AngularDifference(double * tensorA, double * tensorB)
{
	// Use the first eigenVector by default
	return Distance::AngularDifference(1, tensorA, tensorB);
}


//--------------------------[ AngularDifference ]--------------------------\\

double AngularDifference(int eigenVector, double * tensorA, double * tensorB)
{
	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	// The "eigenVector" variable should 1, 2, or 3
	assert(eigenVector >= 1 && eigenVector <= 3);

	// Eigenvector and -values of tensors A and B
	double eigenVecA[9];
	double eigenValA[3];
	double eigenVecB[9];
	double eigenValB[3];

	// Compute eigenvectors and -values
	vtkTensorMath::EigenSystemSorted(tensorA, eigenVecA, eigenValA);
	vtkTensorMath::EigenSystemSorted(tensorB, eigenVecB, eigenValB);

	// Compute the dot product between the specified eigenvectors
	double dot = Distance::DotProduct(&eigenVecA[(eigenVector - 1) * 3], &eigenVecB[(eigenVector - 1) * 3]);

	// Clamp dot product to the range [-1.0, 1.0]
	if(dot >  1.0)	dot =  1.0;
	if(dot < -1.0)	dot = -1.0;

	// Compute the angle between the eigenvectors
	return acos(fabs(dot));
}


//------------------------------[ L2Distance ]-----------------------------\\

double L2Distance(double * tensorA, double * tensorB)
{
	// Sum of squared differences
	double sum = 0.0;

	// Difference between tensor components
	double diff;
  
	// Compute sum of squared tensor component differences
	for (int i = 0; i < 6; i++)
	{
		diff = tensorA[i] - tensorB[i];
		sum += diff * diff;
	}

	// Return the square root of the sum
	return sqrt(sum);
}


//------------------------------[ Geometric ]------------------------------\\

double Geometric(double * tensorA, double * tensorB)
{
	// Output value
	double result = 0.0;

	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	double invTensorA[6];
	double invTensorASQRT[6];
	double invTensorASQRT9[9];
	double tensorB9[9];
	double tensorProductA[9];
	double tensorProductB[9];

	// Compute the inverse of the first tensor
	if ((vtkTensorMath::Inverse(tensorA, invTensorA)) == 0)
	{
		cerr << "Geometric: Could not compute inverse of matrix!" << endl;
		return 0.0;
	}

	// Compute the square root of the inverse
	vtkTensorMath::SquareRoot(invTensorA, invTensorASQRT);

	// Convert six-component tensors to nine-component tensors
	vtkTensorMath::Tensor6To9(invTensorASQRT, invTensorASQRT9);
	vtkTensorMath::Tensor6To9(tensorB, tensorB9);

	// Compute "invTensorASQRT9 * tensorB9 * invTensorASQRT9"
	vtkTensorMath::Product(invTensorASQRT9, tensorB9, tensorProductA);
	vtkTensorMath::Product(tensorProductA, invTensorASQRT9, tensorProductB);

	// Convert final product to 6-element tensor array
	double tensorProductB6[6];
	vtkTensorMath::Tensor9To6(tensorProductB, tensorProductB6);

	// Compute the eigensystem of the product tensor
	double W[3]; 
	double V[9]; 
	if (!(vtkTensorMath::EigenSystemSorted(tensorProductB6, V, W)))
		return 0.0;

	// Compute the logarithms of the eigenvalues
	double logW[3];

	for (int i = 0; i < 3; ++i)
    {
		if(W[i] < 0)
		{
			cerr << "Geometric: Encountered a negative eigenvalue!" << endl;
			return 0.0;
		}

		logW[i] = log(W[i]);
    }
	
	// Compute output
	for (int i = 0; i < 3; ++i)
    {
		result += logW[i] * logW[i];
    }

	return sqrt(result);
}


//-----------------------------[ LogEuclidian ]----------------------------\\

double LogEuclidian(double * tensorA, double * tensorB)
{
	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	// Logarithms of the input tensors
    double logA[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double logB[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	// Compute the logarithms
	vtkTensorMath::Log(tensorA, logA);
	vtkTensorMath::Log(tensorB, logB);

	// Compute the L2 distance of the logarithm tensors
	return Distance::L2Distance(logA, logB);
}


//---------------------------[ KullbackLeibner ]---------------------------\\

double KullbackLeibner(double * tensorA, double * tensorB)
{
	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	double invTensorA[6];

	// Compute the inverse of the first tensor
	if ((vtkTensorMath::Inverse(tensorA, invTensorA)) == 0)
	{
		cerr << "KullbackLeibner: Could not compute inverse of matrix!" << endl;
		return 0.0;
	}

	double invTensorB[6];

	// Compute the inverse of the second tensor
	if ((vtkTensorMath::Inverse(tensorB, invTensorB)) == 0)
	{
		cerr << "KullbackLeibner: Could not compute inverse of matrix!" << endl;
		return 0.0;
	}

	// Convert 6-element tensors to 9-element tensors
	double tensorA9[9];
	double tensorB9[9];
	double invTensorA9[9];
	double invTensorB9[9];

	vtkTensorMath::Tensor6To9(tensorA, tensorA9);
	vtkTensorMath::Tensor6To9(tensorB, tensorB9);
	vtkTensorMath::Tensor6To9(invTensorA, invTensorA9);
	vtkTensorMath::Tensor6To9(invTensorB, invTensorB9);

	// Product and sum matrices
	double prod1[9];
	double prod2[9];
	double sum[9];

	// Compute the tensor products and the sum
	vtkTensorMath::Product(invTensorA9, tensorB9, prod1);
	vtkTensorMath::Product(invTensorB9, tensorA9, prod2);
	vtkTensorMath::Sum(prod1, prod2, sum, 9);

	// Compute the trace of the sum matrix
	double trace = sum[0] + sum[4] + sum[8];

	// Subtract six from the trace
	double dif = trace - 6.0; 
	
	// This should not be negative
	if(dif < 0.0) 
		dif = 0.0;

	return (0.5 * sqrt(dif));
}


//----------------------------[ ScalarProduct ]----------------------------\\

double ScalarProduct(double * tensorA, double * tensorB)
{
	// Perform element-wise multiplication of the two tensors, and compute the sum
	// of the resulting tensors. Off-diagonal elements are counted twice because
	// of the symmetry of the tensors.

	return (	(    tensorA[0] * tensorB[0]) + (    tensorA[3] * tensorB[3]) + (    tensorA[5] * tensorB[5]) +
				(2 * tensorA[1] * tensorB[1]) + (2 * tensorA[2] * tensorB[2]) + (2 * tensorA[4] * tensorB[4]) );
}


//-------------------------[ TensorScalarProduct ]-------------------------\\

double TensorScalarProduct(double * tensorA, double * tensorB)
{
	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	// Compute eigenvalues and -vectors of both tensors
	double AW[3];
	double AV[9];
	double BW[3]; 
	double BV[9]; 

	if (!(vtkTensorMath::EigenSystemSorted(tensorA, AV, AW)) || !(vtkTensorMath::EigenSystemSorted(tensorB, BV, BW)))
	{
		cerr << "TensorScalarProduct: Could not compute eigensystem!" << endl;
		return 0.0;
	}

	// Output value
	double result = 0.0;

	// Dot product of two vectors
	double dot = 0.0;

	for(int i = 0; i < 3; i++)
	{
		dot = Distance::DotProduct(&AV[3 * i + 0], &BV[0]);
		result += AW[i] * BW[0] * dot * dot;
	  
		dot = Distance::DotProduct(&AV[3 * i + 0], &BV[3 * 1 + 0]);
		result += AW[i]*BW[1]*dot*dot;
	  
		dot = Distance::DotProduct(&AV[3 * i + 0], &BV[3 * 2 + 0]);
		result += AW[i] * BW[2] * dot * dot;
	}

	return result;
}


//--------------------[ NormalizedTensorScalarProduct ]--------------------\\

double NormalizedTensorScalarProduct(double * tensorA, double * tensorB)
{
	// Product of the trace of both tensors
	double Trace2 = vtkTensorMath::Trace(tensorA) * vtkTensorMath::Trace(tensorB);

	// Prevent division by zero
	if (Trace2 < ERROR_PRECISION) 
	{
		return 0.0;
	}

	// Normalize tensor scalar product by the trace product
	return Distance::TensorScalarProduct(tensorA, tensorB) / Trace2;
}


//----------------------------[ Bhattacharyya ]----------------------------\\

double Bhattacharyya(double * tensorA, double * tensorB)
{
	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	// Compute the determinants of the tensors
	double detA = vtkTensorMath::Determinant(tensorA);
	double detB = vtkTensorMath::Determinant(tensorB);

	// Determinants should be non-zero
	if ((detA == 0.0) || (detB == 0.0)) 
		return 0.0;

	// Determinants should have the same sign
	if (detA * detB <= 0) 
		return 0.0;

	double tensorAB[6];

	// Compute sum of the tensors, and multiply it by 0.5
	vtkTensorMath::Sum(tensorA, tensorB, tensorAB, 6);
	vtkTensorMath::ProductWithScalar(tensorAB, 0.5, tensorAB, 6);

	// Compute determinant of the sum tensor
	double detAB = vtkTensorMath::Determinant(tensorAB);

	// Compute output
	double result = detAB / sqrt(detA * detB);
	result = -0.5 * log(result);
	return (1.0 - exp(result));
}


//---------------------------[ SimilarityLinear ]--------------------------\\

double SimilarityLinear(double * tensorA, double * tensorB)
{ 
	// Cosine of the angular difference of the first eigenvectors
	double r = cos(Distance::AngularDifference(1, tensorA, tensorB));
	return r;
}


//---------------------------[ SimilarityPlanar ]--------------------------\\

double SimilarityPlanar(double * tensorA, double * tensorB)
{
	// Cosine of the angular difference of the third eigenvectors
	double r = cos(Distance::AngularDifference(3, tensorA, tensorB));
	return r;
}


//-------------------------[ SimilaritySpherical ]-------------------------\\

double SimilaritySpherical(double * tensorA, double * tensorB)
{
	// Return one if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 1.0;

	// Compute trace for both tensors
	double trA = vtkTensorMath::Trace(tensorA);
	double trB = vtkTensorMath::Trace(tensorB);
  
	// Compute maximum trace
	double trMax = 1.0;
  	if (trA > trMax) trMax = trA;
	if (trB > trMax) trMax = trB;

	return 1.0 - (fabs(trA - trB) / trMax);
}


//--------------------------[ SimilarityCombined ]-------------------------\\

double SimilarityCombined(double * tensorA, double * tensorB)
{
	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	// Compute eigenvalues and -vectors of both tensors
	double AW[3];
	double AV[9];
	double BW[3]; 
	double BV[9]; 

	if (!(vtkTensorMath::EigenSystemSorted(tensorA, AV, AW)) || !(vtkTensorMath::EigenSystemSorted(tensorB, BV, BW)))
	{
		cerr << "SimilarityCombined: Could not compute eigensystem!" << endl;
		return 0.0;
	}

	// Pre-compute scalar factors
	double ClA = (AW[0] - AW[1]) / AW[0];
	double ClB = (BW[0] - BW[1]) / BW[0];
	double CpA = (AW[1] - AW[2]) / AW[0];
	double CpB = (BW[1] - BW[2]) / BW[0];
	double CsA =  AW[2] / AW[0];
	double CsB =  BW[2] / BW[0];

	// Output is the weighted sum of three other similarity measures
	double result = 0.0;
	result += ClA * ClB * Distance::SimilarityLinear(tensorA, tensorB);
	result += CpA * CpB * Distance::SimilarityPlanar(tensorA, tensorB);
	result += 0.5 * CsA * CsB * Distance::SimilaritySpherical(tensorA, tensorB);
	return result;
}


//-------------------------------[ Lattice ]-------------------------------\\

double Lattice(double * tensorA, double * tensorB)
{
	// Return zero if one of the input tensors only has zeros
	if ((vtkTensorMath::IsNullTensor(tensorA)) || (vtkTensorMath::IsNullTensor(tensorB))) 
		return 0.0;

	// Compute the deviatorics of the tensors
	double devA[6];
	double devB[6];

	vtkTensorMath::Deviatoric(tensorA, devA);
	vtkTensorMath::Deviatoric(tensorB, devB);

	// Compute the tensor scalar product for different tensor combinations
	double tspDev = Distance::TensorScalarProduct(devA, devB);
	double tspTen = Distance::TensorScalarProduct(tensorA, tensorB);
	double tspAA  = Distance::TensorScalarProduct(tensorA, tensorA);
	double tspBB  = Distance::TensorScalarProduct(tensorB, tensorB);

	// Compute the output value
	double result = sqrt(3.0) / sqrt(8.0);
	result *= sqrt(tspDev) / sqrt(tspTen);
	result += (3.0 / 4.0) * tspDev / (sqrt(tspAA) * sqrt(tspBB));
	return result;
}


//---------------------------[ ScalarDifference ]--------------------------\\

double ScalarDifference(int measure, double * WA, double * WB)
{
	// Return the difference between the anisotropy measures
	double resultA = AnisotropyMeasures::AnisotropyMeasure(measure, WA);
	double resultB = AnisotropyMeasures::AnisotropyMeasure(measure, WB);
	return fabs(resultA - resultB);
}


//------------------------[ ScalarDifferenceTensor ]-----------------------\\

double ScalarDifferenceTensor(int measure, double * tA, double * tB)
{
	// Compute eigenvalues and -vectors of both tensors
	double AW[3];
	double AV[9];
	double BW[3]; 
	double BV[9]; 

	if (!(vtkTensorMath::EigenSystemSorted(tA, AV, AW)) || !(vtkTensorMath::EigenSystemSorted(tB, BV, BW)))
	{
		cerr << "ScalarDifferenceTensor: Could not compute eigensystem!" << endl;
		return 0.0;
	}
  	
	return Distance::ScalarDifference(measure, AW, BW);
}


} // namespace Distance


} // namespace bmia


/** Undefine Temporary Definitions */

#undef ERROR_PRECISION
