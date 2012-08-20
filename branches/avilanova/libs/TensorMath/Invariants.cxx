/**
 * Invariants.cxx
 *
 * 2008-05-08	Paulo Rodrigues
 * - First version.
 *
 * 2010-12-17	Evert van Aart
 * - First version for the DTITool3.
 *
 */


/** Includes */

#include "Invariants.h"


using namespace std;


namespace bmia {


namespace Invariants {


//---------------------------[ computeInvariant ]--------------------------\\

double computeInvariant(int measure, double * tensor)
{
	// Check if the index is within range
	assert(!(measure < 0 || measure >= Invariants::numberOfMeasures));

	double rotation[6];

	// Compute and return the desired measure
	switch (measure)
	{
		case K1:	return Invariants::invariantK1(tensor);
		case K2:	return Invariants::invariantK2(tensor);
		case K3:	return Invariants::invariantK3(tensor);

		case R1:	
			Invariants::rotationTangent1(tensor, rotation);
			return Invariants::tensorNorm(rotation); 
		case R2:	
			Invariants::rotationTangent2(tensor, rotation);
			return Invariants::tensorNorm(rotation); 
		case R3:	
			Invariants::rotationTangent2(tensor, rotation);
			return Invariants::tensorNorm(rotation); 

		default:
			// This should never happen
			assert(false);

	}

	return 0.0;
}


//----------------------[ computeInvariantDifference ]---------------------\\

double computeInvariantDifference(int measure, double * tensorA, double * tensorB)
{
	// Check if the index is within range
	assert(!(measure < 0 || measure >= Invariants::numberOfMeasures));

	// Compute the difference between the two tensors
	double difference[6];
	vtkTensorMath::Subtract(tensorA, tensorB, difference, 6);

	// Compute the mean of the tensors
	double mean[6];
	vtkTensorStatistics::LogEuclidean(tensorA, tensorB, mean);

	double rotation[6];

	// Compute and return the desired measure
	switch (measure)
	{
		case K1:	return fabs(Invariants::invariantK1(tensorA) - Invariants::invariantK1(tensorB));
		case K2:	return fabs(Invariants::invariantK2(tensorA) - Invariants::invariantK2(tensorB));
		case K3:	return fabs(Invariants::invariantK3(tensorA) - Invariants::invariantK3(tensorB));

    	case R1:	
			Invariants::rotationTangent1n(mean, rotation);
			return fabs(Invariants::doubleContraction(difference, rotation));
		case R2:	
			Invariants::rotationTangent2n(mean, rotation);
			return fabs(Invariants::doubleContraction(difference, rotation));
		case R3:	
			Invariants::rotationTangent3n(mean, rotation);
			return fabs(Invariants::doubleContraction(difference, rotation));

		default:
			// This should never happen
			assert(false);

	}

	return 0.0;
}


//-----------------------------[ GetLongName ]-----------------------------\\

const char * GetLongName(int measure)
{
	// Check if the index is within range
	assert(!(measure < 0 || measure >= Invariants::numberOfMeasures));

	return longNames[measure];
}


//-----------------------------[ GetShortName ]----------------------------\\

const char * GetShortName(int measure)
{
	// Check if the index is within range
	assert(!(measure < 0 || measure >= Invariants::numberOfMeasures));

	return shortNames[measure];
}


//-----------------------------[ invariantK1 ]-----------------------------\\

double invariantK1(double * inTensor)
{
	// Return the trace of the tensor
	return vtkTensorMath::Trace(inTensor);
}


//-----------------------------[ invariantK2 ]-----------------------------\\

double invariantK2(double * inTensor)
{
	// Compute deviatoric of input tensor
	double dev[6];
	vtkTensorMath::Deviatoric(inTensor, dev);

	// Return the norm of the deviatoric
	double k2 = Invariants::tensorNorm(dev);
	return k2;
}


//-----------------------------[ invariantK3 ]-----------------------------\\

double invariantK3(double * inTensor)
{
	// Compute deviatoric of input tensor
	double dev[6];
	vtkTensorMath::Deviatoric(inTensor, dev);

	// Compute the norm of the deviatoric
	double norm = Invariants::tensorNorm(dev);

	if(norm > 0.0)
	{
		// Divide the deviatoric by the norm
		double div[6];
		vtkTensorMath::DivisionWithScalar(dev, norm, div, 6);

		// Compute the determinant of the resulting tensor
		double det = vtkTensorMath::Determinant(div);

		return (3.0 * sqrt(6.0) * det);
	}
	else
	{
		return 0.0;
	}
}


//------------------------------[ gradientK1 ]------------------------------\\

void gradientK1(double * inTensor, double * outTensor)
{
	// Create the identity tensor
	outTensor[0] = 1.0;
	outTensor[1] = 0.0;
	outTensor[2] = 0.0;
	outTensor[3] = 1.0;
	outTensor[4] = 0.0;
	outTensor[5] = 1.0;
}


//------------------------------[ gradientK2 ]------------------------------\\

void gradientK2(double * inTensor, double * outTensor)
{
	// Compute deviatoric of input tensor
	double dev[6];
	vtkTensorMath::Deviatoric(inTensor, dev);

	// Compute the norm of the deviatoric
	double norm = Invariants::tensorNorm(dev);

	if(norm > 0.0)
	{
		// Divide the deviatoric by the norm
		vtkTensorMath::DivisionWithScalar(dev, norm, outTensor, 6);
	}
	else
	{
		// If the norm isn't positive, set the output to zero
		memset(outTensor, 0.0, sizeof(double) * 6);
	}
}


//------------------------------[ gradientK3 ]------------------------------\\

void gradientK3(double * inTensor, double * outTensor)
{
	double theta[6];
	
	// Compute the gradient
	Invariants::gradientK1(inTensor, theta);

	// Compute the invariants
	double k3 = Invariants::invariantK3(inTensor);
	double k2 = Invariants::invariantK2(inTensor);

	// Pre-compute some values
	double sqrt6 = sqrt(6.0);
	double tsqrt6 = 3.0 * sqrt6;

	// Compute the output tensor
	for(int i = 0; i < 6; ++i) 
	{
		outTensor[i] = (tsqrt6 * theta[i] * theta[i] - 3 * k3 * theta[i] - sqrt6) / k2;
	}
}


//-------------------------[ normalizedGradientK ]-------------------------\\

void normalizedGradientK(int i, double * inTensor, double * outTensor)
{
	// Compute one of the gradients, depending on the value of "i"
	switch(i)
	{
		case K1:	gradientK1(inTensor, outTensor);		break;
		case K2:	gradientK2(inTensor, outTensor);		break;
		case K3:	gradientK3(inTensor, outTensor);		break;
		
		default:
			cerr << "ERROR: unknown invariant index!" << endl;
	}
		
	// Compute the norm of the gradient tensor
	double norm = tensorNorm(outTensor);

	// Divide the gradient by the norm
	vtkTensorMath::DivisionWithScalar(outTensor, norm, outTensor, 6);
}


//---------------------------[ rotationTangent1 ]--------------------------\\

void rotationTangent1(double * inTensor, double * outTensor)
{
	// Compute the eigensystem of the input tensor
	double W[3];   
	double V[9];   
	
	if (!(vtkTensorMath::EigenSystemSorted(inTensor, V, W)))
	{
		cerr << "rotationTangent1: Could not compute eigensystem!" << endl;
		return;
	}

	// Compute the vector product of the second and third eigenvector in both directions
	double tp23[9];
	double tp32[9];

	Invariants::vectorProduct(&V[3], &V[6], tp23);
	Invariants::vectorProduct(&V[6], &V[3], tp32);

	// Difference between second and third eigenvalue
	double difference = W[1] - W[2];

	double sum[9];

	// Add the two vector product tensors, and multiply them by the difference
	vtkTensorMath::Sum(tp23, tp32, sum, 9);
	vtkTensorMath::ProductWithScalar(sum, difference, sum, 9);
	vtkTensorMath::Tensor9To6(sum, outTensor);
}


//---------------------------[ rotationTangent2 ]--------------------------\\

void rotationTangent2(double * inTensor, double * outTensor)
{
	// Compute the eigensystem of the input tensor
	double W[3];   
	double V[9];   

	if (!(vtkTensorMath::EigenSystemSorted(inTensor, V, W)))
	{
		cerr << "rotationTangent1: Could not compute eigensystem!" << endl;
		return;
	}

	// Compute the vector product of the first and third eigenvector in both directions
	double tp13[9];
	double tp31[9];

	Invariants::vectorProduct(&V[0], &V[6], tp13);
	Invariants::vectorProduct(&V[6], &V[0], tp31);

	// Difference between first and third eigenvalue
	double difference = W[0] - W[2];

	double sum[9];

	// Add the two vector product tensors, and multiply them by the difference
	vtkTensorMath::Sum(tp13, tp31, sum, 9);
	vtkTensorMath::ProductWithScalar(sum, difference, sum, 9);
	vtkTensorMath::Tensor9To6(sum, outTensor);
}


//---------------------------[ rotationTangent3 ]--------------------------\\

void rotationTangent3(double * inTensor, double * outTensor)
{
	// Compute the eigensystem of the input tensor
	double W[3];   
	double V[9];   

	if (!(vtkTensorMath::EigenSystemSorted(inTensor, V, W)))
	{
		cerr << "rotationTangent1: Could not compute eigensystem!" << endl;
		return;
	}

	// Compute the vector product of the first and second eigenvector in both directions
	double tp12[9];
	double tp21[9];

	Invariants::vectorProduct(&V[0], &V[3], tp12);
	Invariants::vectorProduct(&V[3], &V[0], tp21);

	// Difference between first and second eigenvalue
	double difference = W[0] - W[1];

	double sum[9];

	// Add the two vector product tensors, and multiply them by the difference
	vtkTensorMath::Sum(tp12, tp21, sum, 9);
	vtkTensorMath::ProductWithScalar(sum, difference, sum, 9);
	vtkTensorMath::Tensor9To6(sum, outTensor);
}


//--------------------------[ rotationTangent1n ]--------------------------\\

void rotationTangent1n(double * inTensor, double * outTensor)
{
	// Compute the eigensystem of the input tensor
	double W[3];   
	double V[9];   

	if (!(vtkTensorMath::EigenSystemSorted(inTensor, V, W)))
	{
		cerr << "rotationTangent1: Could not compute eigensystem!" << endl;
		return;
	}

	// Compute the vector product of the second and third eigenvector in both directions
	double tp23[9];
	double tp32[9];

	Invariants::vectorProduct(&V[3], &V[6], tp23);
	Invariants::vectorProduct(&V[6], &V[3], tp32);

	double sum[9];

	// Add the two vector product tensors, and divide them by the square root of two
	vtkTensorMath::Sum(tp23, tp32, sum, 9);
	vtkTensorMath::DivisionWithScalar(sum, sqrt(2.0), sum, 9);
	vtkTensorMath::Tensor9To6(sum, outTensor);
}


//--------------------------[ rotationTangent2n ]--------------------------\\

void rotationTangent2n(double * inTensor, double * outTensor)
{
	// Compute the eigensystem of the input tensor
	double W[3];   
	double V[9];   

	if (!(vtkTensorMath::EigenSystemSorted(inTensor, V, W)))
	{
		cerr << "rotationTangent1: Could not compute eigensystem!" << endl;
		return;
	}

	// Compute the vector product of the first and third eigenvector in both directions
	double tp13[9];
	double tp31[9];

	Invariants::vectorProduct(&V[0], &V[6], tp13);
	Invariants::vectorProduct(&V[6], &V[0], tp31);

	double sum[9];

	// Add the two vector product tensors, and divide them by the square root of two
	vtkTensorMath::Sum(tp13, tp31, sum, 9);
	vtkTensorMath::DivisionWithScalar(sum, sqrt(2.0), sum, 9);
	vtkTensorMath::Tensor9To6(sum, outTensor);
}


//--------------------------[ rotationTangent3n ]--------------------------\\

void rotationTangent3n(double * inTensor, double * outTensor)
{
	// Compute the eigensystem of the input tensor
	double W[3];   
	double V[9];   

	if (!(vtkTensorMath::EigenSystemSorted(inTensor, V, W)))
	{
		cerr << "rotationTangent1: Could not compute eigensystem!" << endl;
		return;
	}

	// Compute the vector product of the first and second eigenvector in both directions
	double tp12[9];
	double tp21[9];

	Invariants::vectorProduct(&V[0], &V[3], tp12);
	Invariants::vectorProduct(&V[3], &V[0], tp21);

	double sum[9];

	// Add the two vector product tensors, and divide them by the square root of two
	vtkTensorMath::Sum(tp12, tp21, sum, 9);
	vtkTensorMath::DivisionWithScalar(sum, sqrt(2.0), sum, 9);
	vtkTensorMath::Tensor9To6(sum, outTensor);
}


//--------------------------[ doubleContraction ]--------------------------\\

double doubleContraction(double * inTensorA, double * inTensorB)
{
	// Perform element-wise multiplication of the two tensors, and compute the sum
	// of the resulting tensors. Off-diagonal elements are counted twice because
	// of the symmetry of the tensors.

	return (	(    inTensorA[0] * inTensorB[0]) + (    inTensorA[3] * inTensorB[3]) + (    inTensorA[5] * inTensorB[5]) +
				(2 * inTensorA[1] * inTensorB[1]) + (2 * inTensorA[2] * inTensorB[2]) + (2 * inTensorA[4] * inTensorB[4]) );
}


//------------------------------[ tensorNorm ]-----------------------------\\

double tensorNorm(double * inTensor)
{
	// First perform double contraction, then take the square root
	return sqrt(doubleContraction(inTensor, inTensor));
}


//----------------------------[ vectorProduct ]----------------------------\\

void vectorProduct(double * vecA, double * vecB, double * outTensor)
{
	// Compute the product between two 3-element vectors.
	// O = vecA * vecB, where vecA = [3x1] and vecB = [1x3]

	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			outTensor[i * 3 + j] = vecA[i] * vecB[j];
		}
	}
}


//------------------------------[ meanTensor ]-----------------------------\\

void meanTensor(double * inTensorA, double * inTensorB, double * outTensor)
{
	// Add the two tensors and divide the result by two
	vtkTensorMath::Sum(inTensorA, inTensorB, outTensor, 6);
	vtkTensorMath::DivisionWithScalar(outTensor, 2.0, outTensor, 6);
}


} // namespace Invariants


} // namespace bmia
