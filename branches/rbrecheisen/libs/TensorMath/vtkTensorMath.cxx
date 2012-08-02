/**
 * vtkTensorMath.cxx
 * by Tim Peeters
 * 
 * 2005-01-28	Tim Peeters
 * - First version
 *
 * 2005-02-16	Tim Peeters
 * - New class vtkTensorProperties, derived from the
 *   "old" vtkExtractTensorProperties.
 *
 * 2005-02-26	Tim Peeters
 * - Added DirectionalDiffusion(t, v)
 *
 * 2005-04-22	Tim Peeters
 * - Renamed class from vtkTensorProperties to vtkTensorMath.
 *
 * 2006-03-09	Tim Peeters
 * - Added HelixAngle() function.
 *
 * 2006-12-26	Tim Peeters
 * - Remove anisotropy measures. These are now computed
 *   in AnisotropyMeasures.cxx.
 *
 * 2007-02-14	Tim Peeters
 * - Implementation of EigenSystem that does not create new arrays
 *   (is about 5% faster on my system).
 *
 * 2007-02-13	Anna Vilanova
 * - Improved the efficiency of the EigenSystem calculation. Added several functions for tensor calculation:
 *    EigenSystemToTensor, EigenSystemUnsorted, Substraction, ProductWithScalar, SquareRoot, Square
 *    Also Added a calculation of Min and Max to 0 and then the calculation and computation of the square
 *    (For efficienci is better to have it in one function)
 *
 * 2007-05-15	Tim Peeters
 * - Rename helix angle to out-of-plane component
 * - Rename calculateTensor* functions to *.
 * - Add Product(inTensor1, inTensor2, outTensor) function
 *
 * 2007-05-31	Tim Peeters
 * - Add tensor Inverse() function.
 *
 * 2007-06-04	Tim Peeters
 * - Add Log() function.
 * - Add Trace() function.
 * - Add Deviatoric() function.
 * 
 * 2007-06-07 	Paulo Rodrigues
 * - Added GeometricMean()
 * - corrected Inverse()
 * 
 * 2007-07-25	Paulo Rodrigues
 * - Inverse done by 'hand'. No need for fancy decomposition from VTK!
 *
 * 2010-09-02	Tim Peeters
 * - Remove all eigensystem computation functions except one.
 *   Its enough hassle to make sure that these are correct.
 * - Change (symmetric!) tensor representation from 9 to 6 scalar values.
 * - Remove all the functions that are not used in this plugin.
 *   Otherwise I would have to rewrite them for the 6-scalar tensor
 *   representation also.
 */

#include "vtkTensorMath.h"
#include <vtkMath.h>
#include <assert.h>

namespace bmia {

bool vtkTensorMath::EigenSystemSorted(double * tensor6,	// input
				  double * eigenvec,	// output eigenvectors are in rows and sorted
				  double * eigenval)	// output eigenvalues
{
  int i; int j;
 
  double t1[3],t2[3],t3[3];
  double *t[3];		// tensor
  t[0]=t1;t[1]=t2;t[2]=t3;
  
  double vec1[3],vec2[3],vec3[3];
  double *vec[3];		// triplet of vectors
  vec[0]=vec1;vec[1]=vec2;vec[2]=vec3;

  double vect1[3],vect2[3],vect3[3];
  double *vect[3];	// vec transposed
  vect[0]=vect1;vect[1]=vect2;vect[2]=vect3;

  //for (j=0; j < 3; j++)
  //  for(i=0;i<3;i++)
  //    t[i][j] = tensor9[i*3 + j];
	
  t[0][0] = tensor6[0];
  t[0][1] = t[1][0] = tensor6[1];
  t[0][2] = t[2][0] = tensor6[2];
  t[1][1] = tensor6[3];
  t[1][2] = t[2][1] = tensor6[4];
  t[2][2] = tensor6[5];

  // do the actual calculation
 // vtkMath::Diagonalize3x3(t, eigenval, vect);
  int result = vtkMath::Jacobi((double **)t, eigenval, (double **)vect);
  if (result == 0) return false;
  
  if (eigenval[0] < 0 || eigenval[1] < 0 || eigenval[2] < 0)
    { // negative eigenvalues. Make them positive and re-sort
      // the val and vec arrays.
//      cout<<"vtkTensorMath: NEGATIVE EIGENVALUES! SHOULD NOT HAPPEN!"<<endl;
      for (i=0; i < 3; i++)
        {
        //return false;
        if (eigenval[i]<0)
          {
          // this is nonsense.
          eigenval[i] *= -1.0;
          //for (j=0; j < 3; j++) vect[i][j] *= -1.0;
          } // if (eigenval[i]<0)
        } // for i
    }  // if
	
  // transpose the result so vectors are stored in 1 of the arrays in the matrix,
  // instead of spread over  multiple arrays, using the same index.
  // Jacobi just returns the result in an annoying/unexpected way ;) At first, I
  // assumed the vectors would be in the rows and not in the columns, so the rest of
  // this function and other functions need this matrix to be transposed.
  for (i=0; i < 3; i++) 
    for (j=0; j < 3; j++)
      {
      vec[i][j] = vect[j][i];
      }

  vtkTensorMath::Sort((double **)vec, eigenval);
  // no sorting is needed if all eigenvalues were >= 0, because then
  // Jacobi already returns them sorted.
    
  // Eigenvectors are in columns of the matrix vect
  if (result)
    {
    for (i=0; i < 3; i++)
      {
      for (j=0; j < 3; j++)
        {
        eigenvec[3*i + j] = vec[i][j];
        } // for j
      } // for i
    } // if

  if (!(eigenval[0] >= eigenval[1]))
    {
    cout<<"ev1 = "<<eigenval[0]<<endl;
    cout<<"ev2 = "<<eigenval[1]<<endl;
    cout<<"ev3 = "<<eigenval[2]<<endl;
    }
  assert( eigenval[0] >= eigenval[1] );
  assert( eigenval[1] >= eigenval[2] );
  assert( eigenval[2] >= 0 );

  return (result==1);
}

bool vtkTensorMath::EigenSystemUnsorted(double * tensor6, double * V, double * W)
{
	// Input tensor
	double T[3][3];

	// Output eigenvectors
	double eigenVectors[3][3];

	// Create a full 3x3 matrix out of the six unique tensor components
	T[0][0] = tensor6[0];	T[0][1] = tensor6[1];	T[0][2] = tensor6[2];
	T[1][0] = tensor6[1];	T[1][1] = tensor6[3];	T[1][2] = tensor6[4];
	T[2][0] = tensor6[2];	T[2][1] = tensor6[4];	T[2][2] = tensor6[5];

	// Perform diagonalization to compute the eigensystem
	vtkMath::Diagonalize3x3(T, W, eigenVectors);

	// Eigenvectors are in columns of the matrix "eigenVectors"
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			// Convert 3x3 eigenvector matrix to 1D array
			V[3 * i + j] = eigenVectors[i][j];
		}
	}

	return true;  
}

void vtkTensorMath::Sort(double** vec, double* val)
{
  // assert val is array of size 3, vec is array of size 3x3

  int i; double * tmpvec; double tmpval;
  for (int timessort = 0; timessort < 2; timessort++)
    {
    for (i=0; i < 2; i++)
      {
      if (val[i] < val[i+1])
        {
        tmpval = val[i];	tmpvec = vec[i];
        val[i] = val[i+1];	vec[i] = vec[i+1];
        val[i+1] = tmpval;	vec[i+1] = tmpvec;
        } // if
      } // for i
    } // for timessort
  tmpvec = NULL;
}

bool vtkTensorMath::IsNullTensor(double * tensor6)
{
  if (tensor6 == NULL)
    {
    return true;
    }
  //return false; // speed optimization testing.

  bool result = true; int i=0;
  while (result && i < 6)
    {
    result = result && (tensor6[i] == 0.0);
    i++;
    }

  return result;
}

double vtkTensorMath::OutOfPlaneComponent(double vector[3])
{
  // we assume vector is unit length.

  // 2*Pi is needed a few times, so let's compute it once.
  //double twoPi = 2.0 * vtkMath::DoublePi();

  //if ((vector[0] == 0.0) && (vector[1] == 0.0) && (vector[3] == 0.0)) return 0.0;

  // Since the out-of-plane component depends the out-of-xy-plane component
  // only, we only need the z-component of the vector.
  double z = vector[2];
  // -1.0 <= z <= 1.0

  z = fabs(z);

  double alpha = asin(z); // the angle
  alpha /= vtkMath::DoublePi();
  alpha *= 2.0;

  // XXX: Changed by Tim. For my paper I need 1-alpha
  // in order to generate the mid-wall of a heart.
  return alpha;
  //if (alpha == 0.0) return 1.0;
  //return 1.0-alpha;
}

int vtkTensorMath::Inverse(double * inTensor, double * inverseTensor)
{
	// Compute the determinant of the input tensor
	double det = vtkTensorMath::Determinant(inTensor);

	// Determinant should be non-zero
	if(det == 0.0)
	{
		cerr << "** Inverse: Matrix is singular, and thus not invertible!" << endl;
		memset(inverseTensor, 0.0, 6 * sizeof(double));
		return 0;
	}

	// Inverse of the determinant
	double invDet = 1.0 / det;

	// Cofactors of the tensor elements
	double cofactors[9] = {0.0, 0.0 , 0.0, 0.0, 0.0, 0.0};

	// Compute the cofactors
	cofactors[0] =   inTensor[5] * inTensor[3] - inTensor[4] * inTensor[4] ;
	cofactors[1] = -(inTensor[5] * inTensor[1] - inTensor[4] * inTensor[2]);
	cofactors[2] =   inTensor[4] * inTensor[1] - inTensor[3] * inTensor[2] ;
	cofactors[3] =   inTensor[5] * inTensor[0] - inTensor[2] * inTensor[2] ;
	cofactors[4] = -(inTensor[4] * inTensor[0] - inTensor[1] * inTensor[2]);
	cofactors[5] =   inTensor[3] * inTensor[0] - inTensor[1] * inTensor[1] ;

	// Divide the cofactors by the determinant
	vtkTensorMath::ProductWithScalar(cofactors, invDet, inverseTensor, 6);

	return 1;
}

double vtkTensorMath::Determinant(double * a)
{
	// compute determinant
	double det = 0.0;

	det =	a[1] * a[4] * a[2] + 
			a[2] * a[1] * a[4] - 
			a[0] * a[4] * a[4] - 
			a[1] * a[1] * a[5] + 
			a[0] * a[3] * a[5] -
			a[2] * a[3] * a[2];

	return det;
}

 
void vtkTensorMath::ProductWithScalar(double * inTensor, double scalar, double * outTensor, int n)
{
	// Multiply tensor by scalar
	for (int i = 0; i < n; ++i)
	{
		outTensor[i] = inTensor[i] * scalar;
	}
}


void vtkTensorMath::DivisionWithScalar(double * inTensor, double scalar, double * outTensor, int n)
{
	// Divide tensor by scalar
	for (int i = 0; i < n; ++i)
	{
		outTensor[i] = inTensor[i] / scalar;
	}
}


void vtkTensorMath::SquareRoot(double * inTensor, double * outTensor)
{
	// Eigenvalues ("W") and -vectors ("V")
	double   W[3];  
	double   V[9]; 

	// Compute the (unsorted) eigensystem of the tensor
	if (!(vtkTensorMath::EigenSystemUnsorted(inTensor, V, W)))
	{
		cerr << "** SquareRoot: Failed to create eigensystem!" << endl;
		memset(outTensor, 0.0, 6 * sizeof(double));
		return;
	}

	// Compute square roots of eigenvalues
	for (int i = 0; i < 3; ++i) 
	{
		if (W[i] < 0.0)
			W[i] = 0.0;
		else
			W[i] = sqrt(W[i]);
	}

	// Convert the eigensystem back to a tensor
	vtkTensorMath::EigenSystemToTensor(W, V, outTensor);
}


void vtkTensorMath::Log(double * inTensor, double * outTensor)
{
	// Eigenvalues ("W") and -vectors ("V")
	double   W[3];  
	double   V[9]; 

	// Compute the (unsorted) eigensystem of the tensor
	if (!(vtkTensorMath::EigenSystemUnsorted(inTensor, V, W)))
	{
		cerr << "** Log: Failed to create eigensystem!" << endl;
		memset(outTensor, 0.0, 6 * sizeof(double));
		return;
	}

	// Compute square roots of eigenvalues
	for (int i = 0; i < 3; ++i) 
	{
		if (W[i] < 0.0)
			W[i] = 0.0;
		else
			W[i] = log(W[i]);
	}

	// Convert the eigensystem back to a tensor
	vtkTensorMath::EigenSystemToTensor(W, V, outTensor);
}


void vtkTensorMath::Exp(double * inTensor, double * outTensor)
{
	// Eigenvalues ("W") and -vectors ("V")
	double   W[3];  
	double   V[9]; 

	// Compute the (unsorted) eigensystem of the tensor
	if (!(vtkTensorMath::EigenSystemUnsorted(inTensor, V, W)))
	{
		cerr << "** Exp: Failed to create eigensystem!" << endl;
		memset(outTensor, 0.0, 6 * sizeof(double));
		return;
	}

	// Compute exponent of eigenvalues
	for (int i = 0; i < 3; ++i) 
	{
		W[i] = exp(W[i]);
	}

	// Convert the eigensystem back to a tensor
	EigenSystemToTensor(W, V, outTensor);
}


void vtkTensorMath::Flip(double * inTensor, double * outTensor, int axis)
{
	// Eigenvalues ("W") and -vectors ("V")
	double   W[3];  
	double   V[9]; 

	// Compute the (unsorted) eigensystem of the tensor
	if (!(vtkTensorMath::EigenSystemUnsorted(inTensor, V, W)))
	{
		cerr << "** Exp: Failed to create eigensystem!" << endl;
		memset(outTensor, 0.0, 6 * sizeof(double));
		return;
	}

	V[axis * 3 + 0] *= -1.0;
	V[axis * 3 + 1] *= -1.0;
	V[axis * 3 + 2] *= -1.0;

	// Convert the eigensystem back to a tensor
	EigenSystemToTensor(W, V, outTensor);
}

void vtkTensorMath::EigenSystemToTensor(double * W, double * V, double * outTensor)
{
	// Transposed version of eigenvectors
	double VT[9];

	// Transpose the eigenvectors
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			VT[i * 3 + j] = V[j * 3 + i];
		}
	}

	// Compute the output tensor (symmetric)
	outTensor[0] = V[0] * W[0] * VT[0] + V[1] * W[1] * VT[3] + V[2] * W[2] * VT[6];
	outTensor[1] = V[0] * W[0] * VT[1] + V[1] * W[1] * VT[4] + V[2] * W[2] * VT[7];
	outTensor[2] = V[0] * W[0] * VT[2] + V[1] * W[1] * VT[5] + V[2] * W[2] * VT[8];
	outTensor[3] = V[3] * W[0] * VT[1] + V[4] * W[1] * VT[4] + V[5] * W[2] * VT[7];
	outTensor[4] = V[3] * W[0] * VT[2] + V[4] * W[1] * VT[5] + V[5] * W[2] * VT[8];
	outTensor[5] = V[6] * W[0] * VT[2] + V[7] * W[1] * VT[5] + V[8] * W[2] * VT[8];
} 

void vtkTensorMath::Product(double * inTensor1, double * inTensor2, double * outTensor)
{
	// Compute the product of the matrices
	outTensor[0 * 3 + 0] = inTensor1[0 * 3 + 0] * inTensor2[0 * 3 + 0] + inTensor1[0 * 3 + 1] * inTensor2[1 * 3 + 0] + inTensor1[0 * 3 + 2] * inTensor2[2 * 3 + 0];
	outTensor[0 * 3 + 1] = inTensor1[0 * 3 + 0] * inTensor2[0 * 3 + 1] + inTensor1[0 * 3 + 1] * inTensor2[1 * 3 + 1] + inTensor1[0 * 3 + 2] * inTensor2[2 * 3 + 1];
	outTensor[0 * 3 + 2] = inTensor1[0 * 3 + 0] * inTensor2[0 * 3 + 2] + inTensor1[0 * 3 + 1] * inTensor2[1 * 3 + 2] + inTensor1[0 * 3 + 2] * inTensor2[2 * 3 + 2];

	outTensor[1 * 3 + 0] = inTensor1[1 * 3 + 0] * inTensor2[0 * 3 + 0] + inTensor1[1 * 3 + 1] * inTensor2[1 * 3 + 0] + inTensor1[1 * 3 + 2] * inTensor2[2 * 3 + 0];
	outTensor[1 * 3 + 1] = inTensor1[1 * 3 + 0] * inTensor2[0 * 3 + 1] + inTensor1[1 * 3 + 1] * inTensor2[1 * 3 + 1] + inTensor1[1 * 3 + 2] * inTensor2[2 * 3 + 1];
	outTensor[1 * 3 + 2] = inTensor1[1 * 3 + 0] * inTensor2[0 * 3 + 2] + inTensor1[1 * 3 + 1] * inTensor2[1 * 3 + 2] + inTensor1[1 * 3 + 2] * inTensor2[2 * 3 + 2];

	outTensor[2 * 3 + 0] = inTensor1[2 * 3 + 0] * inTensor2[0 * 3 + 0] + inTensor1[2 * 3 + 1] * inTensor2[1 * 3 + 0] + inTensor1[2 * 3 + 2] * inTensor2[2 * 3 + 0];
	outTensor[2 * 3 + 1] = inTensor1[2 * 3 + 0] * inTensor2[0 * 3 + 1] + inTensor1[2 * 3 + 1] * inTensor2[1 * 3 + 1] + inTensor1[2 * 3 + 2] * inTensor2[2 * 3 + 1];
	outTensor[2 * 3 + 2] = inTensor1[2 * 3 + 0] * inTensor2[0 * 3 + 2] + inTensor1[2 * 3 + 1] * inTensor2[1 * 3 + 2] + inTensor1[2 * 3 + 2] * inTensor2[2 * 3 + 2];
}

void vtkTensorMath::Sum(double * inTensor1, double * inTensor2, double * outTensor, int n)
{
	// Compute the sum of the two tensors
	for (int i = 0; i < n; ++i)
	{
		outTensor[i] = inTensor1[i] + inTensor2[i];
	}
}


void vtkTensorMath::Subtract(double * inTensor1, double * inTensor2, double * outTensor, int n)
{
	// Compute the subtraction of the two tensors
	for (int i = 0; i < n; ++i)
	{
		outTensor[i] = inTensor1[i] - inTensor2[i];
	}
}


double vtkTensorMath::Trace(double * inTensor)
{
	// Return sum of diagonal components
	return inTensor[0] + inTensor[3] + inTensor[5];
}


void vtkTensorMath::Deviatoric(double * inTensor, double * outTensor)
{
	// Do nothing if the tensor is NULL
	if (vtkTensorMath::IsNullTensor(inTensor))
	{
		for (int i = 0; i < 9; i++) 
		{
			outTensor[i] = 0.0;
		}

		return;
	}

	// Compute the trace of the tensor, and divide it by three
	double tr = vtkTensorMath::Trace(inTensor) / 3.0;

	// Copy the input to the output
	for (int i = 0; i < 9; i++) 
	{
		outTensor[i] = inTensor[i];
	}

	// Subtract the divided trace from the elements on the diagonal
	outTensor[0] -= tr;
	outTensor[4] -= tr;
	outTensor[8] -= tr;
}


void vtkTensorMath::Tensor6To9(double * tensor6, double * tensor9)
{
	tensor9[0] = tensor6[0];	tensor9[1] = tensor6[1];	tensor9[2] = tensor6[2]; 
	tensor9[3] = tensor6[1];	tensor9[4] = tensor6[3];	tensor9[5] = tensor6[4]; 
	tensor9[6] = tensor6[2];	tensor9[7] = tensor6[4];	tensor9[8] = tensor6[5]; 
}

void vtkTensorMath::Tensor9To6(double * tensor9, double * tensor6)
{
	tensor6[0] = tensor9[0];	tensor6[1] = tensor9[1];	tensor6[2] = tensor9[2];
								tensor6[3] = tensor9[4];	tensor6[4] = tensor9[5];
															tensor6[5] = tensor9[8];
}

} // namespace bmia
