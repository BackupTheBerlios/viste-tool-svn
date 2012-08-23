#ifndef math_Matrix3_h
#define math_Matrix3_h

#include <cassert>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>

#include "Vector3.h"

namespace math
{
	// A class-based implementation of a 3x3 component matrix.
	// Taken from the Ogre3D (www.ogre3d.org) library and fitted
	// for use with VTK.
	class Matrix3
	{
	public:
		/** Default constructor.
		@note
		It does <b>NOT</b> initialize the matrix for efficiency.
		*/
		inline Matrix3 () {};
		inline explicit Matrix3 (const double arr[3][3])
		{
			memcpy(m,arr,9*sizeof(double));
		}

		inline Matrix3 (const Matrix3& rkMatrix)
		{
			memcpy(m,rkMatrix.m,9*sizeof(double));
		}

		Matrix3 (
			double fEntry00, double fEntry01, double fEntry02,
			double fEntry10, double fEntry11, double fEntry12,
			double fEntry20, double fEntry21, double fEntry22)
		{
			m[0][0] = fEntry00;
			m[0][1] = fEntry01;
			m[0][2] = fEntry02;
			m[1][0] = fEntry10;
			m[1][1] = fEntry11;
			m[1][2] = fEntry12;
			m[2][0] = fEntry20;
			m[2][1] = fEntry21;
			m[2][2] = fEntry22;
		}

		/** Exchange the contents of this matrix with another.
		*/
		inline void swap(Matrix3& other)
		{
			std::swap(m[0][0], other.m[0][0]);
			std::swap(m[0][1], other.m[0][1]);
			std::swap(m[0][2], other.m[0][2]);
			std::swap(m[1][0], other.m[1][0]);
			std::swap(m[1][1], other.m[1][1]);
			std::swap(m[1][2], other.m[1][2]);
			std::swap(m[2][0], other.m[2][0]);
			std::swap(m[2][1], other.m[2][1]);
			std::swap(m[2][2], other.m[2][2]);
		}

		// member access, allows use of construct mat[r][c]
		inline double* operator[] (size_t iRow) const
		{
			return (double*)m[iRow];
		}

		Vector3 GetColumn (size_t iCol) const;
		void SetColumn(size_t iCol, const Vector3& vec);
		void FromAxes(const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis);

		// assignment and comparison
		inline Matrix3& operator= (const Matrix3& rkMatrix)
		{
			memcpy(m, rkMatrix.m,9*sizeof(double));
			return *this;
		}
		bool operator== (const Matrix3& rkMatrix) const;
		inline bool operator!= (const Matrix3& rkMatrix) const
		{
			return !operator==(rkMatrix);
		}

		// arithmetic operations
		Matrix3 operator+ (const Matrix3& rkMatrix) const;
		Matrix3 operator- (const Matrix3& rkMatrix) const;
		Matrix3 operator* (const Matrix3& rkMatrix) const;
		Matrix3 operator- () const;

		// matrix * vector [3x3 * 3x1 = 3x1]
		Vector3 operator* (const Vector3& rkVector) const;

		// vector * matrix [1x3 * 3x3 = 1x3]
		friend Vector3 operator* (const Vector3& rkVector, const Matrix3& rkMatrix);

		// matrix * scalar
		Matrix3 operator* (double fScalar) const;

		// scalar * matrix
		friend Matrix3 operator* (double fScalar, const Matrix3& rkMatrix);

		// utilities
		Matrix3 Transpose () const;
		bool Inverse (Matrix3& rkInverse, double fTolerance = Matrix3::EPSILON) const;
		Matrix3 Inverse (double fTolerance = Matrix3::EPSILON) const;
		double Determinant () const;

		// singular value decomposition
		void SingularValueDecomposition (Matrix3& rkL, Vector3& rkS, Matrix3& rkR) const;
		void SingularValueComposition (const Matrix3& rkL, const Vector3& rkS, const Matrix3& rkR);

		// Gram-Schmidt orthonormalization (applied to columns of rotation matrix)
		void Orthonormalize ();

		// orthogonal Q, diagonal D, upper triangular U stored as (u01,u02,u12)
		void QDUDecomposition (Matrix3& rkQ, Vector3& rkD, Vector3& rkU) const;

		double SpectralNorm () const;

		// eigensolver, matrix must be symmetric
		void EigenSolveSymmetric (double afEigenvalue[3], Vector3 akEigenvector[3]) const;

		static void TensorProduct (const Vector3& rkU, const Vector3& rkV, Matrix3& rkProduct);

		/** Function for writing to a stream.
		*/
		inline friend std::ostream& operator << ( std::ostream& o, const Matrix3& m )
		{
			o << "Matrix3(" << std::endl
				<< m[0][0] << ", " << m[0][1] << ", " << m[0][2] << ", " << std::endl
				<< m[1][0] << ", " << m[1][1] << ", " << m[1][2] << ", " << std::endl
				<< m[2][0] << ", " << m[2][1] << ", " << m[2][2] << std::endl << ")";
			return o;
		}

		static const double EPSILON;
		static const Matrix3 ZERO;
		static const Matrix3 IDENTITY;

	protected:
		// support for eigensolver
		void Tridiagonal (double afDiag[3], double afSubDiag[3]);
		bool QLAlgorithm (double afDiag[3], double afSubDiag[3]);

		// support for singular value decomposition
		static const double ms_fSvdEpsilon;
		static const unsigned int ms_iSvdMaxIterations;
		static void Bidiagonalize (Matrix3& kA, Matrix3& kL, Matrix3& kR);
		static void GolubKahanStep (Matrix3& kA, Matrix3& kL, Matrix3& kR);

		// support for spectral norm
		static double MaxCubicRoot (double afCoeff[3]);

		double m[3][3];
	};
	/** @} */
	/** @} */
}
#endif
