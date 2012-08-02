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


#ifndef bmia_DistanceMeasures_h
#define bmia_DistanceMeasures_h


/** Includes - C++ */

#include <assert.h>

/** Includes - VTK */

#include <vtkMath.h>

/** Includes - Custom Files */

#include "vtkTensorMath.h"
#include "AnisotropyMeasures.h"


namespace bmia {


/** This namespace contains methods for computing distance measures, i.e.,
	scalar values that quantify the distance between two tensors. All output
	measures can be assumed to be in the range 0-1. The inputs are a measure
	index, corresponding to one of the static integers defined below, and two
	tensors, passed as double pointers. We assume that the tensors are stored
	as six-element 1D array (i.e., one element per unique element of the 
	symmetric tensor. */

namespace Distance {

	/** Indices of the different distance measures. */
	static const int i_Angular 			= 0;
	static const int i_L2D 				= 1;
	static const int i_Geometric		= 2;
	static const int i_LogEuclidian		= 3;
	static const int i_KullbackLeibner	= 4;
	static const int i_sp				= 5;
	static const int i_tsp				= 6;
	static const int i_ntsp				= 7;
	static const int i_Bhattacharyya	= 8;
	static const int i_l				= 9;
	static const int i_p				= 10;
	static const int i_s				= 11;
	static const int i_pnl				= 12;
	static const int i_LI				= 13;
	static const int i_Angular2 		= 14;
	static const int i_Angular3 		= 15;

	/** Number of distance measures. */
	static const int numberOfMeasures	= 16;

	/** Long names of the distance measures. */
	static const char * longNames[] = 
	{
		"Angular Difference",
		"L2-Distance",
		"Geometric Distance",
		"Log-Euclidian Distance",
		"Kullback-Leibner",
		"Scalar product",
		"Tensor scalar product",
		"Normalized tensor scalar product",
		"Bhattacharyya",
		"Mode-dependent (linear)",
		"Mode-dependent (planar)",
		"Mode-dependent (spherical)",
		"Mode-dependent (combined)",
		"Lattice index",
		"Angular Difference 2",
		"Angular Difference 3"
	};

	/** Short names of the distance measures. */
	static const char * shortNames[] = 
	{
		"ang",
		"L2D",
		"geom",
		"LE",
		"KL",
		"sp",
		"tsp",
		"ntsp",
		"Bhat",
		"s l",
		"s p",
		"s s",
		"pnl",
		"LI",
		"ang2",
		"ang3"
	};

	/** Compute the distance between two tensor.
		@param measure	Desired distance measure.
		@param tensorA	First tensor, six components.
		@param tensorB	Second tensor, six components. */

	double computeDistance(int measure, double * tensorA, double * tensorB);

	/** Compute the specified distance measure. 
		@param tensorA	First tensor, six components.
		@param tensorB	Second tensor, six components. */

	double AngularDifference(double * tensorA, double * tensorB);
	double L2Distance(double * tensorA, double * tensorB);
	double Geometric(double * tensorA, double * tensorB);
	double LogEuclidian(double * tensorA, double * tensorB);
	double KullbackLeibner(double * tensorA, double * tensorB);
	double ScalarProduct(double * tensorA, double * tensorB);
	double TensorScalarProduct(double * tensorA, double * tensorB);
	double NormalizedTensorScalarProduct(double * tensorA, double * tensorB);
	double Bhattacharyya(double * tensorA, double * tensorB);
	double SimilarityLinear(double * tensorA, double * tensorB);
	double SimilarityPlanar(double * tensorA, double * tensorB);
	double SimilaritySpherical(double * tensorA, double * tensorB);
	double SimilarityCombined(double * tensorA, double * tensorB);
	double Lattice(double * tensorA, double * tensorB);

	/** Compute the angular difference between two eigenvectors.
		@param eigenVector	Eigenvector number. Must be 1, 2, or 3.
		@param tensorA	First tensor, six components.
		@param tensorB	Second tensor, six components. */

	double AngularDifference(int eigenVector, double * tensorA, double * tensorB);

	/** Return the absolute difference between the anisotropy measures.
		@param measure	Desired anisotropy measure.
		@param WA		Eigenvalues of the first tensor.
		@param WB		Eigenvalues of the second tensor. */

	double ScalarDifference(int measure, double * WA, double * WB);

	/** Return the absolute difference between the anisotropy measures.
		@param measure	Desired anisotropy measure.
		@param tA		First tensor, six components. 
		@param tB		Second tensor, six components. */

	double ScalarDifferenceTensor(int measure, double * tA, double * tB);

	/** Compute the product between two 3-element vectors. 
		@param vec1		First vector.
		@param vec2		Second vector.
		@param res		Output vector. */

	void VectorProduct(double vec1[3], double vec2[3], double res[3]);

	/** Return the dot product between two vectors. 
		@param vec1		First vector.
		@param vec2		Second vector. */

	double DotProduct(double vec1[3], double vec2[3]);

} // namespace Distance


} // namespace bmia


#endif // bmia_DistanceMeasures_h
