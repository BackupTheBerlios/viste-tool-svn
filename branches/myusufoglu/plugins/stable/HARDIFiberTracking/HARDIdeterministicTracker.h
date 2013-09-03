/*
 * HARDIdeterministicTracker.h
 *
 * 2011-10-13	Anna Vilanova 
 * - First version. 
 *
 * 2011-10-31 Bart van Knippenberg
 * - Added HARDI fiber tracking functionallity. 
 * - Added class MaximumFinder for finding local maxima on the ODF
 * - Adapted calculateFiber to work with HARDI data
 * - Added function for SH interpolation
 *
 * 2011-11-05 Bart van Knippenberg
 * - Improved speed by search space reduction for ODF maxima
 * - Improved fiber initialisation
 * - Removed a bug that caused premature tract termination
 * - Added overloaded function getOutput for GFA calculatian at beginning of fiber
 * 
  *  2013-03-15 Mehmet Yusufoglu, Bart Knippenberg
 * -Can process a discrete sphere data which already have Spherical Directions and Triangles arrays. 
 *  HARDIdeterministicTracker::CalculateFiberDS and MaximumFinderGetOutputDS functions were added.
 *
 *
 */


/**This version is based on the "streamlineTracker"
*/


#ifndef bmia_HARDIdeterministicTracker_h
#define bmia_HARDIdeterministicTracker_h

/** Includes - Main header */

#include "DTITool.h"


/** Includes - VTK */

#include "vtkImageData.h"
#include "vtkDataArray.h"
#include "vtkCell.h"

/** Includes - Custom Files */

#include "vtkHARDIFiberTrackingFilter.h"
#include "HARDIdeterministicTracker.h"

/** Includes - STL */

#include <vector>

//adapted version with functions to get the triangles-, unitVectors- and anglesArray
#include "vtkGeometryGlyphFromSHBuilder.h"
//needed for vector search
#include <algorithm>
//needed for random numbers
#include <cstdlib>
//needed to seed random numbers
#include <ctime> 

namespace bmia {


/** Class declarations */

class vtkHARDIFiberTrackingFilter;


/** Simple class used to store relevant information about the current
	fiber point. */
	
class HARDIstreamlinePoint
{
	public:
		double	X[3];
		double  dX[3];
		double	AI;
		double	D;
};




/** Basic deterministic tracker, that creates a streamline (fiber) by means 
	Description of the algorithm */

class HARDIdeterministicTracker
{
	public:

		/** Constructor */
		 HARDIdeterministicTracker();

		/** Destructor */
		~HARDIdeterministicTracker();

		/** Initializes the tracker. Stores supplied pointers and parameters,
			and creates and allocates the two cell arrays. 
			@param rHARDIimageData	HARDI image data
			@param rAIImageData		Anisotropy index image data
			@param rHARDIArray		HARDI array data
			@param rAIScalars		Scalars of the AI image
			@param rParentFilter	Filter that created this tracker
			@param rStepSize		Integration step length
			@param rTolerance		Tolerance, used in "FindCell" functions */

		void initializeTracker(		vtkImageData *				rHARDIimageData,
									vtkImageData *				rAIImageData,
									vtkDataArray *				rHARDIArray,
									vtkDataArray *				rAIScalars,
									vtkHARDIFiberTrackingFilter *rParentFilter,
									double						rStepSize,
									double						rTolerance		);

		/** Computes a single fiber in either the positive or negative direction.
			Points along the fibers are computed iteratively
			Points are then stored in "pointList", which at the start only contains the seed point.
			@param direction	1 for positive direction, -1 for negative 
			@param pointList	List of fiber points */

		virtual void calculateFiber(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,  int numberOfIterations, bool cLEANMAXIMA, double TRESHOLD);
		

		/** Computes a single fiber in either the positive or negative direction. Spherical Harmonics Direction Interpolation.
			Points along the fibers are computed iteratively
			Points are then stored in "pointList", which at the start only contains the seed point.
			@param direction	1 for positive direction, -1 for negative 
			@param pointList	List of fiber points */

		virtual void calculateFiberSHDI(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,  int numberOfIterations, bool cLEANMAXIMA, double TRESHOLD);
		virtual void calculateFiberSHDIUseOfflineMaximaDirections(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,  int numberOfIterations, bool cLEANMAXIMA, double TRESHOLD);
		

		void readMaximaVectorsFile();

			/** A version for Discrete Sphere data. Computes a single fiber in either the positive or negative direction. A new version of GetOutput function is used inside.
			Points along the fibers are computed iteratively
			Points are then stored in "pointList", which at the start only contains the seed point.
			@param direction	1 for positive direction, -1 for negative 
			@param pointList	List of fiber points */
		void calculateFiberDS(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,int numberOfIterations, bool CLEANMAXIMA, double TRESHOLD);
		
		double *findFunctionValue(int threshold, std::vector<double*> &anglesArray, double *weights,  vtkIntArray *trianglesArray, std::vector<int> &meshPtIndexList, std::vector<int> &maxima);
		
		double *findFunctionValueUsingMaximaFile(int threshold, std::vector<double*> &anglesArray, double *weights,  vtkIntArray *trianglesArray, std::vector<int> &meshPtIndexList, std::vector<int> &maxima);
		
		bool findFunctionValueAtPoint(double pos[3],vtkCell * currentCell, vtkIdType currentCellId, int threshold, std::vector<double*> &anglesArray, double *interpolatedVector,  vtkIntArray *trianglesArray, std::vector<int> &meshPtIndexList, std::vector<int> &maxima);

		void findFunctionValueRK4(double pos[3],vtkCell * currentCell, vtkIdType currentCellId, int threshold, std::vector<double*> &anglesArray, double *interpolatedVector,  vtkIntArray *trianglesArray, std::vector<int> &meshPtIndexList, std::vector<int> &maxima);

		void findMax2( std::vector<double> &array, std::vector<double> &maxima, std::vector<double*> &maximaunitvectors, std::vector<double *> &anglesReturn);
		/** Sets the unit vectors for this class. */ 
		void setUnitVectors(double ** unitVectors);

		// debug variables
			bool printStepInfo; // before cout
		bool breakLoop; // breaks the main looop
		// if directions and vectors of maxes are read from the file
		void FormMaxDirectionArrays(vtkImageData *maximaVolume);
	
			std::vector<vtkDoubleArray *> outUnitVectorListFromFile;
		vtkIntArray * maximaArrayFromFile ;
		// if maxima directions read from file
		int nMaximaForEachPoint;
	protected:

		vtkImageData *  HARDIimageData;	// HARDI image data
		vtkImageData *  aiImageData;	// Anisotropy index image data
		vtkDataArray *  HARDIArray;		// HARDI Array data
		vtkDataArray *  aiScalars;		// Scalars of the AI image
		
		/** Filter that created this tracker */

		vtkHARDIFiberTrackingFilter * parentFilter;

		double stepSize;	// Integration step length, copied from the GUI
		double step;		// Actual step size, depends on the cell size and fiber direction
		double tolerance;	// Tolerance, used in "FindCell" functions

		/** Streamline Points */

		HARDIstreamlinePoint currentPoint;
		HARDIstreamlinePoint prevPoint;
		HARDIstreamlinePoint nextPoint;

		double prevSegment[3];		// Previous line segment
		double newSegment[3];		// Current line segment

		/** Computes the coordinates of the next point along the fiber by means of
			a second-order Runge-Kutta step. Returns false if the intermediate
			step leaves the volume.
			@param currentCell		Grid cell containing the current point
			@param currentCellId	Index of the current cell
			@param weights			Interpolation weights for current point */

		virtual bool solveIntegrationStepSHDIRK4(vtkCell * currentCell, vtkIdType currentCellId, double * weights);

		/** Computes the coordinates of the next point along the fiber by means of
			 Eulers Algorithm. Returns false if the intermediate
			step leaves the volume.
			@param currentCell		Grid cell containing the current point
			@param currentCellId	Index of the current cell
			@param weights			Interpolation weights for current point */

		virtual bool solveIntegrationStepSHDI(vtkCell * currentCell, vtkIdType currentCellId, double * weights);

		/** Computes the coordinates of the next point along the fiber by means of
			a second-order Runge-Kutta step. Returns false if the intermediate
			step leaves the volume.
			@param currentCell		Grid cell containing the current point
			@param currentCellId	Index of the current cell
			@param weights			Interpolation weights for current point */

		virtual bool solveIntegrationStep(vtkCell * currentCell, vtkIdType currentCellId, double * weights);

		/** Interpolates (to fill in)

			@param interpolatedObject		Output interpolateSH
			@param weights					Interpolation weights
			@param currentCellHARDIData 	Tensors of surrounding grid points */

		void interpolateSH(double * interpolatedSH, double * weights,  int SHOrder);

		/** Interpolated a single scalar value, using the Anisotropy Index image and
			the supplied interpolation weights. The scalar values of the surrounding 
			grid points are stored in the "cellAIScalars" array.
			@param interpolatedScalar		Output scalar
			@param weights					Interpolation weights */

		void interpolateScalar(double * interpolatedScalar, double * weights);

		void interpolateAngles(std::vector<double *> &angles, double * weights, double *interpolatedAngle);

		void interpolateVectors(std::vector<double *> &angles, double * weights, double *interpolatedVector);
	private:

		/** Arrays containing the HARDI and AI scalar values of the
			eight voxels (grid points) surrounding the current fiber point. */

		vtkDataArray  * cellHARDIData;
		vtkDataArray  * cellAIScalars;

		 
			vtkIntArray* maximasCellFromFile;
			std::vector<vtkDataArray *> unitVectorCellListFromFile;
		
		/** Array containing the unit vectors obtained from the tesselation. */
		double ** unitVectors;
		

}; // class HARDIdeterministicTracker


} // namespace bmia


#endif
