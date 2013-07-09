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


/** Class for detecting the maxima in a glyph */


class MaximumFinder
{
 
public:
 
  //constructor
  MaximumFinder(vtkIntArray* trianglesArray); 
  //destructor
  ~MaximumFinder();

  /** calculate the maxima 
  @param pDarraySH		array with SH coefficients
  @param shOrder		order of the SH
  @param treshold		treshold for the maximum detection
  @param anglesArray	array with the tesselation points in spherical coordinates
  @param output			output array with the Id's of local maxima
  @param input			input array with point id's in angle range
  */
  void getOutput(double* pDarraySH, int shOrder, double treshold, std::vector<double*> anglesArray,  std::vector<int> &output, std::vector<int> &input);

  /** overloaded function needed for GFA calculation. DS means discrete sphere data version.
  @param pDarraySH		array with SH coefficients
  @param shOrder		order of the SH
  @param anglesArray	array with the tesselation points in spherical coordinates
  */
  void getOutputDS(double* pDarraySH, int shOrder, std::vector<double*> anglesArray);

  /** calculate the maxima for Discrete Sphere data calculations. 
  @param pDarraySH		array with SH coefficients
  @param shOrder		order of the SH
  @param treshold		treshold for the maximum detection
  @param anglesArray	array with the tesselation points in spherical coordinates
  @param output			output array with the Id's of local maxima
  @param input			input array with point id's in angle range
  */
  void getOutputDS(double* pDarraySH, int shOrder, double treshold, std::vector<double*> anglesArray,  std::vector<int> &output, std::vector<int> &input);

  /** overloaded function needed for GFA calculation
  @param pDarraySH		array with SH coefficients
  @param shOrder		order of the SH
  @param anglesArray	array with the tesselation points in spherical coordinates
  */
  void getOutput(double* pDarraySH, int shOrder, std::vector<double*> anglesArray);

  /** clean the maxima
  function to calculate the average direction at double and triple maxima
  @param output							array with the Id's of local maxima (obtained from getOutput)
  @param outputlistwithunitvectors		array with the resulting directions in unit vectors
  @param pDarraySH						array with SH coefficients
  @param ODFlist						array with the corresponding ODF values
  @param unitVectors					array with known unit vectors (tesselation points)
  @param anglesArray					array with the tesselation points in spherical coordinates
  */
  void cleanOutput(std::vector<int> output, std::vector<double *>& outputlistwithunitvectors,double* pDarraySH, std::vector<double> &ODFlist,double** unitVectors, std::vector<double*> &anglesArray);

  /** calculate the GFA
  function to calculate the average direction at double and triple maxima
  can be replaced by AI dataset (HARDI measures)
  @param GFA		the calculated GFA
  */
  void getGFA(double* GFA);
  
  //vector with normalized radii of the ODF
  std::vector<double> radii_norm;

  //vector with radii of the ODF
  std::vector<double> radii;

private:
	
	//array with triangles, needed for neighbor-search
	vtkIntArray * trianglesArray;

	/** get neighbors of a point at given depth
	@param i					Id of the point
	@param depth				depth of the neighborhood search
	@param neighborlist_final	list with Id's of i's neighbors
	*/
	void MaximumFinder::getNeighbors(int i, int depth, std::vector<int> &neighborlist_final);

	/** get neighbors at depth = 1
	this function is only used by getNeighbors
	@param seedpoints			list with points to get neighbors from
	@param temp_neighbors		temporary list with direct neighbors
	*/
	void getDirectNeighbors(std::vector<int> seedpoints, std::vector<int> &temp_neighbors);
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
		

			/** A version for Discrete Sphere data. Computes a single fiber in either the positive or negative direction. A new version of GetOutput function is used inside.
			Points along the fibers are computed iteratively
			Points are then stored in "pointList", which at the start only contains the seed point.
			@param direction	1 for positive direction, -1 for negative 
			@param pointList	List of fiber points */
		void calculateFiberDS(int direction, std::vector<HARDIstreamlinePoint> * pointList, std::vector<double*> &anglesArray, vtkIntArray * trianglesArray,int numberOfIterations, bool CLEANMAXIMA, double TRESHOLD);

		double distanceSpherical(double *point1, double *point2, int n); 
		void findMax2( std::vector<double> &array, std::vector<double> &maxima, std::vector<double*> &maximaunitvectors, std::vector<double *> &anglesReturn);
		/** Sets the unit vectors for this class. */ 
		void setUnitVectors(double ** unitVectors);

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

	private:

		/** Arrays containing the HARDI and AI scalar values of the
			eight voxels (grid points) surrounding the current fiber point. */

		vtkDataArray  * cellHARDIData;
		vtkDataArray  * cellAIScalars;

		/** Array containing the unit vectors obtained from the tesselation. */
		double ** unitVectors;

}; // class HARDIdeterministicTracker


} // namespace bmia


#endif
