/** Class for detecting the maxima in a glyph */

#ifndef MaximumFinder_h
#define MaximumFinder_h
/** Includes - Main header */

#include "DTITool.h"


/** Includes - VTK */

#include "vtkImageData.h"
#include "vtkDataArray.h"
#include "vtkCell.h"

/** Includes - Custom Files */
#include "vtkGeometryGlyphFromSHBuilder.h"
//#include "vtkHARDIFiberTrackingFilter.h"
//#include "HARDIdeterministicTracker.h"
 
/** Includes - STL */

#include <vector>

//adapted version with functions to get the triangles-, unitVectors- and anglesArray
//#include "vtkGeometryGlyphFromSHBuilder.h"
//needed for vector search
#include <algorithm>
//needed for random numbers
#include <cstdlib>
//needed to seed random numbers
#include <ctime> 

namespace bmia {

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
  @param theMax			output id
  @param input			input array with point id's in angle range
  */
  void getUniqueOutput(double* pDarraySH, int shOrder, double treshold, std::vector<double*> anglesArray,   std::vector<int> &input, int &indexOfMax);

   /** calculate the maximum 
  @param pDarraySH		array with SH coefficients
  @param shOrder		order of the SH
  @param treshold		treshold for the maximum detection
  @param anglesArray	array with the tesselation points in spherical coordinates
  @param output			output array with the Id's of local maxima
  @param input			input array with point id's in angle range
  */
  void getOutput(double* pDarraySH, int shOrder, double treshold, std::vector<double*> anglesArray,  std::vector<int> &output, std::vector<int> &indexOfMax);


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
  void getOutputDS(double* pDarraySH, int shOrder, double treshold, std::vector<double*> anglesArray,  std::vector<int> &output, std::vector<int> &indexOfMax);

  /** overloaded function needed for GFA calculation
  @param pDarraySH		array with SH coefficients
  @param shOrder		order of the SH
  @param anglesArray	array with the tesselation points in spherical coordinates
  */
  void getOutput(double* pDarraySH, int shOrder, std::vector<double*> anglesArray);


 // void maximaToUnitVectors(,std::vector<double*> anglesArray);

  /** clean the maxima
  function to calculate the average direction at double and triple maxima
  @param output							array with the Id's of local maxima (obtained from getOutput)
  @param outputlistwithunitvectors		array with the resulting directions in unit vectors
  @param pDarraySH						array with SH coefficients
  @param ODFlist						array with the corresponding ODF values. Will be filled by the function.
  @param unitVectors					array with known unit vectors (tesselation points). Will be filled by the function.
  @param anglesArray					array with the tesselation points in spherical coordinates
  */
  void cleanOutput(std::vector<int> &indexOfMax, std::vector<double *>& outputlistwithunitvectors,double* pDarraySH, std::vector<double> &ODFlist,double** unitVectors, std::vector<double*> &anglesArray);

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

}
#endif