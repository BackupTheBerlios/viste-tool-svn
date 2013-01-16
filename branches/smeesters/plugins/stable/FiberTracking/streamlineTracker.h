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
 * streamlineTracker.h
 *
 * 2010-09-17	Evert van Aart
 * - First version. 
 *
 * 2010-10-05	Evert van Aart
 * - Replaced "std::list" by "std::vector", which increases performance.
 * - Additional performance optimizations.
 *
 * 2011-03-14	Evert van Aart
 * - Fixed a bug in which it was not always detected when a fiber moved to a new
 *   voxel. Because of this, the fiber tracking process kept using the data of the
 *   old cell, resulting in fibers that kept going in areas of low anisotropy.
 *
 * 2011-03-16	Evert van Aart
 * - Fixed a bug that could cause crashes if a fiber left the volume. 
 *
 */



/** ToDo List for "streamlineTracker"
	Last updated 20-09-2010 by Evert van Aart

	- Maybe we need to check whether the Jacobi function could correctly
	  find the eigenvectors of the tensors? And if couldn't, we should 
	  probably break the tracking process. 
*/


#ifndef bmia_streamlineTracker_h
#define bmia_streamlineTracker_h

/** Includes - Main header */

#include "DTITool.h"


/** Includes - VTK */

#include "vtkImageData.h"
#include "vtkDataArray.h"
#include "vtkCell.h"

/** Includes - Custom Files */

#include "vtkFiberTrackingFilter.h"
#include "streamlineTracker.h"

/** Includes - STL */

#include <vector>


namespace bmia {


/** Class declarations */

class vtkFiberTrackingFilter;


/** Simple class used to store relevant information about the current
	fiber point. For basic fiber tracking, we store more information 
	than we need (e.g., we don't need the second and third eigenvectors),
	but since only a couple hundred of these objects exist at any given 
	time, the increase in memory requirements is not significant. Since
	other fiber tracking filters that use the "streamlineTracker" class
	may need this additional information, storing all relevant information
	increases general compatibility of "streamlineTracker". */
	
class streamlinePoint
{
	public:
		double	X[3];
		double  dX[3];
		double	V0[3];
		double	V1[3];
		double	V2[3];
		double	AI;
		double	D;
};


/** Basic streamline tracker, that creates a streamline (fiber) by means 
	of integration along the direction of the main eigenvector. Inputs 
	are the DTI tensors and the AI scalars images. For both images, we
	pass both the original "vtkImageData" pointer, and the data arrays
	containing the actual values. Tracking is done by passing a "stream-
	linePoint" list containing a seed point to the "calculateFiber" 
	function. Computed fiber points are added to this list. The stopping
	criteria are determined by the "continueTracking" function, which is
	a virtual function in the parent class (i.e., "vtkFiberTrackingFilter"
	or a class inheriting from it). A custom tracker can be made by inhe-
	riting from this class and rewriting the relevant functions. */

class streamlineTracker
{
	public:

		/** Constructor */
		 streamlineTracker();

		/** Destructor */
		~streamlineTracker();

		/** Initializes the tracker. Stores supplied pointers and parameters,
			and creates and allocates the two cell arrays. 
			@param rDTIImageData	DTI image data
			@param  rAIImageData	Anisotropy index image data
			@param rDTITensors		Tensors of the DTI image
			@param  rAIScalars		Scalars of the AI image
			@param rParentFilter	Filter that created this tracker
			@param rStepSize		Integration step length
			@param rTolerance		Tolerance, used in "FindCell" functions */

		void initializeTracker(		vtkImageData *				rDTIImageData,
									vtkImageData *				rAIImageData,
									vtkDataArray *				rDTITensors,
									vtkDataArray *				rAIScalars,
									vtkFiberTrackingFilter *	rParentFilter,
									double						rStepSize,
									double						rTolerance		);

		/** Computes a single fiber in either the positive or negative direction.
			Points along the fibers are computed iteratively by means of a second-
			order Runge-Kutta ODE solver. Points are then stored in "pointList",
			which at the start only contains the seed point.
			@param direction	1 for positive direction, -1 for negative 
			@param pointList	List of fiber points */

		virtual void calculateFiber(int direction, std::vector<streamlinePoint> * pointList);


	protected:

		vtkImageData * dtiImageData;	// DTI image data
		vtkImageData *  aiImageData;	// Anisotropy index image data
		vtkDataArray * dtiTensors;		// Tensors of the DTI image
		vtkDataArray *  aiScalars;		// Scalars of the AI image
		
		/** Filter that created this tracker */

		vtkFiberTrackingFilter * parentFilter;

		double stepSize;	// Integration step length, copied from the GUI
		double step;		// Actual step size, depends on the cell size and fiber direction
		double tolerance;	// Tolerance, used in "FindCell" functions

		/** Streamline Points */

		streamlinePoint currentPoint;
		streamlinePoint prevPoint;
		streamlinePoint nextPoint;

		double prevSegment[3];		// Previous line segment
		double newSegment[3];		// Current line segment

		/** Computes the coordinates of the next point along the fiber by means of
			a second-order Runge-Kutta step. Returns false if the intermediate
			step leaves the volume.
			@param currentCell		Grid cell containing the current point
			@param currentCellId	Index of the current cell
			@param weights			Interpolation weights for current point */

		virtual bool solveIntegrationStep(vtkCell * currentCell, vtkIdType currentCellId, double * weights);

		/** Interpolates a single tensor. Interpolation weights, which are computed at 
			an earlier point, are provided. The "vtkDataArray" contains the tensors of
			the surrounding grid points, which are used in interpolation. 
			@param interpolatedTensor		Output tensor
			@param weights					Interpolation weights
			@param currentCellDTITensors	Tensors of surrounding grid points */

		void interpolateTensor(double * interpolatedTensor, double * weights, vtkDataArray * currentCellDTITensors);

		/** Interpolated a single scalar value, using the Anisotropy Index image and
			the supplied interpolation weights. The scalar values of the surrounding 
			grid points are stored in the "cellAIScalars" array.
			@param interpolatedScalar		Output scalar
			@param weights					Interpolation weights */

		void interpolateScalar(double * interpolatedScalar, double * weights);

	private:

		/** Tensors */

		double auxTensor[6];		// Auxiliary tensor, used in interpolation
		double currentTensor[6];	// Output of interpolation process

		/** Additional Processing Variables */

		double * tempTensor[3];		// Copy of interpolated tensor, used for the Jacobi function
		double tempTensor0[3];		// First row of tensor
		double tempTensor1[3];		// Second row of tensor
		double tempTensor2[3];		// Third row of tensor
		double * eigenVectors[3];	// Matrix containing the eigenvectors
		double eigenVectors0[3];	// First row of eigenvector matrix
		double eigenVectors1[3];	// Second row of eigenvector matrix
		double eigenVectors2[3];	// Third row of eigenvector matrix
		double eigenValues[3];		// Eigenvalues of interpolated tensor

		/** Arrays containing the DTI tensors and AI scalar values of the
			eight voxels (grid points) surrounding the current fiber point. */

		vtkDataArray  * cellDTITensors;
		vtkDataArray  * cellAIScalars;

		/** Computes eigenvectors of the input tensor. 
			@param iTensor		Input tensor, six elements
			@param point		Information of current point */

		void getEigenvectors(double * iTensor, streamlinePoint * point);

		/** Ensures that the directions of the eigenvectors are consistent from
			one point to the next. Since the eigenvectors of DTI tensors are
			essentially bi-directional, we can flip them around if needed, which
			is the case when the dot product between an eigenvector in the current
			point and the same eigenvector in the previous point is negative 
			@param oldPoint		Previous fiber point
			@param newPoint		Current fiber point */

		void fixVectors(streamlinePoint * oldPoint, streamlinePoint * newPoint);

}; // class streamlineTracker


} // namespace bmia


#endif
