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
 * geodesicFiberNeighborhood.h
 *
 * 2011-05-25	Evert van Aart
 * - First version. 
 *
 */


#ifndef bmia_FiberTrackingPlugin_geodesicFiberNeighborhood_h
#define bmia_FiberTrackingPlugin_geodesicFiberNeighborhood_h


/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkMath.h>

/** Includes - Qt */

#include <QList>

/** Includes - Custom Files */

#include "geodesicPreProcessor.h"


namespace bmia {


/** The geodesic fiber neighborhood is used to store the data necessary for 
	geodesic fiber-tracking in a 4x4x4 voxel region around the current fiber
	position. It is divided into two regions: The full 64 voxel neighborhood,
	and the central eight voxels surrounding the current fiber position. For the
	full neighborhood, we store the (pre-processed) DTI tensors, the inverse
	tensors, scalar values (for stopping criteria), and the point IDs of the
	voxels. For the central eight points, we store only the derivatives of the
	inverse tensors in all three directions. Additionally, the neighborhood can
	interpolate scalar values and tensors, and compute the Christoffel symbols
	used for fiber-tracking. The full neighborhood arrays use 3D offset indices
	to designate the start of the slices, rows, and columns. This allows us to 
	'move' the neighborhood from one cell to the next without moving around the
	actual data, which should save some time.
*/

class geodesicFiberNeighborhood
{
	public: 

		/** Constructor. */

		geodesicFiberNeighborhood();

		/** Destructor. */

		~geodesicFiberNeighborhood();

		/** Number of steps used in computing the mobility. After every set of 
			the defined number of steps, the mobility of the fiber is computed,
			and if this mobility is below a certain threshold, the fiber-tracking
			process will terminate. See also, "computeMobility". */

		static int MOBILITY_NUMBER_OF_STEPS;

		/** Move the neighborhood in the specified direction(s). The direction
			indices can be -1 (move in negative direction), 0 (don't move), or
			1 (positive direction). In practice, movement is achieved by in- or
			decrementing the 3D array offset, and subsequently fetching new data
			from the pre-processor. Once this data has been fetched, the derivatives
			are re-computed. 
			@param dir			Direction of movement. */

		bool move(int dir[3]);

		/** Compute the mobility. Every time we move from one cell to another (using
			the "move" function"), the direction of movement is added to a list. This
			function sums up all these 3D movement vectors for the past "MOBILITY_
			NUMBER_OF_STEPS" steps, and returns the length of the sum vector. If the
			length of the sum vector is less than one, we assume that the fiber got
			"stuck" somehow - for example, it may be going in circles - and we stop
			tracking the fiber. Should be called by the tracker after every X steps. */

		double computeMobility();

		/** Set the DTI image containing the original DTI tensors. Only used to
			locate points, and to fetch the spacing and extents of the image; actual
			DTI data is fetched through the pre-processor class. 
			@param rImage		Input DTI image. */

		void setDTIImage(vtkImageData * rImage);

		/** Set the scalar array. The scalars in this array are used for the 
			scalar threshold stopping criteria. If the scalar stopping criteria 
			are disabled, this function does not need to be called; in that case,
			the class automatically uses all zeros for the scalar values.
			@param scalarImage	Input scalar image. */

		void setScalarArray(vtkImageData * scalarImage);

		/** Interpolate a scalar value, using the values for the eight central
			points stored in "scalars" and the input weights.
			@param weights		Array of eight interpolation weights. */

		double interpolateScalar(double * weights);

		/** Compute the Christoffel symbols, which are used for geodesic fiber-tracking.
			First interpolates the (pre-processed) DTI tensors and the derivatives
			of its inverse, and subsequently uses these interpolated tensors to 
			compute the symbols.
			@param weights		Array of eight interpolation weights.
			@param symbols		Array of 18 Christoffel symbols. */

		void computeChristoffelSymbols(double * weights, double * symbols);

		/** Initialize the fiber neighborhood around the specified location. First
			finds the index of the lower-left point of the neighborhood, then 
			fetches the data for this sub-volume, and finally computes the
			derivatives. Should be called after creating this class, and when
			the current fiber position suddenly moves a large distance.
			@param p			Coordinates of the current fiber point. */

		void initializeNeighborhood(double p[3]);

		/** Set the pre-processor class, which is used to fetch/compute the input tensors. 
			@param rPP			Pointer to the pre-processor. */

		void setPreProcessor(geodesicPreProcessor * rPP)
		{
			PreProcessor = rPP;
		}

		/** Enumeration for the Christoffel symbols. The first number is the superscript
			value of the symbol; the last two numbers are the two subscript values. */

		enum ChristoffelIndex
		{
			G111=0,	G112,	G122,	G113,	G123,	G133,	
			G211,	G212,	G222,	G213,	G223,	G233,	
			G311,	G312,	G322,	G313,	G323,	G333
		};

		/** Copy the base indices ("ijk") to a user-provided integer array.
			@param ijkOut	Output indices array, should be size 3. */

		void getBaseIndices(int * ijkOut)
		{
			ijkOut[0] = ijk[0];
			ijkOut[1] = ijk[1];
			ijkOut[2] = ijk[2];
		}

	private:

		double ppTensors[64][6];	/**< Pre-processed DTI tensors. */
		double invTensors[64][6];	/**< Inverted tensors. */
		double scalars[64];			/**< Scalar values. */
		int pointIds[64];			/**< Point IDs. */
		double du[8][6];			/**< Inverse tensor derivatives. */
		double dv[8][6];			/**< Inverse tensor derivatives. */
		double dw[8][6];			/**< Inverse tensor derivatives. */

		/** 3D offset indices for the 64-element arrays. The array element at
			this offset location (specifically, at "offset[2] * 16 + offset[1] * 
			4 + offset[0]" is actually element (0, 0, 0) of the 4x4x4 cube
			describing the neighborhood; the index of all other elements is
			relative to this base element, wrappin around in all three dimensions.
			This system allows us to 'move' the neighborhood without actually
			moving the data inside the arrays around. */

		int offset[3];

		/** Index within the DTI image of the lower-left voxel. For example,
			if the voxel at (0, 0, 0) (after compensating for the offset) has
			indices (3, 2, 1), it is locates at (3 * spacing[0], 2 * spacing[1],
			1 * spacing[2]), and the element at (3, 0, 0) will be located at 
			(6 * spacing[0], 2 * spacing[1], 1 * spacing[2]). */

		int ijk[3];

		/** Spacing of the DTI image. Used to compute point coordinates. */

		double spacing[3];

		/** Extents of the DTI image. Used to check if a neighborhood point lies
			within the image; if not, the point ID of a neightboring central point
			is used (all central points are always inside the image, otherwise
			the fiber has left the volume). */

		int extents[6];

		/** Indices within the 64-element arrays of the eight central voxels. 
			Computed whenever the neighborhood moves, to save some time when
			interpolating scalar values or tensors. Compensates for the offset. */

		int ci[8];

		/** DTI input image with the original DTI tensors. Only used for its
			structural information (i.e., spacing and extents); actual DTI data
			is fetched through the pre-processor. */

		vtkImageData * dtiImage;

		/** Scalar array. Scalar data can be used as an optional stopping criteria
			when performing geodesic fiber-tracking. In this case, the scalar array
			will be used to fetch these values, which are then stored in the "scalars"
			array. If we do not use the scalar value stopping criteria, we do not
			need the scalar array, and its pointer can remain NULL; in this case,
			zeros are used for all scalar values. */

		vtkDataArray * scalarArray;

		/** Pre-processor, used to fetch and compute DTI tensors and their inverse. */

		geodesicPreProcessor * PreProcessor;

		/** Movement vector, expressed in cell indices. For example, when the
			neighborhood moves one cell in the positive X-direction, the movement
			vector "[1, 0, 0]" will be added to the "moveList" list. */

		struct MovementVector
		{
			int dir[3];		/**< Direction of movement in cell indices. */
		};

		/** List containing all movement vectors for the last "MOBILITY_NUMBER_OF_STEPS"
			steps. Used to compute the mobility when "computeMobility" is called; after
			computing the mobility, the list is cleared. */

		QList<MovementVector> moveList;

		/** Invalidate the data for one slice of 16 voxels, its orientation determined
			by the axis value. Invalidating data is done by settings its point ID
			to -1. The function "getNeighborhoodData" will subsequently fetch new
			data for all invalidated points. The function "initializeNeighborhood"
			invalidates all data; when moving the neighborhood in one direction,
			one slice of data is invalidated, and subsequently replaced by new data. */

		void invalidateData(int axis, int pos);

		/** Fetch new data for all invalid points in the neighborhood (i.e., all
			points with point ID equal to -1. First computes a new point ID based
			on the current values of "ijk", and then fetches the (pre-processed)
			DTI tensor and its inverse from the pre-processor. Also fetches a
			scalar value from the scalar array, if it has been set (otherwise,
			all scalar values are set to zero). */

		void getNeighborhoodData();

		/** Update the indices within the 64-element arrays that describe the eight
			central points. This function is called whenever the neighborhood is
			moved; its purpose is to pre-compute these indices once (which requires
			compensating for the 3D offset), in order to save computations when 
			interpolating scalars or tensors (for which these indices are needed). */

		void updateCenterIndices();

		/** Compute the derivatives of the inverse tensors in three dimensions.
			Derivatives are only computed for the central eight points; we require a
			full 4x4x4 cube to be able to compute these derivates (using two-sided
			derivation). Called whenever the neighborhood is moved. */

		void computeDerivatives();

		/** Compute the index within the 64-element arrays. The input indices are
			indices of the 4x4x4 voxel cube representing the neighborhood; this 
			function first adds the 3D offset indices to these input indices,
			and then maps the resulting indices to a 1D index. 
			@param i	Index in the X-dimension.
			@param j	Index in the Y-dimension.
			@param k	Index in the Z-dimension. */

		int computeIndex(int i, int j, int k);
};


} // namespace bmia


#endif // bmia_FiberTrackingPlugin_geodesicFiberNeighborhood_h
