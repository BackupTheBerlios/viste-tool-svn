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
 * geodesicFiberTracker_CUDA.h
 *
 * 2011-07-06	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberTracking_geodesicFiberTrackerCUDA_h
#define bmia_FiberTracking_geodesicFiberTrackerCUDA_h


/** Includes */

#include <cutil_inline.h>


/** Structure for 3D floating-point coordinates. */

struct GC_dim3
{
	float u;					/**< U-Component (X). */
	float v;					/**< V-Component (Y). */
	float w;					/**< W-Component (Z). */
};


/** Structure for a single fiber point, containing both its position and direction. */

struct __align__(4) GC_fiberPoint
{
	GC_dim3 x;					/**< Position of the fiber point. */
	GC_dim3 d;					/**< Fiber direction in the point. */
};


/** Output buffer for the fiber points. We group together eight points (with three
	floating-point coordinates each), since this is more efficient than reading and
	writing single points from and to the main memory. The three coordinates of
	each point are grouped together (i.e., "x[0] x[1] x[2]" is the first point). */

struct __align__(16) GC_outBuffer
{
	float x[24];				/**< Array containing coordinates for eight fiber points. */
};


/** Structure containing preprocessing parameters. */

struct GC_ppParameters
{
	float	gain;				/**< Constant tensor gain. */
	float	threshold;			/**< Scalar threshold for sharpening. */
	int		exponent;			/**< Exponent used when sharpening tensors. */
	int		sharpeningMethod;	/**< Method used for sharpening. See "preProcessingKernel.cu"
									 for possible values. */
};


/** Structure containing basic information about the DTI image. */

struct GC_imageInfo
{
	float du;					/**< Voxel distance (U). */
	float dv;					/**< Voxel distance (V). */
	float dw;					/**< Voxel distance (W). */
	int su;						/**< Number of voxels (U). */
	int sv;						/**< Number of voxels (V). */
	int sw;						/**< Number of voxels (W). */
};


/** Structure containing information about the tracking (and postprocessing) stage. */

struct trackingParameters
{
	int loadSize;				/**< Number of fibers per load. */
	int maxIter;				/**< Maximum integration steps per fiber. */
};



/** Copy data between the CPU and the GPU. Return true on success, false otherwise. 
	@param arrayExtent		Extent of the array. 
	@param direction		Direction of the data transfer (usually "cudaMemcpyHostToDevice"). 
	@param dst				Destination CUDA array. 
	@param src				Source array. 
	@param elementSize		Size in Bytes of a single element. */

bool copyMemoryToArray(cudaExtent arrayExtent, cudaMemcpyKind direction, cudaArray * dst, void * src, int elementSize);

/** Copy the contents of a 3D array to a CUDA array, both located on the GPU. 
	Used to copy 3D arrays containing the output of preprocessing kernels to
	the correct texture arrays (since kernels cannot directly write into textures.
	@param arrayExtent		Extent of the array. 
	@param d_array			Source array.
	@param dst				Destination array. */

bool copy3DArrayToArray(cudaExtent arrayExtent, cudaPitchedPtr d_array, cudaArray * dst);

/** Copy the contents of a 3D array (used here to store the output of preprocessing
	kernels) back to the CPU. Can be used to inspect the output values during
	debugging; not used in release code. 
	@param arrayExtent		Extent of the array. 
	@param d_array			Source array.
	@param dst				Destination array (on CPU). */

bool copy3DArrayToHost(cudaExtent arrayExtent, cudaPitchedPtr d_array, void * dst);

/** Initializes the GPU for CUDA. Selects the best device, check if it has enough
	memory to hold the images, and allocates and binds the textures. Returns true
	on success, and false otherwise. 
	@param grid				Information about the DTI image (dimensions and spacing). 
	@param trP				Tracking parameters (load size, maximum number of steps). */

bool GC_GPU_Init(GC_imageInfo grid, trackingParameters trP);

/** Preprocess the tensor fields. Consists a call to the preprocessing kernel (optional),
	and three calls to the derivatives kernel. If everything goes okay (in which case
	the function returns true), the six tensor textures will contain all data
	needed by the tracking stage. 
	@param grid				Information about the DTI image (dimensions and spacing). 
	@param ppP				Preprocessing parameters.
	@param h_inTensorsA		First four tensor values.
	@param h_inTensorsB		Last two tensor values, and the scalar values (element "w").
	@param doPreProc		If false, the preprocessing kernel will be skipped. */

bool GC_GPU_PreProcessing(GC_imageInfo grid, GC_ppParameters ppP, float4 * h_inTensorsA, float4 * h_inTensorsB, bool doPreProc);

/** Perform fiber tracking for a fixed amount of seed points.
	@param grid				Information about the DTI image (dimensions and spacing).
	@param step				Step size.
	@param trP				Tracking parameters (load size, maximum number of steps).
	@param h_seedPoints		Input seed point array.
	@param h_outFibers		Output fiber array. */

bool GC_GPU_Tracking(GC_imageInfo grid, float step, trackingParameters trP, GC_fiberPoint * h_seedPoints, GC_outBuffer * h_outFibers);

/** Launches the "angleKernel", which checks the computed fibers for sharp angles.
	If a sharp angle is encountered, the current fiber point is invalidated (meaning
	that the fiber now ends at that point). If no sharp angles are encountered, and
	if all "trP.maxIter" fiber points are valid (i.e., the fiber ended because the 
	maximum number of steps was reached), the kernel writes the last point and the 
	last segment to the provided array. During the next call to this function, a new
	thread can use these values to check the angle between the last segment of the
	previous fiber part, and the first segment of the new one.
	@param h_anglePrevPoint	Array containing last point/segment of previous fiber parts.
	@param h_outFibers		Output fiber array.
	@param dotThreshold		Dot product threshold used to check for sharp angles.
	@param trP				Tracking parameters (load size, maximum number of steps).
	@param lastPostProcess	If true, fibers are loaded back to the CPU after the kernel call. */

bool GC_GPU_StopAngle(GC_fiberPoint * h_anglePrevPoints, GC_outBuffer * h_outFibers, float dotThreshold, trackingParameters trP, bool lastPostProcess);

/** Terminates fibers if their 'mobility' is low. With geodesic fiber tracking, it
	is possible for fibers to start looping indefinitely; this function aims 
	terminate these fibers, to prevent a large amount of kernel calls. This function
	is not used when the Maximum Fiber Length stopping criterion is enabled. Mobility
	is computed by checking the distances between the first, middle, and last point
	of the fiber part; if all three distances are less than the threshold, the
	fiber has not moved significantly in the last "trP.maxIter" steps, and we 
	conclude that it is stuck.
	@param h_outFibers		Output fiber array.
	@param minMobility		Minimum distance. Ideally between 1 and 5 times the cell diagonal.
	@param trP				Tracking parameters (load size, maximum number of steps).
	@param lastPostProcess	If true, fibers are loaded back to the CPU after the kernel call. */

bool GC_GPU_StopMobility(GC_outBuffer * h_outFibers, float minMobility, trackingParameters trP, bool lastPostProcess);

/** Terminates fibers if the maximum length is exceeded. An array provides the
	kernel with initial distances, which are either zero (for new fibers), or 
	the combined length of all previous fiber parts. Terminates the fiber as
	soon as the maximum distance is exceeded. If the distance is not exceeded,
	and if all "trP.maxIter" fiber points are valid, the kernel writes the
	new total distance back to the initial distance array. The initial distance 
	array is stored in the same memory space as the fiber seed points. 
	@param h_outFibers		Output fiber array.
	@param initDistances	Initial distance array.
	@param maxD				Maximum distance. 
	@param trP				Tracking parameters (load size, maximum number of steps).
	@param lastPostProcess	If true, fibers are loaded back to the CPU after the kernel call. */

bool GC_GPU_StopDistance(GC_outBuffer * h_outFibers, float * initDistances, float maxD, trackingParameters trP, bool lastPostProcess);

/** Terminates fibers if the local scalar value (from the user-selected scalar image)
	is lower than a minimum threshold or higher than a maximum threshold. The input
	scalar array is written to a new texture, so that the kernel can use texture
	filtering interpolation. This new texture is destroyed at the end of this function.
	@param h_outFibers		Output fiber array.
	@param scalarArray		Array containing scalar values.
	@param grid				Information about the DTI image (dimensions and spacing).
	@param minS				Lower scalar threshold.
	@param maxS				Upper scalar threshold.
	@param trP				Tracking parameters (load size, maximum number of steps).
	@param lastPostProcess	If true, fibers are loaded back to the CPU after the kernel call. */

bool GC_GPU_StopScalar(GC_outBuffer * h_outFibers, float * scalarArray, GC_imageInfo grid, float minS, float maxS, trackingParameters trP, bool lastPostProcess);

/** Cleans up the variables used for preprocessing. */

void GC_GPU_CleanUpPP();

/** cleans up the variables used for fiber tracking. */

void GC_GPU_CleanUpTR();


#endif // bmia_FiberTracking_geodesicFiberTrackerCUDA_h
