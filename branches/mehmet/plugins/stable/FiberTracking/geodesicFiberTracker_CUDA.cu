/*
 * geodesicFiberTracker_CUDA.cu
 *
 * 2011-07-06	Evert van Aart
 * - First version for the DTITool3.
 *
 */


/** Includes */

#include "geodesicFiberTracker_CUDA.h"
#include <stdio.h> 


/** Textures */

texture<float4, 3, cudaReadModeElementType> textureDA;
texture<float4, 3, cudaReadModeElementType> textureDB;
texture<float4, 3, cudaReadModeElementType> textureDGDUA;
texture<float4, 3, cudaReadModeElementType> textureDGDVA;
texture<float4, 3, cudaReadModeElementType> textureDGDVB;
texture<float4, 3, cudaReadModeElementType> textureDGDWA;
texture<float,  3, cudaReadModeElementType> textureScalar;


/** CUDA Arrays (for textures) */

cudaArray * cuArrayDA;
cudaArray * cuArrayDB;
cudaArray * cuArrayDGDUA;
cudaArray * cuArrayDGDVA;
cudaArray * cuArrayDGDVB;
cudaArray * cuArrayDGDWA;
cudaArray * cuArrayScalar;


/** Various arrays on the GPU */

GC_fiberPoint * d_seedPoints;
GC_outBuffer *  d_outFibers;
cudaPitchedPtr  d_outTensorsA;
cudaPitchedPtr  d_outTensorsB;


/** CUDA Kernel Code */

#include "CUDA/preProcessingKernel.cu"
#include "CUDA/derivativesKernel.cu"
#include "CUDA/traceKernel.cu"
#include "CUDA/angleKernel.cu"
#include "CUDA/mobilityKernel.cu"
#include "CUDA/distanceKernel.cu"
#include "CUDA/scalarKernel.cu"


using namespace std;


//--------------------------[ copyMemoryToArray ]--------------------------\\

bool copyMemoryToArray(cudaExtent arrayExtent, cudaMemcpyKind direction, cudaArray * dst, void * src, int elementSize)
{
	// Create memory copy parameters
	cudaMemcpy3DParms copyParams = {0};
	copyParams.extent	= arrayExtent;
	copyParams.kind		= direction;
	copyParams.dstArray = dst;
	copyParams.srcPtr	= make_cudaPitchedPtr((void *) src, arrayExtent.width * elementSize, arrayExtent.width, arrayExtent.height);

	// Copy the data
	return (cudaMemcpy3D(&copyParams) == cudaSuccess);
}


//--------------------------[ copy3DArrayToArray ]-------------------------\\

bool copy3DArrayToArray(cudaExtent arrayExtent, cudaPitchedPtr d_array, cudaArray * dst)
{
	// Create memory copy parameters
	cudaMemcpy3DParms copyParams = {0};
	copyParams.extent	= arrayExtent;
	copyParams.kind		= cudaMemcpyDeviceToDevice;
	copyParams.srcPtr	= d_array;
	copyParams.dstArray	= dst;

	// Copy the data
	return (cudaMemcpy3D(&copyParams) == cudaSuccess);
}


//--------------------------[ copy3DArrayToHost ]--------------------------\\

bool copy3DArrayToHost(cudaExtent arrayExtent, cudaPitchedPtr d_array, void * dst)
{
	// Create memory copy parameters
	cudaMemcpy3DParms copyParams = {0};
	copyParams.extent	= arrayExtent;
	copyParams.kind		= cudaMemcpyDeviceToHost;
	copyParams.srcPtr	= d_array;
	copyParams.dstPtr	= make_cudaPitchedPtr((void *) dst, arrayExtent.width, arrayExtent.width, arrayExtent.height);

	// Copy the data
	return (cudaMemcpy3D(&copyParams) == cudaSuccess);
}


//-----------------------------[ GC_GPU_Init ]-----------------------------\\

bool GC_GPU_Init(GC_imageInfo grid, trackingParameters trP)
{
	// Initialize all pointers to NULL
	cuArrayDA			= NULL;
	cuArrayDB			= NULL;
	cuArrayDGDUA		= NULL;
	cuArrayDGDVA		= NULL;
	cuArrayDGDVB		= NULL;
	cuArrayDGDWA		= NULL;

	// Use the CUDA device with the most GFLOPS
	cudaSetDevice(cutGetMaxGflopsDeviceId());

	// Get the properties of the selected device
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, cutGetMaxGflopsDeviceId());

	// Compute the amount of memory needed to store all images
	int mem_images = grid.su * grid.sv * grid.sw * 5 * 6 * sizeof(float);

	// Compute the amount of memory needed to store the seed points
	int mem_seeds = trP.loadSize * sizeof(GC_fiberPoint);

	// Compute the amount of memory needed to store the output fibers
	int mem_output = trP.loadSize * (trP.maxIter / 8) * sizeof(GC_outBuffer);

	// Print information about the memory requirements
	printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
	printf("Executing geodesic fiber tracking algorithm on %s\n\n", devProps.name);
	printf("   Device Memory:\t%f\t(MB)\n\n", (devProps.totalGlobalMem / 1000000.0f));
	printf("   Minimal Required Memory:\n");
	printf("\t- Images\t%f\t(MB)\n", mem_images / 1000000.0f);
	printf("\t- Seed Points\t%f\t(MB)\n", mem_seeds / 1000000.0f);
	printf("\t- Output\t%f\t(MB)\n", mem_output / 1000000.0f);
	printf("\t\t\t---------------------+\n");
	printf("\t- Total\t\t%f\t(MB)\n", (mem_images + mem_seeds + mem_output) / 1000000.0f);
	printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");

	// Check if we've got enough space
	if (mem_images + mem_seeds + mem_output > devProps.totalGlobalMem)
	{
		printf("GPU does not have enough memory. \nPlease use a smaller image, or use less stream lines.\n");
		printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
		return false;
	}
	else
	{
		printf("Minimal memory requirements met. \nActual required memory may be larger due to pitched pointers.\n");
		printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
	}

	printf("Initializing CUDA\n");
	printf("   Allocating output tensor arrays... ");

	// Create extents for one-element and four-element images
	cudaExtent arrayExtent			= make_cudaExtent(grid.su, grid.sv, grid.sw);
	cudaExtent arrayExtentFloat4	= make_cudaExtent(grid.su * sizeof(float4), grid.sv, grid.sw);

	// Allocate the 3D arrays that will be used to store computed tensors (before copying them to textures)
	if (cudaMalloc3D(&d_outTensorsA, arrayExtentFloat4) != cudaSuccess || cudaMalloc3D(&d_outTensorsB, arrayExtentFloat4) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to allocate output tensor array(s), aborting execution...\n\n");
		return false;
	}

	// Initialize tensor arrays to NULL
	if (cudaMemset3D(d_outTensorsA, 0, arrayExtentFloat4) != cudaSuccess || cudaMemset3D(d_outTensorsB, 0, arrayExtentFloat4) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to initialize output tensor array(s) to zero, aborting execution...\n\n");
		return false;
	}

	printf("Done!\n");
	printf("   Creating textures... ");

	// Channel description for 4-element floating point vectors
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	// Create the textures. For each texture, we first allocate an array on the GPU,
	// and we then bind the texture to this array.

	// DA: Contains first four elements of original or pre-processed tensor

	if (cudaMalloc3DArray(&cuArrayDA, &channelDesc, arrayExtent) != cudaSuccess)
	{
		printf("Failed\nERROR: Could not allocate space for 'textureDA', aborting execution...\n\n");
		return false;
	}

	textureDA.addressMode[0]	= cudaAddressModeClamp;
	textureDA.addressMode[1]	= cudaAddressModeClamp;
	textureDA.addressMode[2]	= cudaAddressModeClamp;
	textureDA.filterMode		= cudaFilterModeLinear;
	textureDA.normalized		= false;

	if (cudaBindTextureToArray(textureDA, cuArrayDA, channelDesc) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to bind 'textureDA' to the allocated array, aborting execution...\n\n");
		return false;
	}

	// DB: Contains last two elements of original or pre-processed tensor, and first two
	//	   elements of derivative of inverse tensor in U-direction

	if (cudaMalloc3DArray(&cuArrayDB, &channelDesc, arrayExtent) != cudaSuccess)
	{
		printf("Failed\nERROR: Could not allocate space for 'textureDB', aborting execution...\n\n");
		return false;
	}

	textureDB.addressMode[0]	= cudaAddressModeClamp;
	textureDB.addressMode[1]	= cudaAddressModeClamp;
	textureDB.addressMode[2]	= cudaAddressModeClamp;
	textureDB.filterMode		= cudaFilterModeLinear;
	textureDB.normalized		= false;

	if (cudaBindTextureToArray(textureDB, cuArrayDB, channelDesc) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to bind 'textureDB' to the allocated array, aborting execution...\n\n");
		return false;
	}

	// DGDUA: Contains last four elements of derivative of inverse tensor in U-direction

	if (cudaMalloc3DArray(&cuArrayDGDUA, &channelDesc, arrayExtent) != cudaSuccess)
	{
		printf("Failed\nERROR: Could not allocate space for 'textureDGDUA', aborting execution...\n\n");
		return false;
	}

	textureDGDUA.addressMode[0]	= cudaAddressModeClamp;
	textureDGDUA.addressMode[1]	= cudaAddressModeClamp;
	textureDGDUA.addressMode[2]	= cudaAddressModeClamp;
	textureDGDUA.filterMode		= cudaFilterModeLinear;
	textureDGDUA.normalized		= false;

	if (cudaBindTextureToArray(textureDGDUA, cuArrayDGDUA, channelDesc) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to bind 'textureDGDUA' to the allocated array, aborting execution...\n\n");
		return false;
	}

	// DGDVA: Contains first four elements of derivative of inverse tensor in V-direction

	if (cudaMalloc3DArray(&cuArrayDGDVA, &channelDesc, arrayExtent) != cudaSuccess)
	{
		printf("Failed\nERROR: Could not allocate space for 'textureDGDVA', aborting execution...\n\n");
		return false;
	}

	textureDGDVA.addressMode[0]	= cudaAddressModeClamp;
	textureDGDVA.addressMode[1]	= cudaAddressModeClamp;
	textureDGDVA.addressMode[2]	= cudaAddressModeClamp;
	textureDGDVA.filterMode		= cudaFilterModeLinear;
	textureDGDVA.normalized		= false;

	if (cudaBindTextureToArray(textureDGDVA, cuArrayDGDVA, channelDesc) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to bind 'textureDGDVA' to the allocated array, aborting execution...\n\n");
		return false;
	}

	// DGDVB: Contains last two elements of derivative of inverse tensor in V-direction, and
	//        first two elements of derivative of inverse tensor in W-direction.

	if (cudaMalloc3DArray(&cuArrayDGDVB, &channelDesc, arrayExtent) != cudaSuccess)
	{
		printf("Failed\nERROR: Could not allocate space for 'textureDGDVB', aborting execution...\n\n");
		return false;
	}

	textureDGDVB.addressMode[0]	= cudaAddressModeClamp;
	textureDGDVB.addressMode[1]	= cudaAddressModeClamp;
	textureDGDVB.addressMode[2]	= cudaAddressModeClamp;
	textureDGDVB.filterMode		= cudaFilterModeLinear;
	textureDGDVB.normalized		= false;

	if (cudaBindTextureToArray(textureDGDVB, cuArrayDGDVB, channelDesc) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to bind 'textureDGDVB' to the allocated array, aborting execution...\n\n");
		return false;
	}

	// DGDWA: Contains last four elements of derivative of inverse tensor in W-direction

	if (cudaMalloc3DArray(&cuArrayDGDWA, &channelDesc, arrayExtent) != cudaSuccess)
	{
		printf("Failed\nERROR: Could not allocate space for 'textureDGDWA', aborting execution...\n\n");
		return false;
	}

	textureDGDWA.addressMode[0]	= cudaAddressModeClamp;
	textureDGDWA.addressMode[1]	= cudaAddressModeClamp;
	textureDGDWA.addressMode[2]	= cudaAddressModeClamp;
	textureDGDWA.filterMode		= cudaFilterModeLinear;
	textureDGDWA.normalized		= false;

	if (cudaBindTextureToArray(textureDGDWA, cuArrayDGDWA, channelDesc) != cudaSuccess)
	{
		printf("Failed\nERROR: Failed to bind 'textureDGDWA' to the allocated array, aborting execution...\n\n");
		return false;
	}

	printf("Done!\nInitialization sucessful\n");
	printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");

	// Done!
	return true;
}


//-------------------------[ GC_GPU_PreProcessing ]------------------------\\

bool GC_GPU_PreProcessing(GC_imageInfo grid, GC_ppParameters ppP, float4 * h_inTensorsA, float4 * h_inTensorsB, bool doPreProc)
{
	// Get the properties of the active CUDA device
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, cutGetMaxGflopsDeviceId());
 
	// By default, we want 256 threads per block (optimal configuration, according to tests)
	int targetBlockSize = 256;

	// If the target does not support 256 threads per block, find the largest power
	// of two that it does support. Most modern devices support far more than 256
	// threads per block, so this should not be an issue.

	while (targetBlockSize > devProps.maxThreadsPerBlock)
		targetBlockSize /= 2;

	// Find the limit for the thread block size in the Z-dimension. We use the minimum
	// of 1) the Z-size of the DTI image, and 2) the maximum number of threads per
	// block in the Z-dimension.

	int zThreadLimit = (grid.sw < devProps.maxThreadsDim[2]) ? (grid.sw) : (devProps.maxThreadsDim[2]);

	// Z-size of thread blocks cannot be larger than the number of threads per block
	if (zThreadLimit > targetBlockSize)
		zThreadLimit = targetBlockSize;

	// Thread block size
	dim3 ppBlockSize;

	// Initialize Z-size to one
	ppBlockSize.z = 1;

	// Find the smallest power of two which is less than the maximum Z-size
	while (ppBlockSize.z * 2 < zThreadLimit)
		ppBlockSize.z *= 2;

	// Compute the number of threads per slice (X-size * Y-size of blocks)
	int sliceSize = targetBlockSize / ppBlockSize.z;

	// If we can take the square root, the X-size and Y-size will be equal
	if (sliceSize == 1 || sliceSize == 4 || sliceSize == 16 || sliceSize == 64)
	{
		ppBlockSize.x = (int) sqrt((float) sliceSize);
		ppBlockSize.y = (int) sqrt((float) sliceSize);
	}
	// Otherwise, use one of the pre-defined configurations
	else if (sliceSize == 2)
	{
		ppBlockSize.x = 1;
		ppBlockSize.y = 2;
	}
	else if (sliceSize == 8)
	{
		ppBlockSize.x = 2;
		ppBlockSize.y = 4;
	}
	else if (sliceSize == 32)
	{
		ppBlockSize.x = 4;
		ppBlockSize.y = 8;
	}
	// This should never happen
	else
	{
		ppBlockSize.x = 2;
		ppBlockSize.y = sliceSize / 2;
	}

	// Grid dimensions
	dim3 ppGridSize;

	// Compute the X-size and Y-size of the grid, based on the block size and the image dimensions
	ppGridSize.x = (int) ceil((float) grid.su / (float) ppBlockSize.x);
	ppGridSize.y = (int) ceil((float) grid.sv / (float) ppBlockSize.y);

	// The Z-size is always one, since CUDA does not (yet) support 3D grids
	ppGridSize.z = 1;

	// Since CUDA does not support 3D grids, we need to process images with large
	// Z-size (or Z-size not a multiple of two) in multiple layers. Each of these
	// layers is processed with a seperate kernel call.

	int numberOfLayers = (int) ceil((float) grid.sw / (float) ppBlockSize.z);

	// Print the configuration that we've just determined
	printf("Image Size:\t\t\t%d %d %d\n", grid.su, grid.sv, grid.sw);
	printf("Thread Block Size:\t\t%d %d %d\n", ppBlockSize.x, ppBlockSize.y, ppBlockSize.z);
	printf("Grid Size per Layer:\t\t%d %d %d\n", ppGridSize.x, ppGridSize.y, ppGridSize.z);
	printf("Threads per Block:\t\t%d\n", ppBlockSize.x * ppBlockSize.y * ppBlockSize.z);
	printf("Number of Blocks per Layer:\t%d\n", ppGridSize.x * ppGridSize.y * ppGridSize.z);
	printf("Number of Layers:\t\t%d\n", numberOfLayers);

 	printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
	printf("Preprocessing Stage\n");
	printf("   Copying tensors to GPU... ");

	// Create the extent for the image arrays
	cudaExtent arrayExtent = make_cudaExtent(grid.su, grid.sv, grid.sw);

	// Copy the tensor arrays (which optionally contain scalar values (e.g., from 
	// an Anisotropy Index image) for preprocessing) to the GPU.

	if (!(copyMemoryToArray(arrayExtent, cudaMemcpyHostToDevice, cuArrayDA, (void *) h_inTensorsA, sizeof(float4))) ||
		!(copyMemoryToArray(arrayExtent, cudaMemcpyHostToDevice, cuArrayDB, (void *) h_inTensorsB, sizeof(float4))) )
	{
		printf("Failed\nERROR: Could not copy tensor data to the GPU, aborting execution...\n\n");
		return false;
	}

	printf("Done!\n");

	// Skip pre-processing if it is disabled in the GUI
	if (doPreProc)
	{
		printf("   Preprocessing tensors... ");

		// Loop through all layers
		for (int li = 0; li < numberOfLayers; ++li)
		{
			// Launch pre-processing kernel
			preprocKernel<<<ppGridSize, ppBlockSize>>>(d_outTensorsA, d_outTensorsB, ppP, grid.su, grid.sv, grid.sw, (float) (li * ppBlockSize.z));

			cudaThreadSynchronize();

			// Check if everything went okay
			if (cudaGetLastError() != cudaSuccess)
			{
				printf("Failed\nERROR: Pre-processing kernel failed, aborting execution...\n\n");
				return false; 
			}
		}

		// Copy the pre-processed tensors to the DA and DB arrays, so that the derivative
		// kernels can use this data through the bound texture references.

		if (!(copy3DArrayToArray(arrayExtent, d_outTensorsA, cuArrayDA)) || !(copy3DArrayToArray(arrayExtent, d_outTensorsB, cuArrayDB)))
		{
			printf("Failed\nERROR: Could not copy computed tensor field(s) to texture array(s), aborting execution...\n\n");
			return false;
		}

		printf("Done!\n");
	}

	printf("   Computing derivatives of inverse tensors (U)... ");

	// Loop through all layers
	for (int li = 0; li < numberOfLayers; ++li)
	{
		// Run the first derivatives kernel (dGdu)
		derivativesKernel<<<ppGridSize, ppBlockSize>>>(d_outTensorsA, d_outTensorsB, grid, 0, (float) (li * ppBlockSize.z));

		cudaThreadSynchronize();

		// Check if everything went okay
		if (cudaGetLastError() != cudaSuccess)
		{
			printf("Failed\nERROR: Derivatives kernel failed, aborting execution...\n\n");
			return false;
		}
	}

	// Copy results to arrays for use as texture data in tracking function
	if (!(copy3DArrayToArray(arrayExtent, d_outTensorsA, cuArrayDB)) || !(copy3DArrayToArray(arrayExtent, d_outTensorsB, cuArrayDGDUA)))
	{
		printf("Failed\nERROR: Could not copy computed tensor field(s) to texture array(s), aborting execution...\n\n");
		return false;
	}

	printf("Done!\n");
	printf("   Computing derivatives of inverse tensors (V)... ");

	// Loop through all layers
	for (int li = 0; li < numberOfLayers; ++li)
	{
		// Run the second derivatives kernel (dGdv)
		derivativesKernel<<<ppGridSize, ppBlockSize>>>(d_outTensorsA, d_outTensorsB, grid, 1, (float) (li * ppBlockSize.z));

		cudaThreadSynchronize();

		// Check if everything went okay
		if (cudaGetLastError() != cudaSuccess)
		{
			printf("Failed\nERROR: Derivatives kernel failed, aborting execution...\n\n");
			return false;
		}
	}

	// Copy results to arrays for use as texture data in tracking function
	if (!(copy3DArrayToArray(arrayExtent, d_outTensorsA, cuArrayDGDVA)) || !(copy3DArrayToArray(arrayExtent, d_outTensorsB, cuArrayDGDVB)))
	{
		printf("Failed\nERROR: Could not copy computed tensor field(s) to texture array(s), aborting execution...\n\n");
		return false;
	}

	printf("Done!\n");
	printf("   Computing derivatives of inverse tensors (W)... ");

	// Loop through all layers
	for (int li = 0; li < numberOfLayers; ++li)
	{
		// Run the third derivatives kernel (dGdw)
		derivativesKernel<<<ppGridSize, ppBlockSize>>>(d_outTensorsA, d_outTensorsB, grid, 2, (float) (li * ppBlockSize.z));

		cudaThreadSynchronize();

		// Check if everything went okay
		if (cudaGetLastError() != cudaSuccess)
		{
			printf("Failed\nERROR: Derivatives kernel failed, aborting execution...\n\n");
			return false;
		}
	}

	// Copy results to arrays for use as texture data in tracking function
	if (!(copy3DArrayToArray(arrayExtent, d_outTensorsA, cuArrayDGDVB)) || !(copy3DArrayToArray(arrayExtent, d_outTensorsB, cuArrayDGDWA)))
	{
		printf("Failed\nERROR: Could not copy computed tensor field(s) to texture array(s), aborting execution...\n\n");
		return false;
	}

	// Free the temporary output arrays
	if (d_outTensorsA.ptr)	
	{
		cudaFree(d_outTensorsA.ptr);
		d_outTensorsA.ptr = NULL;
	}

	if (d_outTensorsB.ptr)	
	{
		cudaFree(d_outTensorsB.ptr);
		d_outTensorsB.ptr = NULL;
	}

	printf("Done!\nPreprocessing sucessful!\n"); 
	printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");

	// Done!
	return true;
}


//---------------------------[ GC_GPU_Tracking ]---------------------------\\

bool GC_GPU_Tracking(GC_imageInfo grid, float step, trackingParameters trP, GC_fiberPoint * h_seedPoints, GC_outBuffer * h_outFibers)
{
	// Compute the thread and block sizes
	int threadsPerBlock = 64;
	int numberOfBlocks = (int) ceil((float) trP.loadSize / (float) threadsPerBlock);

	// Allocate and set seed point data on the device
	if (!d_seedPoints)
	{
		printf("Allocating memory for seed points... ");

		if (cudaMalloc((void**) &d_seedPoints, trP.loadSize * sizeof(GC_fiberPoint)) != cudaSuccess)
		{
			printf("Failed\nERROR: Could not allocate memory for seed points, aborting execution...\n\n");
			return false;
		}

		printf("Done!\n");
	}

	// Allocate and set output fiber array on the device
	if (!d_outFibers)
	{
		printf("Allocating memory for output fibers... "); 

		if (cudaMalloc((void**) &d_outFibers, trP.loadSize * (trP.maxIter / 8) * sizeof(GC_outBuffer)) != cudaSuccess)
		{
			printf("Failed\nERROR: Could not allocate memory for output fibers, aborting execution...\n\n");
			return false;
		}

		printf("Done!\n");
	}
   
	// Copy the seed data to the device 
	if (cudaMemcpy(d_seedPoints, h_seedPoints, sizeof(GC_fiberPoint) * trP.loadSize, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("ERROR: Could not copy seed points to the device, aborting execution...\n\n");
		return false;
	}
 
	// Launch the tracing algorithm
	traceKernel<<<numberOfBlocks, threadsPerBlock>>>(d_seedPoints, d_outFibers, grid, step, trP);

	// Check if the kernel execution succeeded
	if (cudaThreadSynchronize() != cudaSuccess)
	{
		printf("ERROR: Tracking kernel failed, aborting execution...\n\n");
		return false;
	}

	// NOTE: We do not copy the computed fibers back to the CPU yet, because we
	//       will always execute at least one postprocessing kernel (always the
	//       mobility kernel, unless the distance kernel will be used). The
	//       post-processed fibers will be copied back after the last post-
	//       processing kernel.

	// Copy the seed points back to the CPU
	if(cudaMemcpy(h_seedPoints, d_seedPoints, trP.loadSize * sizeof(GC_fiberPoint), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("ERROR: Could not copy seed points back to the CPU, aborting execution... \n\n");
		return false;
	} 

	// Done!
	return true;
}


//---------------------------[ GC_GPU_StopAngle ]--------------------------\\

bool GC_GPU_StopAngle(GC_fiberPoint * h_anglePrevPoints, GC_outBuffer * h_outFibers, float dotThreshold, trackingParameters trP, bool lastPostProcess)
{
	// Compute the thread and block sizes
	int threadsPerBlock = 64;
	int numberOfBlocks = (int) ceil((float) trP.loadSize / (float) threadsPerBlock);

	// Copy the array containing the last point coordinates and segment of the
	// previous load of fibers to the GPU. For the first part of a fiber, the
	// X-component of the coordinates will be -1001.0f.

	if (cudaMemcpy(d_seedPoints, h_anglePrevPoints, sizeof(GC_fiberPoint) * trP.loadSize, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("ERROR: Could not copy seed points to the device, aborting execution...\n\n");
		return false;
	}

	// Launch the angle stopping criterion kernel
	angleKernel<<<numberOfBlocks, threadsPerBlock>>>(d_seedPoints, d_outFibers, dotThreshold, trP);

	cudaThreadSynchronize();

	// Check if the kernel execution succeeded
	if (cudaGetLastError() != cudaSuccess)
	{
		printf("ERROR: Angle stopping criterion kernel failed, aborting execution...\n\n");
		return false;
	}

	// If this was the last postprocessing kernel, copy the fibers back to the GPU
	if (lastPostProcess)
	{
		if (cudaMemcpy(h_outFibers, d_outFibers, trP.loadSize * (trP.maxIter / 8) * sizeof(GC_outBuffer), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			printf("ERROR: Could not copy output fibers back to the CPU, aborting execution... \n\n");
			return false;
		}
	}

	// Copy the last point coordinates and segment back to the CPU
	if(cudaMemcpy(h_anglePrevPoints, d_seedPoints, trP.loadSize * sizeof(GC_fiberPoint), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("ERROR: Could not copy seed points back to the CPU, aborting execution... \n\n");
		return false;
	} 

	return true;
}


//-------------------------[ GC_GPU_StopMobility ]-------------------------\\

bool GC_GPU_StopMobility(GC_outBuffer * h_outFibers, float minMobility, trackingParameters trP, bool lastPostProcess)
{
	// Compute the thread and block sizes
	int threadsPerBlock = 64;
	int numberOfBlocks = (int) ceil((float) trP.loadSize / (float) threadsPerBlock);

	// Launch the mobility stopping criterion kernel
	mobilityKernel<<<numberOfBlocks, threadsPerBlock>>>(d_outFibers, minMobility, trP);

	cudaThreadSynchronize();

	// Check if the kernel execution succeeded
	if (cudaGetLastError() != cudaSuccess)
	{
		printf("ERROR: Mobility kernel failed, aborting execution...\n\n");
		return false;
	}

	// If this was the last postprocessing kernel, copy the fibers back to the GPU
	if (lastPostProcess)
	{
		if (cudaMemcpy(h_outFibers, d_outFibers, trP.loadSize * (trP.maxIter / 8) * sizeof(GC_outBuffer), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			printf("ERROR: Could not copy output fibers back to the CPU, aborting execution... \n\n");
			return false;
		}
	}

	return true;
}


//-------------------------[ GC_GPU_StopDistance ]-------------------------\\

bool GC_GPU_StopDistance(GC_outBuffer * h_outFibers, float * initDistances, float maxD, trackingParameters trP, bool lastPostProcess)
{
	// Compute the thread and block sizes
	int threadsPerBlock = 64;
	int numberOfBlocks = (int) ceil((float) trP.loadSize / (float) threadsPerBlock);

	// Copy the current distances to the GPU. For new fibers, the distance value 
	// will be zero; otherwise, we use the sum length of the previous fiber parts.
	// Distances are stored in the seed point array; this array is six times larger
	// than what we need, but this is no problem.

	if (cudaMemcpy(d_seedPoints, initDistances, sizeof(float) * trP.loadSize, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("ERROR: Could not copy initial distances to the device, aborting execution...\n\n");
		return false;
	}
 
	// Launch the distance stopping criterion kernel
	distanceKernel<<<threadsPerBlock, numberOfBlocks>>>((float *) d_seedPoints, d_outFibers, maxD, trP);

	cudaThreadSynchronize();

	// Check if the kernel execution succeeded
	if (cudaGetLastError() != cudaSuccess)
	{
		printf("ERROR: Distance kernel failed, aborting execution...\n\n");
		return false;
	}

	// Copy the current distances back to the CPU
	if(cudaMemcpy(initDistances, d_seedPoints, trP.loadSize * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("ERROR: Could not copy distances back to the CPU, aborting execution... \n\n");
		return false;
	} 

	// If this was the last postprocessing kernel, copy the fibers back to the GPU
	if (lastPostProcess)
	{
		if (cudaMemcpy(h_outFibers, d_outFibers, trP.loadSize * (trP.maxIter / 8) * sizeof(GC_outBuffer), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			printf("ERROR: Could not copy output fibers back to the CPU, aborting execution... \n\n");
			return false;
		}
	}

	return true;
}


//--------------------------[ GC_GPU_StopScalar ]--------------------------\\

bool GC_GPU_StopScalar(GC_outBuffer * h_outFibers, float * scalarArray, GC_imageInfo grid, float minS, float maxS, trackingParameters trP, bool lastPostProcess)
{
	// Compute the thread and block sizes
	int threadsPerBlock = 64;
	int numberOfBlocks = (int) ceil((float) trP.loadSize / (float) threadsPerBlock);

	// Create a channel description and extent for the new texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent arrayExtent = make_cudaExtent(grid.su, grid.sv, grid.sw);

	// Allocate an array for the scalars
	if (cudaMalloc3DArray(&cuArrayScalar, &channelDesc, arrayExtent) != cudaSuccess)
	{
		printf("Failed\nERROR: Could not allocate space for 'textureScalar', aborting execution...\n\n");
		cuArrayScalar = NULL;
		return false;
	}

	// Create a texture for the scalars
	textureScalar.addressMode[0]	= cudaAddressModeClamp;
	textureScalar.addressMode[1]	= cudaAddressModeClamp;
	textureScalar.addressMode[2]	= cudaAddressModeClamp;
	textureScalar.filterMode		= cudaFilterModeLinear;
	textureScalar.normalized		= false;

	if (cudaBindTextureToArray(textureScalar, cuArrayScalar, channelDesc) != cudaSuccess)
	{
		printf("ERROR: Failed to bind 'textureScalar' to the allocated array, aborting execution...\n\n");
		cudaFreeArray(cuArrayScalar);
		cuArrayScalar = NULL;
		return false;
	}

	// Copy the scalar array to the texture array
	if (!(copyMemoryToArray(arrayExtent, cudaMemcpyHostToDevice, cuArrayScalar, (void *) scalarArray, sizeof(float))))
	{
		printf("Failed\nERROR: Could not copy scalar data to the GPU, aborting execution...\n\n");
		cudaUnbindTexture(textureScalar);
		cudaFreeArray(cuArrayScalar);
		cuArrayScalar = NULL;
		return false;
	}

	// Launch the scalar stopping criterion kernel 
	scalarKernel<<<numberOfBlocks, threadsPerBlock>>>(d_outFibers, minS, maxS, grid, trP);
 
	// If this was the last postprocessing kernel, copy the fibers back to the GPU
	if (lastPostProcess)
	{
		if (cudaMemcpy(h_outFibers, d_outFibers, trP.loadSize * (trP.maxIter / 8) * sizeof(GC_outBuffer), cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			printf("ERROR: Could not copy output fibers back to the CPU, aborting execution... \n\n");
			cudaUnbindTexture(textureScalar);
			cudaFreeArray(cuArrayScalar);
			cuArrayScalar = NULL;
			return false;
		}
	}

	// Delete the texture
	cudaUnbindTexture(textureScalar);
	cudaFreeArray(cuArrayScalar);
	cuArrayScalar = NULL;

	return true;
}


//---------------------------[ GC_GPU_CleanUpPP ]--------------------------\\

void GC_GPU_CleanUpPP()
{
	// Unbind all textures
	cudaUnbindTexture(textureDA);
	cudaUnbindTexture(textureDB);
	cudaUnbindTexture(textureDGDUA);
	cudaUnbindTexture(textureDGDVA);
	cudaUnbindTexture(textureDGDVB);
	cudaUnbindTexture(textureDGDWA);

	// Free texture arrays
	if (cuArrayDA)			cudaFreeArray(cuArrayDA);
	if (cuArrayDB)			cudaFreeArray(cuArrayDB);
	if (cuArrayDGDUA)		cudaFreeArray(cuArrayDGDUA);
	if (cuArrayDGDVA)		cudaFreeArray(cuArrayDGDVA);
	if (cuArrayDGDVB)		cudaFreeArray(cuArrayDGDVB);
	if (cuArrayDGDWA)		cudaFreeArray(cuArrayDGDWA);
	if (d_outTensorsA.ptr)	cudaFree(d_outTensorsA.ptr);
	if (d_outTensorsB.ptr)	cudaFree(d_outTensorsB.ptr);

	// Reset pointers to NULL
	cuArrayDA			= NULL;
	cuArrayDB			= NULL;
	cuArrayDGDUA		= NULL;
	cuArrayDGDVA		= NULL;
	cuArrayDGDVB		= NULL;
	cuArrayDGDWA		= NULL;
	d_outTensorsA.ptr	= NULL;
	d_outTensorsB.ptr	= NULL;
}


//---------------------------[ GC_GPU_CleanUpTR ]--------------------------\\

void GC_GPU_CleanUpTR()
{
	// Free GPU arrays
	if (d_seedPoints)		cudaFree(d_seedPoints);
	if (d_outFibers)		cudaFree(d_outFibers);

	// Reset pointers to NULL
	d_seedPoints		= NULL;
	d_outFibers			= NULL;
}