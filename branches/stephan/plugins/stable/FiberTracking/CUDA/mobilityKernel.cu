/*
 * mobilityKernel.cu
 *
 * 2011-06-05	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_FiberTracking_MobilityKernel_h
#define bmia_FiberTracking_MobilityKernel_h


//---------------------------[ mobilityKernel ]----------------------------\\

__global__ void mobilityKernel(GC_outBuffer * fiberOutput, float minimumMobility, trackingParameters trP)
{
	// Compute the index of the fiber
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Check if the index is within range
	if (idx > (trP.loadSize - 1))
		return;

	// Current fiber buffer (eight fiber points)
	GC_outBuffer buffer;

	// Get the first point of this part of the fiber
	buffer = fiberOutput[idx * (trP.maxIter / 8)];

	float3 firstPoint;
	firstPoint.x = buffer.x[0];
	firstPoint.y = buffer.x[1];
	firstPoint.z = buffer.x[2];

	// Get the (approximate) middle point of the fiber part
	buffer = fiberOutput[(idx * (trP.maxIter / 8)) + (trP.maxIter / 16)];

	float3 midPoint;
	midPoint.x = buffer.x[0];
	midPoint.y = buffer.x[1];
	midPoint.z = buffer.x[2];

	// Get the last point of the fiber part
	buffer = fiberOutput[(idx + 1) * (trP.maxIter / 8) - 1];

	float3 lastPoint;
	lastPoint.x = buffer.x[21];
	lastPoint.y = buffer.x[22];
	lastPoint.z = buffer.x[23];

	float3 v;
	v.x = midPoint.x - firstPoint.x;
	v.y = midPoint.y - firstPoint.y;
	v.z = midPoint.z - firstPoint.z;

	// Compute the distance between the first and middle points...
	float distanceAB = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	v.x = lastPoint.x - midPoint.x;
	v.y = lastPoint.y - midPoint.y;
	v.z = lastPoint.z - midPoint.z;

	// ...the middle and last points...
	float distanceBC = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	v.x = lastPoint.x - firstPoint.x;
	v.y = lastPoint.y - firstPoint.y;
	v.z = lastPoint.z - firstPoint.z;

	// ...and the first and last points
	float distanceAC = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	// If all three distances are less than the threshold...
	if (distanceAB < minimumMobility && distanceAC < minimumMobility && distanceBC < minimumMobility)
	{
		// ...invalidate the first point of the first buffer...
		buffer = fiberOutput[idx * (trP.maxIter / 8)];
		buffer.x[0] = -1001.0f;
		buffer.x[1] = -1001.0f;
		buffer.x[2] = -1001.0f;

		// ..and write it back to the main memory
		fiberOutput[idx * (trP.maxIter / 8)] = buffer;
	}
}


#endif // bmia_FiberTracking_MobilityKernel_h