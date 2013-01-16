/*
 * distanceKernel.cu
 *
 * 2011-06-05	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberTracking_DistanceKernel_h
#define bmia_FiberTracking_DistanceKernel_h


//---------------------------[ distanceKernel ]----------------------------\\

__global__ void distanceKernel(float * initDistances, GC_outBuffer * fiberOutput, float maxD, trackingParameters trP)
{
	// Compute the index of the fiber
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Check if the index is within range
	if (idx > (trP.loadSize - 1))
		return;

	// Current fiber buffer (eight fiber points)
	GC_outBuffer buffer;

	// Position of the current buffer in the output
	int outputPosition = 0;

	// Load the first buffer for this fiber
	buffer = fiberOutput[idx * (trP.maxIter / 8)];

	float3 prevPoint;		// Previous fiber point
	float3 currentPoint;	// Current fiber point
	float3 v;				// Vector between the two points

	// Point index within the fiber buffer
	int bufferIndex = 0;

	// Set the previous point to the first fiber point
	prevPoint.x = buffer.x[0];
	prevPoint.y = buffer.x[1];
	prevPoint.z = buffer.x[2];

	// Get the initial distance from the main memory. If this is the first part we
	// process for this fiber, this value will be zero; otherwise, it will contain
	// the total length of all previously processed fiber parts.

	float D = initDistances[idx];

	// Loop through all output buffers
	while (outputPosition < (trP.maxIter / 8))
	{
		// Loop through all eight points in a buffer
		while (bufferIndex < 8)
		{
			// Load the current fiber point
			currentPoint.x = buffer.x[bufferIndex*3+0];
			currentPoint.y = buffer.x[bufferIndex*3+1];
			currentPoint.z = buffer.x[bufferIndex*3+2];

			// Stop if we find an invalid fiber point
			if (currentPoint.x < -1000.0f && currentPoint.y < -1000.0f && currentPoint.z < -1000.0f)
				return;

			// Compute the segment between the current and previous points
			v.x = currentPoint.x - prevPoint.x;
			v.y = currentPoint.y - prevPoint.y;
			v.z = currentPoint.z - prevPoint.z;

			// Add the segment length to the total distance
			D += sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

			// If the maximum distance is exceeded...
			if (D > maxD)
			{
				// ...invalidate the current fiber point...
				buffer.x[bufferIndex*3+0] = -1001.0f;
				buffer.x[bufferIndex*3+1] = -1001.0f;
				buffer.x[bufferIndex*3+2] = -1001.0f;

				// ...and write the buffer back to the memory
				fiberOutput[(idx * (trP.maxIter / 8)) + outputPosition] = buffer;

				return;
			}

			// Update the previous point
			prevPoint.x = currentPoint.x;
			prevPoint.y = currentPoint.y;
			prevPoint.z = currentPoint.z;

			// Increment the buffer index
			bufferIndex++;

		} // for [all points in a buffer]

		// If we've processed one entire buffer, first reset the index
		bufferIndex = 0;

		// Next, increment the output position, and load the next buffer
		outputPosition++;
		buffer = fiberOutput[(idx * (trP.maxIter / 8)) + outputPosition];

	} // for [all buffers]

	// We've processed all points without exceeding the maximum distance and without
	// encountering an invalid fiber point. Write the current distance to the main
	// memory; when the next part of the fiber is processed, this value will be used
	// as the initial distance.

	initDistances[idx] = D;
}


#endif // bmia_FiberTracking_DistanceKernel_h
