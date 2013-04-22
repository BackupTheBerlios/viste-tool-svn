/*
 * scalarKernel.cu
 *
 * 2011-06-05	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_FiberTracking_ScalarKernel_h
#define bmia_FiberTracking_ScalarKernel_h


//-----------------------------[ scalarKernel ]----------------------------\\

__global__ void scalarKernel(GC_outBuffer * fiberOutput, float minS, float maxS, GC_imageInfo grid, trackingParameters trP)
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

	// Current fiber point
	float3 currentPoint;	

	// Point index within the fiber buffer
	int bufferIndex = 0;

	// Interpolated scalar
	float currentScalar = 0.0f;

	// Temporary point coordinates (with spacing and 0.5f offset)
	float3 tx;

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

			// Apply grid spacing and add an offset of 0.5f; this is necessary for
			// correct addressing of textures.

			tx.x = (currentPoint.x / grid.du) + 0.5f;
			tx.y = (currentPoint.y / grid.dv) + 0.5f;
			tx.z = (currentPoint.z / grid.dw) + 0.5f;

			// Get the interpolated scalar value
			currentScalar = tex3D(textureScalar, tx.x, tx.y, tx.z);

			// If the scalar is outside of the valid range...
			if (currentScalar > maxS || currentScalar < minS)
			{
				// ...invalidate the current fiber point...
				buffer.x[bufferIndex*3+0] = -1001.0f;
				buffer.x[bufferIndex*3+1] = -1001.0f;
				buffer.x[bufferIndex*3+2] = -1001.0f;

				// ...and write the buffer back to the memory
				fiberOutput[(idx * (trP.maxIter / 8)) + outputPosition] = buffer;

				return;
			}

			// Increment the buffer index
			bufferIndex++;

		} // for [all points in a buffer]

		// If we've processed one entire buffer, first reset the index
		bufferIndex = 0;

		// Next, increment the output position, and load the next buffer
		outputPosition++;
		buffer = fiberOutput[(idx * (trP.maxIter / 8)) + outputPosition];

	} // for [all buffers]
}



#endif // bmia_FiberTracking_ScalarKernel_h