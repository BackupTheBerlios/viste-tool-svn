/*
 * angleKernel.cu
 *
 * 2011-06-05	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberTracking_AngleKernel_h
#define bmia_FiberTracking_AngleKernel_h


//-----------------------------[ angleKernel ]-----------------------------\\

__global__ void angleKernel(GC_fiberPoint * seedPoints, GC_outBuffer * fiberOutput, float dotThreshold, trackingParameters trP)
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
	float3 prevSegment;		// Previous segment (normalized)
	float3 currentSegment;	// Current segment (normalized)

	// Point index within the fiber buffer
	int bufferIndex;

	// Determine whether or not we should test the very first fiber point
	int firstPoint;

	// Get a 'seed point' from the memory. This is not an actual seed point; rather,
	// it either contains no relevant data (in which case "x.u" will be less than
	// -1000.0f), or it will contain the last point and segment of the previous
	// part of this fiber. This data was written by a previous instantiation of 
	// this kernel, which processed the previous part of the same fiber. This 
	// only happens if A) all "trP.maxIter" points of the previous part
	// were valid (i.e., the fiber was not stopped prematurely), and B) no sharp 
	// angles were found. 

	GC_fiberPoint seedPoint = seedPoints[idx];

	// If the U-element is less than -1000.0f, it means that this is the first part
	// we're processing for this fiber. 

	if (seedPoint.x.u < -1000.0f)
	{
		// Set the previous point to the first fiber point
		prevPoint.x = buffer.x[0];
		prevPoint.y = buffer.x[1];
		prevPoint.z = buffer.x[2];

		// Set "firstPoint" to one; this way, we'll know not to test the dot product
		// for the first fiber point (because during the first iteration of the loops
		// below, we do not yet have a valid "prevSegment").

		firstPoint = 1;

		// Set the buffer index to one (since we've already taken the first point)
		bufferIndex = 1;	
	}

	// If the U-element is not less than -1000.0f, it means that we're continuing
	// work on a fiber that was already processed in part during a previous kernel call.

	else
	{
		// In this case, we load the previous point and segment from the 'seed
		// point' (which are the last point and segment of the previous part of the fiber).

		prevPoint.x = seedPoint.x.u;
		prevPoint.y = seedPoint.x.v;
		prevPoint.z = seedPoint.x.w;
		
		prevSegment.x = seedPoint.d.u;
		prevSegment.y = seedPoint.d.v;
		prevSegment.z = seedPoint.d.w;

		// We've already got a valid "prevSegment", so we can test the very first point
		firstPoint = 0;

		// Set the buffer index to zero, since we have not yet loaded any points
		// of this part of the fiber
		bufferIndex = 0;
	}

	// Dot product, compared against the threshold value
	float testDot;

	// Norm (length) of a fiber segment, used for normalization
	float segmentNorm;

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

			// If this point is invalid, the fiber ends here (for example due to
			// leaving the volume). In response, we invalidate the 'seed point'.

			if (currentPoint.x < -1000.0f && currentPoint.y < -1000.0f && currentPoint.z < -1000.0f)
			{
				GC_fiberPoint outPoint;

				// Entries are invalid if the U-element of X is -1001.0f;
				outPoint.x.u = -1001.0f;

				// Write the invalid point to the memory
				seedPoints[idx] = outPoint;

				return;
			}

			// Compute the segment between the current and previous points
			currentSegment.x = currentPoint.x - prevPoint.x;
			currentSegment.y = currentPoint.y - prevPoint.y;
			currentSegment.z = currentPoint.z - prevPoint.z;

			// Normalize the fiber segment
			segmentNorm = sqrtf(currentSegment.x * currentSegment.x + currentSegment.y * currentSegment.y + currentSegment.z * currentSegment.z);

			if (segmentNorm != 0.0f)
			{
				currentSegment.x /= segmentNorm;
				currentSegment.y /= segmentNorm;
				currentSegment.z /= segmentNorm;
			} 

			// If "firstPoint" is one, we do not yet have a valid "prevSegment",
			// so we simply use 1.0f as the dot product (which will always pass).

			if (firstPoint == 1)
			{
				testDot = 1.0f;

				firstPoint = 0;
			}
			else
			{
				// Compute the dot product of the previous and current segments (both normalized)
				testDot = currentSegment.x * prevSegment.x + currentSegment.y * prevSegment.y + currentSegment.z * prevSegment.z;
			}

			// If the computed dot product is less than the threshold, we've got a sharp angle
			if (testDot < dotThreshold)
			{
				// Invalidate the current fiber point. This will cause the fiber 
				// to end at this point. We do not need to invalidate the points
				// after the current point, as the CPU code will stop copying
				// the fiber points as soon as it encounters an invalid point

				buffer.x[bufferIndex*3+0] = -1001.0f;
				buffer.x[bufferIndex*3+1] = -1001.0f;
				buffer.x[bufferIndex*3+2] = -1001.0f;
				fiberOutput[(idx * (trP.maxIter / 8)) + outputPosition] = buffer;

				// Invalidate the seed point. The value "2001.0f" can be used in
				// debugging to distinguish between fibers that ended due to a
				// sharp angle, and fibers that ended due to another reason (-1001.0f).

				GC_fiberPoint outPoint;
				outPoint.x.u = -2001.0f;
				seedPoints[idx] = outPoint;

				return;
			}

			// Update the previous point and segment
			prevPoint.x = currentPoint.x;
			prevPoint.y = currentPoint.y;
			prevPoint.z = currentPoint.z;

			prevSegment.x = currentSegment.x;
			prevSegment.y = currentSegment.y;
			prevSegment.z = currentSegment.z;

			// Increment the buffer index
			bufferIndex++;

		} // for [all points in a buffer]

		// If we've processed one entire buffer, first reset the index
		bufferIndex = 0;

		// Next, increment the output position, and load the next buffer
		outputPosition++;
		buffer = fiberOutput[(idx * (trP.maxIter / 8)) + outputPosition];

	} // for [all buffers]

	// Write the current point and segment to the 'seed point' array
	GC_fiberPoint outPoint;
	outPoint.x.u = currentPoint.x;
	outPoint.x.v = currentPoint.y;
	outPoint.x.w = currentPoint.z;
	outPoint.d.u = currentSegment.x;
	outPoint.d.v = currentSegment.y;
	outPoint.d.w = currentSegment.z;
	seedPoints[idx] = outPoint;
}


#endif // bmia_FiberTracking_AngleKernel_h
