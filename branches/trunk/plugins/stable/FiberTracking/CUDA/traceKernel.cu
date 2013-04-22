/*
 * traceKernel.cu
 *
 * 2011-06-05	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberTracking_TraceKernel_h
#define bmia_FiberTracking_TraceKernel_h



//-----------------------------[ traceKernel ]-----------------------------\\

__global__ void traceKernel(GC_fiberPoint * seedPoints, GC_outBuffer * fiberOutput, GC_imageInfo grid, float step, trackingParameters trP)
{
	// Output buffer, eight fiber points
	GC_outBuffer outBuffer;

	// Current index for the "outBuffer" array
	int bufferIndex = 1;

	// Index for the write location of the current "outBuffer" array
	int outputIndex = 0;

	// Loop index
	int i;

	// Weight, used in normilization
	float weight;

	// Christoffel symbols
	float symbols[18];

	// Temporary fiber direction, used for RK2 integration.
	float tdu, tdv, tdw;

	// Coordinates of fiber point, compensated for voxel width and texture offset
	float txu, txv, txw;

	// Six four-element vectors that contain the interpolated image data for the current fiber point.
	// 
	//	T(1,1) = iDA.x		dGdu(1,1) = iDB.z		dGdv(1,1) = iDGDVA.x	dGdw(1,1) = iDGDVB.z
	//	T(1,2) = iDA.y		dGdu(1,2) = iDB.w		dGdv(1,2) = iDGDVA.y	dGdw(1,2) = iDGDVB.w
	//	T(1,3) = iDA.z		dGdu(1,3) = iDGDUA.x	dGdv(1,3) = iDGDVA.z	dGdw(1,3) = iDGDWA.x
	//	T(2,2) = iDA.w		dGdu(2,2) = iDGDUA.y	dGdv(2,2) = iDGDVA.w	dGdw(2,2) = iDGDWA.y
	//	T(2,3) = iDB.x		dGdu(2,3) = iDGDUA.z	dGdv(2,3) = iDGDVB.x	dGdw(2,3) = iDGDWA.z
	//	T(3,3) = iDB.y		dGdu(3,3) = iDGDUA.w	dGdv(3,3) = iDGDVB.y	dGdw(3,3) = iDGDWA.w

	float4 iDA;		
	float4 iDB;
	float4 iDGDUA;
	float4 iDGDVA;	
	float4 iDGDVB;
	float4 iDGDWA;

	// Compute global index of current fiber
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Exit if we have passed the number of seed points
	if (idx > (trP.loadSize - 1))
		return;

	// Load the seed point and store it in "currentPoint"
	GC_fiberPoint currentPoint = seedPoints[idx];

	// Make the next point equal to the starting point
	GC_fiberPoint nextPoint = currentPoint;

	// Store the seed point in the output buffer
	outBuffer.x[0] = currentPoint.x.u;
	outBuffer.x[1] = currentPoint.x.v;
	outBuffer.x[2] = currentPoint.x.w;

	// Perform the integration step a number of times. The loop is broken when the fiber
	// leaves the volume, or when the maximum number of steps ("TR_MAX_ITERATIONS") is
	// reached. The maximum number of iterations allows us to control the space in global
	// memory reserved for the fiber output.

	for (i = 0; i < trP.maxIter - 1; i++)
	{
		// Load the image data for the current fiber point. The "tex3D" function automatically applies linear
		// interpolation, which reduces both the number of computational instructions and the required memory
		// space on the processors (since texture interpolation is done on dedicated hardware). The 0.5f offset
		// is due to the way the textures handle coordinates.

		txu = (currentPoint.x.u / grid.du) + 0.5f;
		txv = (currentPoint.x.v / grid.dv) + 0.5f;
		txw = (currentPoint.x.w / grid.dw) + 0.5f;

		iDA		= tex3D(textureDA,    txu, txv, txw);
		iDB		= tex3D(textureDB,	  txu, txv, txw);
		iDGDUA	= tex3D(textureDGDUA, txu, txv, txw);
		iDGDVA	= tex3D(textureDGDVA, txu, txv, txw);
		iDGDVB	= tex3D(textureDGDVB, txu, txv, txw);
		iDGDWA	= tex3D(textureDGDWA, txu, txv, txw);

		// Compute the 18 Christoffel symbols needed for the integration step.

		symbols[ 0] = 0.5 * iDA.x * iDB.z + 0.5 * iDA.y * (2 * iDB.w - iDGDVA.x) + 0.5 * iDA.z * (2 * iDGDUA.x - iDGDVB.z);
		symbols[ 1] = 0.5 * iDA.x * iDGDVA.x + 0.5 * iDA.y * iDGDUA.y + 0.5 * iDA.z * (iDGDUA.z + iDGDVA.z - iDGDVB.w);
		symbols[ 2] = 0.5 * iDA.x * (2 * iDGDVA.y - iDGDUA.y) + 0.5 * iDA.y * iDGDVA.w + 0.5 * iDA.z * (2 * iDGDVB.x - iDGDWA.y);
		symbols[ 3] = 0.5 * iDA.x * iDGDVB.z + 0.5 * iDA.y * (iDGDUA.z + iDGDVB.w - iDGDVA.z) + 0.5 * iDA.z * iDGDUA.w;
		symbols[ 4] = 0.5 * iDA.x * (iDGDVA.z + iDGDVB.w - iDGDUA.z) + 0.5 * iDA.y * iDGDWA.y + 0.5 * iDA.z * iDGDVB.y;
		symbols[ 5] = 0.5 * iDA.x * (2 * iDGDWA.x - iDGDUA.w) + 0.5 * iDA.y * (2 * iDGDWA.z - iDGDVB.y) + 0.5 * iDA.z * iDGDWA.w;

		symbols[ 6] = 0.5 * iDA.y * iDB.z + 0.5 * iDA.w * (2 * iDB.w - iDGDVA.x) + 0.5 * iDB.x * (2 * iDGDUA.x - iDGDVB.z);
		symbols[ 7] = 0.5 * iDA.y * iDGDVA.x + 0.5 * iDA.w * iDGDUA.y + 0.5 * iDB.x * (iDGDUA.z + iDGDVA.z - iDGDVB.w);
		symbols[ 8] = 0.5 * iDA.y * (2 * iDGDVA.y - iDGDUA.y) + 0.5 * iDA.w * iDGDVA.w + 0.5 * iDB.x * (2 * iDGDVB.x - iDGDWA.y);
		symbols[ 9] = 0.5 * iDA.y * iDGDVB.z + 0.5 * iDA.w * (iDGDUA.z + iDGDVB.w - iDGDVA.z) + 0.5 * iDB.x * iDGDUA.w;
		symbols[10] = 0.5 * iDA.y * (iDGDVA.z + iDGDVB.w - iDGDUA.z) + 0.5 * iDA.w * iDGDWA.y + 0.5 * iDB.x * iDGDVB.y;
		symbols[11] = 0.5 * iDA.y * (2 * iDGDWA.x - iDGDUA.w) + 0.5 * iDA.w * (2 * iDGDWA.z - iDGDVB.y) + 0.5 * iDB.x * iDGDWA.w;

		symbols[12] = 0.5 * iDA.z * iDB.z + 0.5 * iDB.x * (2 * iDGDUA.z - iDGDVA.x) + 0.5 * iDB.y * (2 * iDGDUA.x - iDGDVB.z);
		symbols[13] = 0.5 * iDA.z * iDGDVA.x + 0.5 * iDB.x * iDGDUA.y + 0.5 * iDB.y * (iDGDUA.z + iDGDVA.z - iDGDVB.w);
		symbols[14] = 0.5 * iDA.z * (2 * iDGDVA.y - iDGDUA.y) + 0.5 * iDB.x * iDGDVA.w + 0.5 * iDB.y * (2 * iDGDVB.x - iDGDWA.y);
		symbols[15] = 0.5 * iDA.z * iDGDVB.z + 0.5 * iDB.x * (iDGDUA.z + iDGDVB.w - iDGDVA.z) + 0.5 * iDB.y * iDGDUA.w;
		symbols[16] = 0.5 * iDA.z * (iDGDVA.z + iDGDVB.w - iDGDUA.z) + 0.5 * iDB.x * iDGDWA.y + 0.5 * iDB.y * iDGDVB.y;
		symbols[17] = 0.5 * iDA.z * (2 * iDGDWA.x - iDGDUA.w) + 0.5 * iDB.x * (2 * iDGDWA.z - iDGDVB.y) + 0.5 * iDB.y * iDGDWA.w;

		// Compute an average direction of the fiber, using its current direction and the Christoffel symbols. Computes
		// the direction the fiber would have if we were to jump to the middle of the vector obtained by a simple Euler
		// step (hence the 0.5f). Next, we will use this mid-point direction to compute the actual location of the next 
		// point. In this way, we implement a second-order Runge-Kutta solver.

		tdu = currentPoint.d.u + 0.5f * step * (	-(symbols[ 0] * powf(currentPoint.d.u, 2.0f))
													-(symbols[ 1] + symbols[ 1]) * currentPoint.d.u * currentPoint.d.v 
													-(symbols[ 2] * powf(currentPoint.d.v, 2.0f))
													-(symbols[ 3] + symbols[ 3]) * currentPoint.d.u * currentPoint.d.w 
													-(symbols[ 4] + symbols[ 4]) * currentPoint.d.v * currentPoint.d.w 
													-(symbols[ 5] * powf(currentPoint.d.w, 2.0f)));
		tdv = currentPoint.d.v + 0.5f * step * (	-(symbols[ 6] * powf(currentPoint.d.u, 2.0f))
													-(symbols[ 7] + symbols[ 7]) * currentPoint.d.u * currentPoint.d.v 
													-(symbols[ 8] * powf(currentPoint.d.v, 2.0f)) 
													-(symbols[ 9] + symbols[ 9]) * currentPoint.d.u * currentPoint.d.w 
													-(symbols[10] + symbols[10]) * currentPoint.d.v * currentPoint.d.w 
													-(symbols[11] * powf(currentPoint.d.w, 2.0f)));
		tdw = currentPoint.d.w + 0.5f * step * (	-(symbols[12] * powf(currentPoint.d.u, 2.0f)) 
													-(symbols[13] + symbols[13]) * currentPoint.d.u * currentPoint.d.v 
													-(symbols[14] * powf(currentPoint.d.v, 2.0f)) 
													-(symbols[15] + symbols[15]) * currentPoint.d.u * currentPoint.d.w 
													-(symbols[16] + symbols[16]) * currentPoint.d.v * currentPoint.d.w 
													-(symbols[17] * powf(currentPoint.d.w, 2.0f)));

		// Compute the location of the next point based on the direction at the mid-point.

		nextPoint.x.u = currentPoint.x.u + step * tdu;
		nextPoint.x.v = currentPoint.x.v + step * tdv;
		nextPoint.x.w = currentPoint.x.w + step * tdw;

		// Compute the new direction in this point, using the current direction, the Christoffel symbols,
		// and the direction at the mid-point. This is the second step in the RK2 mid-point solver.

		nextPoint.d.u = currentPoint.d.u + step * ( -(symbols[ 0] * powf(tdu, 2.0f))
															  -(symbols[ 1] + symbols[ 1]) * tdu * tdv 
														      -(symbols[ 2] * powf(tdv, 2.0f))
														      -(symbols[ 3] + symbols[ 3]) * tdu * tdw 
														      -(symbols[ 4] + symbols[ 4]) * tdv * tdw 
														      -(symbols[ 5] * powf(tdw, 2.0f)));
		nextPoint.d.v = currentPoint.d.v + step * ( -(symbols[ 6] * powf(tdu, 2.0f))
															  -(symbols[ 7] + symbols[ 7]) * tdu * tdv 
															  -(symbols[ 8] * powf(tdv, 2.0f)) 
															  -(symbols[ 9] + symbols[ 9]) * tdu * tdw 
															  -(symbols[10] + symbols[10]) * tdv * tdw 
															  -(symbols[11] * powf(tdw, 2.0f)));
		nextPoint.d.w = currentPoint.d.w + step * ( -(symbols[12] * powf(tdu, 2.0f)) 
															  -(symbols[13] + symbols[13]) * tdu * tdv 
															  -(symbols[14] * powf(tdv, 2.0f)) 
															  -(symbols[15] + symbols[15]) * tdu * tdw 
															  -(symbols[16] + symbols[16]) * tdv * tdw 
															  -(symbols[17] * powf(tdw, 2.0f)));

		// Checks if the fiber has left the volume
		if (nextPoint.x.u < 0 || nextPoint.x.u >= (grid.su * grid.du) || 
			nextPoint.x.v < 0 || nextPoint.x.v >= (grid.sv * grid.dv) || 
			nextPoint.x.w < 0 || nextPoint.x.w >= (grid.sw * grid.dw) 
			)
		{
			// If so, write -1 to all fields of the next point...
			nextPoint.x.u = -1001.0f;		nextPoint.d.u = -1001.0f;
			nextPoint.x.v = -1001.0f;		nextPoint.d.v = -1001.0f;
			nextPoint.x.w = -1001.0f;		nextPoint.d.w = -1001.0f;

			// ...and write it back to the seed point array in global memory. This way, we can see which fibers terminated due
			// to leaving the volume, and which terminated because they reached the maximum number of iterations. We can then
			// resume the tracking process for the incomplete fibers where we left off, and optionally replace the seed points
			// of completed fibers by new seed points.

			seedPoints[idx] = nextPoint;

			// Write the next point to the output buffer.
			outBuffer.x[bufferIndex*3+0] = -1001.0f;
			outBuffer.x[bufferIndex*3+1] = -1001.0f;
			outBuffer.x[bufferIndex*3+2] = -1001.0f;

			// Write the output buffer to the global memory. Its location is determined by the global thread index "idx", the
			// maximum number of iterations (since each "outBuffer" holds eight points, the amount of output buffer locations
			// per thread equals "(trP.maxIter / 8)"), and the output index.

			fiberOutput[idx * (trP.maxIter / 8) + outputIndex] = outBuffer;

			// The fiber has left the volume, so we're done here.
			return;
		}			

		// Write the next point to the output buffer.
		outBuffer.x[bufferIndex*3+0] = nextPoint.x.u;
		outBuffer.x[bufferIndex*3+1] = nextPoint.x.v;
		outBuffer.x[bufferIndex*3+2] = nextPoint.x.w;

		// Increment the buffer index.
		bufferIndex++;

		// Make the current point equal to the next point.
		currentPoint = nextPoint;

		// Apply normilization of the current direction.
		weight = sqrtf(currentPoint.d.u * currentPoint.d.u + currentPoint.d.v * currentPoint.d.v + currentPoint.d.w * currentPoint.d.w);
		currentPoint.d.u /= weight;
		currentPoint.d.v /= weight;
		currentPoint.d.w /= weight;

		// Once every eight iterations, dump the full output buffer to global memory.
		if (bufferIndex == 8)
		{
			// Reset the buffer index.
			bufferIndex = 0;

			// Write the output buffer to the global memory. Its location is determined by the global thread index "idx", the
			// maximum number of iterations (since each "outBuffer" holds eight points, the amount of output buffer locations
			// per thread equals "(trP.maxIter / 8)"), and the output index.

			fiberOutput[idx * (trP.maxIter / 8) + outputIndex] = outBuffer;

			// Increment the output index.
			outputIndex++;
		}
	}

	// Write the next point to the seed point array. If necessary, we can pick up the tracking process where we
	// left off by first sorting the seed points and then calling this kernel again.

	seedPoints[idx] = nextPoint;
}

#endif // bmia_FiberTracking_TraceKernel_h
