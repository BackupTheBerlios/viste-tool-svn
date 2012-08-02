/*
 * preProcessingKernel.cu
 *
 * 2011-06-05	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberTracking_PreProcessingKernel_h
#define bmia_FiberTracking_PreProcessingKernel_h


// Value close to zero, used as error margin
#define PP_CLOSE_TO_ZERO 0.0001f



//----------------------------[ preprocKernel ]----------------------------\\

__global__ void preprocKernel(cudaPitchedPtr outA, cudaPitchedPtr outB, GC_ppParameters P, int su, int sv, int sw, float z_offset)
{
	// Four-element used to contain the tensors. For the input tensors, the following holds:
	//
	// txDA.x	=	T(1,1)
	// txDA.y	=	T(1,2)
	// txDA.z	=	T(1,3)
	// txDA.w	=	T(2,2)
	// txDB.x	=	T(2,3)
	// txDB.y	=	T(3,3)

	float4 txDA;	float4 txDB;	// Original + Sharpened Tensor
	float4 txAA;	float4 txAB;	// Auxilary Tensor
	float4 txBA;	float4 txBB;	// Auxilary Tensor
	
	// Compute 3D coordinates for texture fetching
	float tx = blockIdx.x * blockDim.x + threadIdx.x;
	float ty = blockIdx.y * blockDim.y + threadIdx.y;
	float tz = threadIdx.z + z_offset;
	
	// Compute global threadId for global memory access
	int idx = (int) tx + (int) ty * su + (int) tz * su * sv;
	int jdx = 0;

	// Compute the pointers for the output
	char *		sliceA	= (char *) outA.ptr + (int) tz * outA.pitch * sv;
	char *		sliceB	= (char *) outB.ptr + (int) tz * outB.pitch * sv;
	float4 *	rowA	= (float4 *) (sliceA + (int) ty * outA.pitch);
	float4 *	rowB	= (float4 *) (sliceB + (int) ty * outB.pitch);
	int			col		= (int) tx;

	// Increment by 0.5f to align with data points
	tx += 0.5f;
	ty += 0.5f;
	tz += 0.5f;

	// Index must remain within the range of tensors
	if (tx >= su || tx < 0 || ty >= sv || ty < 0 || tz >= sw || tz < 0 || idx >= su * sv * sw)
		return;

	// Read first four tensor elements from first texture
	txDA = tex3D(textureDA, tx, ty, tz);;

	// Read last two tensor elements from second texture
	txDB = tex3D(textureDB, tx, ty, tz);

	// Apply a constant gain factor to all tensor elements.
	txDA.x *= P.gain;
	txDA.y *= P.gain;
	txDA.z *= P.gain;
	txDA.w *= P.gain;
	txDB.x *= P.gain;
	txDB.y *= P.gain;

	// Compute the determinant of the tensor (with gain applied).
	float determinant = txDA.x * txDA.w * txDB.y + txDA.y * txDB.x * txDA.z + txDA.y * txDB.x * txDA.z
					  - txDA.z * txDA.w * txDA.z - txDA.y * txDA.y * txDB.y - txDA.x * txDB.x * txDB.x;
	
	// Checks if the matrix is invertible (determinant != 0)
	if (determinant <= PP_CLOSE_TO_ZERO && determinant >= -PP_CLOSE_TO_ZERO)
	{
		txDA.x = 1.0f;
		txDA.y = 0.0f;
		txDA.z = 0.0f;
		txDA.w = 1.0f;
		txDB.x = 0.0f;
		txDB.y = 1.0f;
	}

	// Apply sharpening if necessary. "sharpeningMethod" can have the following values:
	// 0 - No sharpening.
	// 1 - Exponentiate tensors
	// 2 - Divide tensors elements by the trace
	// 3 - Exponentiation followed by division by trace

	if (P.sharpeningMethod != 0)
	{
		// Only sharpen tensors with scalar value below the threshold
		if (txDB.w < P.threshold)
		{
			// Exponentiate the tensor
			if (P.sharpeningMethod == 1 || P.sharpeningMethod == 3)
			{
				txAA = txDA;
				txAB = txDB;

				// Multiply the tensor with itself a fixed number of times
				for (jdx = 0; jdx < (P.exponent - 1); jdx++)
				{
					txBA.x = txAA.x * txDA.x + txAA.y * txDA.y + txAA.z * txDA.z;
					txBA.y = txAA.x * txDA.y + txAA.y * txDA.w + txAA.z * txDB.x;
					txBA.z = txAA.x * txDA.z + txAA.y * txDB.x + txAA.z * txDB.y;
					txBA.w = txAA.y * txDA.y + txAA.w * txDA.w + txAB.x * txDB.x;
					txBB.x = txAA.y * txDA.z + txAA.w * txDB.x + txAB.x * txDB.y;
					txBB.y = txAA.z * txDA.z + txAB.x * txDB.x + txAB.y * txDB.y;

					txAA = txBA;
					txAB = txBB;
				}

				txDA = txAA;
				txDB = txAB;
			}

			// Divide all elements by the trace
			if (P.sharpeningMethod == 2 || P.sharpeningMethod == 3)
			{
				float trace = txDA.x + txDA.w + txDB.y;

				if (trace == 0.0f)	trace = 1.0f;

				txDA.x /= trace;
				txDA.y /= trace;
				txDA.z /= trace;
				txDA.w /= trace;
				txDB.x /= trace;
				txDB.y /= trace;
			}
		}
	}

	// Store the pre-processed (gain + sharpening) tensor in global memory.
	rowA[col] = txDA;
	rowB[col] = txDB;
}


#undef PP_CLOSE_TO_ZERO


#endif // bmia_FiberTracking_PreProcessingKernel_h
