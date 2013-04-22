/*
 * derivativesKernel.cu
 *
 * 2011-06-05	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberTracking_DerivativesKernel_h
#define bmia_FiberTracking_DerivativesKernel_h


// Value close to zero, used as error margin
#define PP_CLOSE_TO_ZERO 0.0001f


//--------------------------[ derivativesKernel ]--------------------------\\

__global__ void derivativesKernel(cudaPitchedPtr outA, cudaPitchedPtr outB, GC_imageInfo grid, int dir, float offset_z)
{

	// Set image dimensions
	int dim[3];
	dim[0] = grid.su;
	dim[1] = grid.sv;
	dim[2] = grid.sw;

	// Set voxels spacing
	float spacing[3];
	spacing[0] = grid.du;
	spacing[1] = grid.dv;
	spacing[2] = grid.dw;

	// Four-element used to contain the tensors. For the input tensors, the following holds:
	//
	// txDA.x	=	T(1,1)
	// txDA.y	=	T(1,2)
	// txDA.z	=	T(1,3)
	// txDA.w	=	T(2,2)
	// txDB.x	=	T(2,3)
	// txDB.y	=	T(3,3)

	float4 txDA;	float4 txDB;	// Pre-Processed Tensor
	float4 txAA;	float4 txAB;	// Auxilary Tensor
	float4 txBA;	float4 txBB;	// Auxilary Tensor
	float4 txOA;	float4 txOB;	// Output derivative tensor

	// Compute 3D coordinates for texture fetching
	float tc[3];
	tc[0] = blockIdx.x * blockDim.x + threadIdx.x;
	tc[1] = blockIdx.y * blockDim.y + threadIdx.y;
	tc[2] = threadIdx.z + offset_z;

	// Compute the pointers for the output
	char *		sliceA	= (char *) outA.ptr + (int) tc[2] * outA.pitch * grid.sv;
	char *		sliceB	= (char *) outB.ptr + (int) tc[2] * outB.pitch * grid.sv;
	float4 *	rowA	= (float4 *) (sliceA + (int) tc[1] * outA.pitch);
	float4 *	rowB	= (float4 *) (sliceB + (int) tc[1] * outB.pitch);
	int			col		= (int) tc[0];

	// Compute global threadId for global memory access
	int idx = (int) tc[0] + (int) tc[1] * dim[0] + (int) tc[2] * dim[0] * dim[1];

	// Index must remain within the range of tensors
	if (tc[0] >= dim[0] || tc[0] < 0 || tc[1] >= dim[1] || tc[1] < 0 || tc[2] >= dim[2] || tc[2] < 0 || idx >= dim[0] * dim[1] * dim[2])
		return;

	// Increment by 0.5f to align with data points
	tc[0] += 0.5f;
	tc[1] += 0.5f;
	tc[2] += 0.5f;
	
	// Denominator used for computing the derivative
	float div;

	// Determinant of the tensor
	float determinant;

	// Set the denominator to once or twice the voxel spacing in the dimension of derivation,
	// depending on whether or not the current voxel is located on the border of the image.
	if (tc[dir] == 0 || tc[dir] == (dim[dir] - 1))
		div = spacing[dir];
	else
		div = 2.0f * spacing[dir];

	// Increment index in selected dimension. Since the addressing mode of the textures is set
	// to "cudaAddressModeClamp", we do not need to worry about exceeding the image dimensions.

	tc[dir]++;

	// Fetch the pre-processed tensor from texture memory
	txDA = tex3D(textureDA, tc[0], tc[1], tc[2]);
	txDB = tex3D(textureDB, tc[0], tc[1], tc[2]);

	// Compute the determinant
	determinant = txDA.x * txDA.w * txDB.y + txDA.y * txDB.x * txDA.z + txDA.z * txDA.y * txDB.x
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
		determinant = 1.0f;
	}

	// Compute the inverse of the tensor. We use the analytic solution, in which we divide the
	// transposed matrix of cofactors by the determinant of the 3 x 3 tensor. 

	txAA.x =  (txDA.w * txDB.y - txDB.x * txDB.x) / determinant;
	txAA.y = -(txDA.y * txDB.y - txDB.x * txDA.z) / determinant;
	txAA.z =  (txDA.y * txDB.x - txDA.w * txDA.z) / determinant;
	txAA.w =  (txDA.x * txDB.y - txDA.z * txDA.z) / determinant;
	txAB.x = -(txDA.x * txDB.x - txDA.y * txDA.z) / determinant;
	txAB.y =  (txDA.x * txDA.w - txDA.y * txDA.y) / determinant;

	// Divide by the denominator and store in txB
	txBA.x = txAA.x / div;
	txBA.y = txAA.y / div;
	txBA.z = txAA.z / div;
	txBA.w = txAA.w / div;
	txBB.x = txAB.x / div;
	txBB.y = txAB.y / div;

	// Decrease the coordinates by two in the selected dimension; 
	// "tc[dir]" is now one less than its original value.

	tc[dir] -= 2;

	// Fetch the pre-processed tensor from texture memory
	txDA = tex3D(textureDA, tc[0], tc[1], tc[2]);
	txDB = tex3D(textureDB, tc[0], tc[1], tc[2]);

	// Compute the determinant
	determinant = txDA.x * txDA.w * txDB.y + txDA.y * txDB.x * txDA.z + txDA.z * txDA.y * txDB.x
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
		determinant = 1.0f;
	}

	// Compute the inverse of the tensor. We use the analytic solution, in which we divide the
	// transposed matrix of cofactors by the determinant of the 3 x 3 tensor. 

	txAA.x =  (txDA.w * txDB.y - txDB.x * txDB.x) / determinant;
	txAA.y = -(txDA.y * txDB.y - txDB.x * txDA.z) / determinant;
	txAA.z =  (txDA.y * txDB.x - txDA.w * txDA.z) / determinant;
	txAA.w =  (txDA.x * txDB.y - txDA.z * txDA.z) / determinant;
	txAB.x = -(txDA.x * txDB.x - txDA.y * txDA.z) / determinant;
	txAB.y =  (txDA.x * txDA.w - txDA.y * txDA.y) / determinant;

	// Divide by the denominator and substract from txB to compute the final derivative
	txBA.x -= txAA.x / div;
	txBA.y -= txAA.y / div;
	txBA.z -= txAA.z / div;
	txBA.w -= txAA.w / div;
	txBB.x -= txAB.x / div;
	txBB.y -= txAB.y / div;

	// Increment coordinate, so that we're back at the original point
	tc[dir]++;

	// Merge computed derivate with existing data if needed
	if (dir == 0)
	{
		// Derivative dGdu is stored in txDB.{z,w} and txDGDUA.{x,y,z,w}. We load txDB for
		// the current point, and overwrite its last two elements with the first two elements
		// of the computed derivative.

		txDB = tex3D(textureDB, tc[0], tc[1], tc[2]);

		txOA.x = txDB.x;
		txOA.y = txDB.y;
		txOA.z = txBA.x;
		txOA.w = txBA.y;
		txOB.x = txBA.z;
		txOB.y = txBA.w;
		txOB.z = txBB.x;
		txOB.w = txBB.y;
	}
	else if (dir == 1)
	{
		// Derivative dGdv is stored in txDGDVA.{x,y,z,w} and txDGDVB.{x,y}. We do not need
		// to merge the computed data with existing data, so we directly copy it to the output

		txOA.x = txBA.x;
		txOA.y = txBA.y;
		txOA.z = txBA.z;
		txOA.w = txBA.w;
		txOB.x = txBB.x;
		txOB.y = txBB.y;
	}
	else
	{
		// Derivative dGdw is stored in txDGDVB.{z,w} and txDGDWA.{x,y,z,w}. We load txDGDVB for
		// the current point, and overwrite its last two elements with the first two elements
		// of the computed derivative.

		txDB = tex3D(textureDGDVB, tc[0], tc[1], tc[2]);

		txOA.x = txDB.x;
		txOA.y = txDB.y;
		txOA.z = txBA.x;
		txOA.w = txBA.y;
		txOB.x = txBA.z;
		txOB.y = txBA.w;
		txOB.z = txBB.x;
		txOB.w = txBB.y;
	}

	// Write the two output vectors to the output arrays
	rowA[col] = txOA;
	rowB[col] = txOB;
}


#undef PP_CLOSE_TO_ZERO


#endif // bmia_FiberTracking_DerivativesKernel_h