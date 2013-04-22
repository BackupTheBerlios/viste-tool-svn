#include <device_functions.h>

#include "pba3D.h"
#include <stdio.h>

// Parameters for CUDA kernel executions
#define BLOCKX      32
#define BLOCKY      4
#define BLOCKXY     16

#define MYINFINITY    0x3ff

/****** Global Variables *******/
int **pbaTextures;							//aray of 2 textures, each pbaMemSize bytes; used in a sort of ping-pong scheme
int pbaCurrentBuffer; 

int pbaMemSize;								//size of a texture, pbaTexX*pbaTexY*pbaTexZ ints
int pbaTexX,pbaTexY,pbaTexZ;
int pbaTexXY;								//REMARK: our implem, assumes X and Y dimensions are equal
int log2Width; 

float spacingX = 1.0f;
float spacingY = 1.0f;
float spacingZ = 1.0f;

texture<int>	pbaTexColor;				//tex-reference
texture<int>	pbaTexLinks; 
texture<short>	pbaTexPointer; 

/********* Kernels ********/
#include "pba3DKernel.h"

///////////////////////////////////////////////////////////////////////////
//
// Initialize necessary memory for 3D Voronoi Diagram computation
// - textureSize: The size of the Discrete Voronoi Diagram (width = height)
//
///////////////////////////////////////////////////////////////////////////
void pba3DInitialization(int tx,int ty,int tz)
{
	pbaTexX  = tx; pbaTexY = ty; pbaTexZ = tz;
	pbaTexXY = tx;													//REMARK: assumes tx==ty

    int tmp = pbaTexZ;											    
    while (tmp > 1) { tmp /= 2; ++log2Width; }
	pbaMemSize = pbaTexX * pbaTexY * pbaTexZ * sizeof(int);
	
	//printf("Memo %f\n",pbaMemSize*2.0/(1024*1024));

	pbaTextures = (int**) malloc(2 * sizeof(int *)); 
	cudaMalloc((void **) &pbaTextures[0], pbaMemSize);				//Allocate 2 textures
	cudaMalloc((void **) &pbaTextures[1], pbaMemSize); 
}

void pba3DInitialization2(int tx,int ty,int tz, float sx, float sy, float sz)
{
	pbaTexX  = tx; pbaTexY = ty; pbaTexZ = tz;
	pbaTexXY = tx;													//REMARK: assumes tx==ty

	spacingX = sx;
	spacingY = sy;
	spacingZ = sz;

	int tmp = pbaTexZ;
	while (tmp > 1) { tmp /= 2; ++log2Width; }
	pbaMemSize = pbaTexX * pbaTexY * pbaTexZ * sizeof(int);

	//printf("Memo %f\n",pbaMemSize*2.0/(1024*1024));

	pbaTextures = (int**) malloc(2 * sizeof(int *));
	cudaMalloc((void **) &pbaTextures[0], pbaMemSize);				//Allocate 2 textures
	cudaMalloc((void **) &pbaTextures[1], pbaMemSize);
}

///////////////////////////////////////////////////////////////////////////
//
// Deallocate all allocated memory
//
///////////////////////////////////////////////////////////////////////////
void pba3DDeinitialization()
{
	cudaFree(pbaTextures[0]); 
	cudaFree(pbaTextures[1]); 
	free(pbaTextures); 
}

// Copy input to GPU 
void pba3DInitializeInput(int *input)
{
    cudaMemcpy(pbaTextures[0], input, pbaMemSize, cudaMemcpyHostToDevice); 

    // Set Current Source Buffer
	pbaCurrentBuffer = 0;
}

// In-place transpose a cubic texture. Transposition are performed on each XY plane. Point coordinates are also swapped. 
void pba3DTransposeXY(int *texture) 
{
    dim3 block(BLOCKXY, BLOCKXY); 
    
	//dim3 grid((pbaTexXY / BLOCKXY) * pbaTexZ, pbaTexXY / BLOCKXY); 
    //kernelTransposeXY<<< grid, block >>>(texture, log2Width, pbaTexZ - 1); 

    //OK
	dim3 grid(pbaTexX/BLOCKXY,pbaTexY/BLOCKXY);
	kernelTransposeXY<<< grid, block >>>(texture, pbaTexXY, pbaTexZ);
}

// Phase 1 of PBA. m1 must divide texture size. Sweeping is done along the Z axiz. 
void pba3DColorZAxis(int m1) 
{
   	dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid  = dim3((pbaTexXY / block.x) * m1, pbaTexXY / block.y); 

    cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]); 
    kernelFloodZ<<< grid, block >>>(pbaTextures[1 - pbaCurrentBuffer], pbaTexXY, pbaTexXY / block.x, pbaTexZ / m1); 
    pbaCurrentBuffer = 1 - pbaCurrentBuffer; 

	if (m1 > 1)											// Passing information between bands, if more such bands exist
	{
		cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]); 
		kernelPropagateInterband<<< grid, block >>>(pbaTextures[1 - pbaCurrentBuffer], pbaTexXY, pbaTexXY / block.x, pbaTexZ / m1);
		cudaBindTexture(0, pbaTexLinks, pbaTextures[1 - pbaCurrentBuffer]);
		kernelUpdateVertical<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], pbaTexXY, pbaTexXY / block.x, pbaTexZ / m1);
	}
}

// Phase 2 of PBA. m2 must divide texture size. This method work along the Y axis
void pba3DComputeProximatePointsYAxis(int m2) 
{
	int iStack = 1 - pbaCurrentBuffer; 
	int iForward = pbaCurrentBuffer; 

	dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid = dim3((pbaTexXY / block.x) * m2, pbaTexZ / block.y); 
	
	// Compute proximate points locally in each band
	cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]); 
    kernelMaurerAxis<<< grid, block >>>(pbaTextures[iStack], pbaTexXY, pbaTexXY / block.x, pbaTexXY / m2); 
	
	//!!cudaMemcpy(pbaTextures[pbaCurrentBuffer], pbaTextures[1-pbaCurrentBuffer], pbaMemSize, cudaMemcpyDeviceToDevice); 

	// Construct forward pointers
	cudaBindTexture(0, pbaTexLinks, pbaTextures[iStack]); 
	kernelCreateForwardPointers<<< grid, block >>>((short *) pbaTextures[iForward], pbaTexXY, pbaTexXY / block.x, pbaTexXY / m2); 
	
	cudaBindTexture(0, pbaTexPointer, pbaTextures[iForward]); 

	// Repeatly merging two bands into one
	for (int noBand = m2; noBand > 1; noBand /= 2) {
		grid = dim3((pbaTexXY / block.x) * (noBand / 2), pbaTexXY / block.y); 
		kernelMergeBands<<< grid, block >>>(pbaTextures[iStack], 
			(short *) pbaTextures[iForward], pbaTexXY, pbaTexXY / block.x, pbaTexXY / noBand); 
	}

	cudaUnbindTexture(pbaTexLinks); 
	cudaUnbindTexture(pbaTexColor); 
	cudaUnbindTexture(pbaTexPointer); 
}

// Phase 3 of PBA. m3 must divide texture size. This method colors along the Y axis
void pba3DColorYAxis(int m3) 
{
	dim3 block = dim3(BLOCKX, m3); 
    dim3 grid = dim3(pbaTexXY / block.x, pbaTexZ); 

    cudaBindTexture(0, pbaTexColor, pbaTextures[1 - pbaCurrentBuffer]); 
    kernelColorAxis<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], pbaTexXY); 
    cudaUnbindTexture(pbaTexColor); 
}

void pba3DCompute(int m1, int m2, int m3)
{
    /************* Compute Z axis *************/
    // --> (X, Y, Z)
	pba3DColorZAxis(m1); 
	
	/************* Compute Y axis *************/
    // --> (X, Y, Z)
	pba3DComputeProximatePointsYAxis(m2);
	
	pba3DColorYAxis(m3); 
			
    // --> (Y, X, Z)
    pba3DTransposeXY(pbaTextures[pbaCurrentBuffer]); 
	

    /************** Compute X axis *************/
    // Compute X axis
	pba3DComputeProximatePointsYAxis(m2);
	pba3DColorYAxis(m3); 

    // --> (X, Y, Z)
    pba3DTransposeXY(pbaTextures[pbaCurrentBuffer]); 
}




__global__ void kernelThreshold(unsigned char* output, int sizeXY, int sizeZ, float thr2, int mod)
//Input:    pbaTexColor (closest-site-ids per pixel)
//Output:   'output', binary 1-byte-per-voxel foreground/background volume
{
    int blkX = blockIdx.x % mod;	
	int tx = blkX * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;  
	
    int voroid; 
	int nx,ny,nz,dx,dy,dz;
	
	for(int tz=0;tz<sizeZ;++tz)
	{
      int id = TOID(tx, ty, tz, sizeXY);
	  voroid = tex1Dfetch(pbaTexColor,id);							//get the closest-site to tx,ty,tz	
	  DECODE(voroid,nx,ny,nz);										//nx,ny,nz = coords of closest-site
	  dx = tx - nx; dy = ty - ny; dz = tz - nz; 
	  float D2 = dx * dx + dy * dy + dz * dz;						//distance to closest-site i.e. DT of tx,ty,tz
	  output[id] = (D2>thr2)? 1:0;									//do segmentation
    }
}




__global__ void kernelDT(float* output, int sizeXY, int sizeZ, int mod, float sx, float sy, float sz)
//Input:    pbaTexColor (closest-site-ids per pixel)
//Output:   'output' DT (floats)
{
    int blkX = blockIdx.x % mod;	
	int tx = blkX * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;  
	
    int voroid; 
	int nx,ny,nz,dx,dy,dz;
	
	for(int tz=0;tz<sizeZ;++tz)
	{
      int id = TOID(tx, ty, tz, sizeXY);
	  voroid = tex1Dfetch(pbaTexColor,id);							//get the closest-site to tx,ty,tz
	  DECODE(voroid,nx,ny,nz);										//nx,ny,nz = coords of closest-site
//	  dx = tx - nx; dy = ty - ny; dz = tz - nz;
	  dx = sx * (tx - nx);							// Multiply the deltas with correct voxel spacing
	  dy = sy * (ty - ny);
	  dz = sz * (tz - nz);
	  float D = sqrtf(dx * dx + dy * dy + dz * dz);					//distance to closest-site i.e. DT of tx,ty,tz
	  output[id] = D;												//store distance in output
    }
}





void threshold(float thr2)
{
	dim3 block(BLOCKXY,BLOCKXY);
	dim3 grid(pbaTexXY/block.x,pbaTexXY/block.y);
	
	cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]);	//This is the input of the kernel	
	
    kernelThreshold<<< grid, block >>>((unsigned char*)pbaTextures[1-pbaCurrentBuffer], pbaTexXY, pbaTexZ, thr2, pbaTexXY / block.x);
   
   //REMARK: we use only 1 byte/pixel for the segmented image, but we reuse the alloc'd buffer, that's the reason of the cast above.
}


void compute_dt()
{
	dim3 block(BLOCKXY,BLOCKXY);
	dim3 grid(pbaTexXY/block.x,pbaTexXY/block.y);
	
	cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]);	//This is the input of the kernel	
	
	kernelDT<<< grid, block >>>((float*)pbaTextures[1-pbaCurrentBuffer], pbaTexXY, pbaTexZ, pbaTexXY / block.x,
								spacingX, spacingY, spacingZ);
   
   //REMARK: we interpret the output buffer as a float buffer
}




// ENTRY POINT
// Compute 3D Voronoi diagram
// Input: a 3D texture. Each pixel is an integer encoding 3 coordinates. 
//    For each site at (x, y, z), the pixel at coordinate (x, y, z) should contain 
//    the encoded coordinate (x, y, z). Pixels that are not sites should contain 
//    the integer MARKER. Use ENCODE (and DECODE) macro to encode (and decode).
// See original paper for the effect of the three parameters: 
//     phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
void pba3DVoronoiDiagram(int *input, int *output, int phase1Band, int phase2Band, int phase3Band) 
{
	pba3DInitializeInput(input); 
	
    // Compute the 3D Voronoi Diagram
    pba3DCompute(phase1Band, phase2Band, phase3Band); 
	
    // Copy back the result
    cudaMemcpy(output, pbaTextures[pbaCurrentBuffer], pbaMemSize, cudaMemcpyDeviceToHost); 
}

void pba3DThreshold(int* voronoiInput, unsigned char* thresholdedOutput, float thr)
{
    pba3DInitializeInput(voronoiInput);

	threshold(thr);

    // Copy back the result
    cudaMemcpy(thresholdedOutput, pbaTextures[1-pbaCurrentBuffer], pbaMemSize/sizeof(int), cudaMemcpyDeviceToHost); 
}

void pba3DDT(int* voronoiInput, float* DT)
{
    pba3DInitializeInput(voronoiInput);

	compute_dt();

    // Copy back the result
    cudaMemcpy(DT, pbaTextures[1-pbaCurrentBuffer], pbaMemSize, cudaMemcpyDeviceToHost); 
}
