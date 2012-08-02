#include <math.h>
#include <iostream>
#include "utils.h"
#include "pba/pba3D.h"
#include "dtengine.h"



// Global data structures


int*			inputPoints=0;                  //Input sites, coded for the CUDA 3D-DT, placed in a simple 1D vector, for easy iteration over all sites
int*			outputVoronoi=0;                //Output (computed) Voronoi regions from the CUDA 3D-DT
unsigned char*  inputVolume=0;
float*			outputDT=0;                     //DT of the input sites
int*			buffer0 = 0;
int				fboSizeX,fboSizeY,fboSizeZ;     //Sizes of 3D DT volume

void dtcuda_bindBuffer(int*& ptr)
{
   ptr = (int*) buffer0;
}


void dtcuda_bindBuffer(float*& ptr)
{
   ptr = (float*) buffer0;
}


void dtcuda_initialization(int maxNVertices,int nx,int ny,int nz)               //Initialize CPU-side and CUDA-side buffers
{
    fboSizeX = nx; fboSizeY = ny; fboSizeZ = nz; 

    pba3DInitialization(fboSizeX,fboSizeY,fboSizeZ);							//Initialize CUDA structures

    buffer0         = new int[fboSizeX * fboSizeY * fboSizeZ];
	inputVolume     = new unsigned char[fboSizeX * fboSizeY * fboSizeZ];
    inputPoints     = new int[maxNVertices];									//Allocate CPU-side buffers
    outputVoronoi   = new int[fboSizeX * fboSizeY * fboSizeZ]; 
}

void dtcuda_initialization2( int maxNVertices, int nx, int ny, int nz, float sx, float sy, float sz )
{
	fboSizeX = nx; fboSizeY = ny; fboSizeZ = nz;

	//pba3DInitialization(fboSizeX,fboSizeY,fboSizeZ);							//Initialize CUDA structures
	pba3DInitialization2( nx, ny, nz, sx, sy, sz );

	buffer0         = new int[fboSizeX * fboSizeY * fboSizeZ];
	inputVolume     = new unsigned char[fboSizeX * fboSizeY * fboSizeZ];
	inputPoints     = new int[maxNVertices];									//Allocate CPU-side buffers
	outputVoronoi   = new int[fboSizeX * fboSizeY * fboSizeZ];
}

void dtcuda_deinitialization()                                                  //Deinitialize all CPU-side and CUDA-side buffers
{
    pba3DDeinitialization();													//Clean up CUDA structures

    delete[] inputVolume;
    delete[] inputPoints;														//Deallocate CPU-side buffers
    delete[] outputVoronoi; 
	delete[] buffer0;
}



