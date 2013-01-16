#pragma once

//  This file:                              Interface to the CUDA-based DT/FT/erosion/dilation implementation
//
//



extern int fboSizeX;						//Sizes of 3D volume; must be powers of 2; size X so far must equal size Y; product must fit in GPU memory
extern int fboSizeY;						//Set by initialization()
extern int fboSizeZ;

const int phase1Band     = 1;				//Band-sizes for the 3 passes of the CUDA Voronoi algorithm
const int phase2Band	 = 1; 
const int phase3Band	 = 2;

extern int*		inputPoints;				//Input sites, coded for the PBA, placed in a simple 1D vector, for easy iteration over all sites
extern int*		outputVoronoi;				//Output (computed) Voronoi regions from the PBA
extern unsigned char*  inputVolume;			//?
extern float*	outputDT;					//DT of the input sites



void			dtcuda_initialization(int nVertices,int nx,int ny,int nz);
void			dtcuda_initialization2(int nVertices,int nx,int ny,int nz,float sx,float sy,float sz);
void			dtcuda_deinitialization();


void            dtcuda_bindBuffer(int*&);	//Bind an already allocated internal opaque buffer to given client pointer
void            dtcuda_bindBuffer(float*&);	//Pointers already bound to that buffer will become silently invalid


