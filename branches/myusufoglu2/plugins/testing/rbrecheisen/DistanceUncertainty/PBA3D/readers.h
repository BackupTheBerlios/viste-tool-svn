#pragma once

//This file:	Declares several readers for different types of 3D input files.
//				Data is read into the PBA-specific structures.
//
//


int readPLY(const char* f,int tx, int ty, int tz, int* inputVoronoi,int* inputPoints,			//Read point cloud from PLY files into the specific encoding of this program
			bool tight_fit);

int readBinvox(const char* f,int tx, int ty, int tz, unsigned char* inputVolume, int* inputVoronoi,int* inputPoints,			//Read voxel model from binvox file
			bool tight_fit);

void generateRandomPoints(int* inputVoronoi, int* inputPoints, int texSizeX, int texSizeY,		//Generate random points
		    int texSizeZ, int& nPoints);														
																							
//--------



void saveVolume(float* vol,unsigned char* mask /*=0*/, int x,int y,int z,const char* file);


void saveThresholdedVolume(float* v,float thr,unsigned char* mask,int X,int Y,int Z,const char* f);


void writeBinvox(const unsigned char* f, int width, int height, int depth, const char* fname);