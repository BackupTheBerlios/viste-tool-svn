#pragma once


// Initialize CUDA and allocate memory
// textureSize is tx*ty*tz, with at least 2^5 values per dimension
extern "C" void pba3DInitialization(int tx,int ty,int tz); 

extern "C" void pba3DInitialization2(int tx,int ty,int tz, float sx, float sy, float sz);


// Deallocate memory on GPU
extern "C" void pba3DDeinitialization(); 

// Compute 3D Voronoi diagram
// Input: a 3D texture. Each pixel is an integer encoding 3 coordinates. 
//    For each site at (x, y, z), the pixel at coordinate (x, y, z) should contain 
//    the encoded coordinate (x, y, z). Pixels that are not sites should contain 
//    the integer MARKER. Use ENCODE (and DECODE) macro to encode (and decode).
// Output: a 3D texture. Each pixel is an integer encoding 3 coordinates
//    of its nearest site. 
// See original paper for the effect of the three parameters: 
//     phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
extern "C" void pba3DVoronoiDiagram(int *input, int *output, int phase1Band, int phase2Band, int phase3Band);

// Compute a binary thresholding of a Voronoi-encoded volume. Arg1 is the output of pba3DVoronoiDiagram.
extern "C" void pba3DThreshold(int* voronoiInput, unsigned char* thresholdedOutput, float thr2);

// Compute a floating-point distance transform of a Voronoi-encoded volume. Arg1 is the output of pbaVoronoiDiagram.
extern "C" void pba3DDT(int* voronoiInput, float* DT);





#define MARKER	    -1
#define MAX_INT 	201326592

#define ENCODE(x, y, z)  (((x) << 20) | ((y) << 10) | (z))
#define DECODE(value, x, y, z) \
    x = (value) >> 20; \
    y = ((value) >> 10) & 0x3ff; \
    z = (value) & 0x3ff

#define GET_X(value)	((value) >> 20)
#define GET_Y(value)	(((value) >> 10) & 0x3ff)
#define GET_Z(value)	(((value) == MARKER) ? MAX_INT : ((value) & 0x3ff))
