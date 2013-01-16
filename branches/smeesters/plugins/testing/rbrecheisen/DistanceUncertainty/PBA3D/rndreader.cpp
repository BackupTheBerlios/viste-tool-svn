#include "myrandom.h"
#include <math.h>
#include "utils.h"
#include "gpudefs.h"
#include "pba/pba3D.h"


// Generate input points
void generateRandomPoints(int* inputVoronoi, int* inputPoints, int texSizeX, int texSizeY, int texSizeZ, int& nPoints)
{	
    int tx, ty, tz, id; 

    for (int i = 0; i < texSizeX * texSizeY * texSizeZ; i++)
        inputVoronoi[i] = MARKER; 
		
	float xy = myMin(texSizeX,texSizeY);	
		
	float  R = xy*0.4;
	float dr = 0;
	float dR = xy*0.4;	
	float PI = acos(0.0f)*2;
	float ox = texSizeX/2, oy = texSizeY/2, oz = texSizeZ/2;

    int pid = 0;
	for (int i = 0; i < nPoints; i++)
	{
	        float  r = R + dr*(myrandom()*2-1);
		    float  a = myrandom()*2*PI;			//0..2PI
			float  b = myrandom()*PI - PI/2;	//-PI/2,PI/2
			tx = ox + r*sin(a);
			ty = oy + r*cos(a);
			tz = oz + dR*(myrandom()*2-1);
            id = TOID(tx, ty, tz, texSizeX, texSizeY);
			if (inputVoronoi[id] == MARKER)
			{
              inputVoronoi[id] = ENCODE(tx, ty, tz); 
              inputPoints[pid] = ENCODE(tx, ty, tz);
			  ++pid;
			}
    }	
	
	nPoints = pid;
}




