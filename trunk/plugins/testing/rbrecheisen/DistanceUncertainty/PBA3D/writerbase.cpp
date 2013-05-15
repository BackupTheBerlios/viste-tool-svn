#include <fstream>
#include <iostream>
#include "readers.h"
#include "gpudefs.h"
#include "pba/pba3D.h"


using namespace std;


void saveVolume(float* v,unsigned char* mask,int X,int Y,int Z,const char* f)
{
   FILE* fp = fopen(f,"wb");
   if (!fp) return;
   fwrite(&X,sizeof(int),1,fp);
   fwrite(&Y,sizeof(int),1,fp);
   fwrite(&Z,sizeof(int),1,fp);
   
   unsigned char sz = sizeof(float);   
   fwrite(&sz,1,1,fp);											//Write file type (float, i.e. 4 bytes/voxel)

   for(int kk = 0; kk < Z; kk++)								//REMARK: we write in the order k,j,i (important to know for readers)
	for(int jj = 0; jj < Y; jj++)				
	  for(int ii = 0; ii < X; ++ii)
	  {
		int    id = TOID(ii,jj,kk,X,Y);
		fwrite(v+id,sizeof(float),1,fp);
	  }
   fclose(fp);
}


void saveThresholdedVolume(float* v,float thr,unsigned char* mask,int X,int Y,int Z,const char* f)
//Save binvox volume obtained from v[] thresholded with thr and masked with mask[]
{
   int idx = 0;
   for(int ii = 0; ii < X; ii++)								//REMARK: be careful, mask[] is not encoded with PBA's scheme, but raw binvox order..
	for(int kk = 0; kk < Z; kk++)								//REMARK: also, note that this overwrites mask[] with the thresholding result
	  for(int jj = 0; jj < Y; ++jj,++idx)
	  {
		int id  = TOID(ii,jj,kk,X,Y);
		unsigned char val = (mask[idx])? 255 :					//'rump': inside eroded shape before inflation
							(v[id]<thr)? 255 :					//'inflated': inside inflated shape but outside rump
							0;									//outside thresholded inflation (background)
		mask[idx] = val;										
	  }
	  
   writeBinvox(mask,X,Y,Z,f);	  
}
