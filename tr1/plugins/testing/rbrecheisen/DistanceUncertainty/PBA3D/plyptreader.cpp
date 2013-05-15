#include <iostream>
#include <fstream>
#include <string>
#include "gpudefs.h"
#include "pba/pba3D.h"

using namespace std;



int readPLYPoints(const char* f, float* v, float* vmin, float* vmax,const int MAXPTS)	//Read all 3D vertices from f into v[] and determine 3D bbox (vmin,vmax) too
{
   ifstream fs(f);
   if (!fs.good()) return -1;
   
   int nvertices = 0; int state = 0;
   string k; bool vread = false; int vprops = 0;
   
   while (!fs.eof())
   {
     fs>>k;
	 if (state==0)						//reading header:
	 {
       if (!nvertices)					//before finding 'element vertex'
       {
         if (k=="element")
	     {
	        fs>>k;
		    if (k=="vertex")
		    {
		      fs>>nvertices;
			  vread = true;
		    }
		 }
	   }
	   else
	   {
		if (k=="property" && vread) ++vprops;
		if (k=="element") vread = false;
	   }	
	   
	  
	   if (k=="end_header")
		 if (!nvertices)
		 { cout<<"Error: end_header, no vertices count found"<<endl; return 0; }
		 else state = 1;
	 }
	 
     if (state==1) //state==1					//reading vertices:
	 {
	   if (nvertices>MAXPTS) cout<<"Error: vertex buffer "<<MAXPTS<<" but should read "<<nvertices<<" vertices"<<endl; 
	 
	   float d; vprops -= 3;
	   fs>>v[0]>>v[1]>>v[2];
	   for(int p=0;p<vprops;++p) fs>>d;
       vmin[0]=vmax[0]=v[0];
       vmin[1]=vmax[1]=v[1];
       vmin[2]=vmax[2]=v[2];
	   v+=3;
	   
	   for (int i=1;i<nvertices;++i,v+=3)
	   {
	     fs>>v[0]>>v[1]>>v[2];
  	     for(int p=0;p<vprops;++p) fs>>d;
		 if (vmin[0]>v[0]) vmin[0]=v[0]; if (vmax[0]<v[0]) vmax[0]=v[0];
		 if (vmin[1]>v[1]) vmin[1]=v[1]; if (vmax[1]<v[1]) vmax[1]=v[1];
		 if (vmin[2]>v[2]) vmin[2]=v[2]; if (vmax[2]<v[2]) vmax[2]=v[2];
       }
	   return nvertices;
 	 }
   }	 
  	 
   return nvertices;
}





int readPLY(const char* f,int tX, int tY, int tZ,int* inputVoronoi,int* inputPoints, bool tight_fit)	
//Read vertices, store them in encoded textures inputVoronoi, inputPoints
{
  const int MAXPTS = 1000000;
  float* v = new float[MAXPTS*3]; 
  
  float vmin[3],vmax[3],sz[3];
  
  int N = readPLYPoints(f,v,vmin,vmax,MAXPTS);							//1. read vertices in v[]
  if (N==-1) return -1;													//   file reading error (file not found)
  if (!N) return 0;

  sz[0] = vmax[0]-vmin[0];												//2. determine bbox sizes
  sz[1] = vmax[1]-vmin[1];
  sz[2] = vmax[2]-vmin[2];

  float S = tX/sz[0];													//3. determine scaling factor from world (bbox) to texture-box (tX,tY,tZ)
  if (tight_fit)														//a) tight fit: max uniform scaling factor which would still fit the entire object in the tex
  {
    if (tY/sz[1]<S) S=tY/sz[1];
    if (tZ/sz[2]<S) S=tZ/sz[2];
  }
  else
  {
    if (tY/sz[1]>S) S=tY/sz[1];
    if (tZ/sz[2]>S) S=tZ/sz[2];
  }
  
  for (int i = 0; i < tX * tY * tZ; ++i) inputVoronoi[i] = MARKER;		//4. initialize the voronoi tex	

  float* vp = v;
  int pid = 0;
  for (int i=0;i<N;++i,vp+=3)											//5. process all read points, add them as possible into the tex:
  {
    int x = S*(vp[0]-vmin[0]);											//transform world->tex coords (i.e. vp[] -> (x,y,z))
    int y = S*(vp[1]-vmin[1]);
    int z = S*(vp[2]-vmin[2]);
	if (x>=tX || y>=tY || z>= tZ) continue;								//xformed point falls outside of tex (due to non tight-fit), skip it
	
    int id = TOID(x, y, z, tX, tY); 
	if (inputVoronoi[id] != MARKER) continue;							//voxel x,y,z already occupied (too dense input points..) so skip current one
																		//since we simply cannot fit 2 points in a voxel
	inputVoronoi[id] = ENCODE(x,y,z); 
	inputPoints[pid] = ENCODE(x,y,z);
	++pid;
  }  
  
  cout<<"Warning: model has "<<N<<" points but voxel res sees only "<<pid<<endl;
  
  delete[] v;

  return pid;
}

