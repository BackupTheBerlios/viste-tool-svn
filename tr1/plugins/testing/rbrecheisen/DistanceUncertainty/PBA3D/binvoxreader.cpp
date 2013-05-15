#include <string>
#include <fstream>
#include <iostream>
#include "gpudefs.h"
#include "pba/pba3D.h"


using namespace std;

typedef unsigned char byte;


template <typename T> inline T& get_value(T* voxels, int x, int y, int z, int wxh, int width)
{
  return voxels[x * wxh + z * width + y];  // wxh = width * height = d * d
} 




static bool warnings_on = false;



static int read_binvox(string filespec,unsigned char* inputVolume,float* output_vertices,int& nvertices,float* vmin,float* vmax,const int MAXVERTS)
{

  ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);
  if (!input->good()) return -1;														//   File cannot be opened

  string line;																			//1. Read header
  *input >> line;  // #binvox
  if (line.compare("#binvox") != 0) 
  {
    cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
    delete input; return 0;
  }
  int version;
  *input >> version;

  int depth, height, width;
  depth = -1;
  int done = 0;
  while(input->good() && !done) 
  {
    *input >> line;
    if (line.compare("data") == 0) done = 1;
    else if (line.compare("dim") == 0) 
      *input >> depth >> height >> width;
    else 
	{
      if (warnings_on) cout << "Warning: unrecognized keyword [" << line << "], skipping" << endl;
      char c;
      do { c = input->get(); } while(input->good() && (c != '\n')); // skip until end of line
    }
  }
  
  if (!done) { cout << "  error reading header" << endl; return 0; }
  
  if (depth == -1) { cout << "  missing dimensions in header" << endl; return 0; }

  int size = width * height * depth;
 
  byte value, count;																	//2. Read binary data
  int index = 0, end_index = 0, nr_voxels = 0;
  
  input->unsetf(ios::skipws);															// need to read every byte now (!)
  *input >> value;																		// read the linefeed char

  while((end_index < size) && input->good()) 
  {
    *input >> value >> count;
    if (!input->good()) break;

	end_index = index + count;
	if (end_index > size) return 0;
	for(int i=index; i < end_index; ++i) inputVolume[i] = (value)? 255:0;
	if (value) nr_voxels += count;
	index = end_index;
  }

  input->close();
  delete input;

  const int wxh = width*height;
  const int BOUNDARY = 100;																//This should be OK as we use only 0,255 in the volume now
  vmin[0]=vmin[1]=vmin[2] =  10000; 
  vmax[0]=vmax[1]=vmax[2] = -10000; 
  float* vertices = output_vertices;
  nvertices = 0;
  bool first_time = true;

  for(int i=1;i<width-1;++i)															//3. Mark all boundary voxels with some special value BOUNDARY
   for(int j=1;j<height-1;++j)															//   Boundary: fg voxels which have a bg neighbor
    for(int k=1;k<depth-1;++k)
	{
	   unsigned char& v = get_value(inputVolume,i,j,k,wxh,width);
	   if (v)
	   {
	      if (!get_value(inputVolume,i-1,j,k,wxh,width) || !get_value(inputVolume,i+1,j,k,wxh,width) ||
			  !get_value(inputVolume,i,j-1,k,wxh,width) || !get_value(inputVolume,i,j+1,k,wxh,width) ||
			  !get_value(inputVolume,i,j,k-1,wxh,width) || !get_value(inputVolume,i,j,k+1,wxh,width))
		  {
			 v = BOUNDARY;
		     if (nvertices==MAXVERTS && first_time)
			 {
			   cout<<"Warning: reached max boundary vertices, ignoring rest: "<<nvertices<<endl;
			   first_time = false;
			   continue;
			 }
		 
			 *vertices++ = i; *vertices++ = j; *vertices++ = k;
			 ++nvertices;
			 if (vmin[0]>i) vmin[0]=i; if (vmax[0]<i) vmax[0]=i;
			 if (vmin[1]>j) vmin[1]=j; if (vmax[1]<j) vmax[1]=j;
			 if (vmin[2]>k) vmin[2]=k; if (vmax[2]<k) vmax[2]=k;
	      }
	   }
	}

  
  return 1;
}



int readBinvox(const char* f,int tX, int tY, int tZ,unsigned char* inputVolume,int* inputVoronoi,int* inputPoints, bool tight_fit)
//Read vertices, store them in encoded textures inputVoronoi, inputPoints
{
  float vmin[3],vmax[3],sz[3];
  const int MAXVERTS = 1000000;
  float* v = new float[MAXVERTS*3];
 
  int N = 0; 
  int ret = read_binvox(f,inputVolume,v,N,vmin,vmax,MAXVERTS);			//reads volume, keeps only boundary points in v[0..N-1]
  if (ret==-1) return -1;
  if (!ret) return 0;
  
  vmin[0]=vmin[1]=vmin[2]=0;
  //vmax[0]=tX; vmax[1]=tY; vmax[2]=tZ;									//!!This should have sth to do with the fitting...

  cout<<"PBA readBinvox: "<<vmax[0]<<" "<<vmax[1]<<" "<<vmax[2]<<endl;

  sz[0] = vmax[0]-vmin[0];												//2. determine bbox sizes
  sz[1] = vmax[1]-vmin[1];
  sz[2] = vmax[2]-vmin[2];

  float S=1;
  if (tight_fit)
  {
    S = tX/sz[0];														//3. determine scaling factor from world (bbox) to texture-box (tX,tY,tZ)
																		//a) tight fit: max uniform scaling factor which would still fit the entire object in the tex
    if (tY/sz[1]<S) S=tY/sz[1];
    if (tZ/sz[2]<S) S=tZ/sz[2];
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
  
  if (N>pid) cout<<"Warning: model has "<<N<<" points but voxel res sees only "<<pid<<endl;
  
  delete[] v;
  return pid;
}




void writeBinvox(const unsigned char* f, int width, int height, int depth, const char* fname)
{
    ofstream* output = new ofstream(fname, ios::out | ios::binary);

    // write header
    *output << "#binvox 1" << endl;
    //  *output << "bbox [-1,1][-1,1][-1,1]" << endl;  // no use for 'bbox'
    //  *output << "dim [" << depth << "," << height << "," << width << "]" << endl;
    //  *output << "type RLE" << endl;
    *output << "dim " << depth << " " << height << " " << width << endl;
    *output << "translate " << 0 <<" "<<0<<" "<< 0 << endl;
    *output << "scale " << 1.0 << endl;
    *output << "data" << endl;

    byte value;
    byte count;
    int index = 0;
    int bytes_written = 0;
    int size = width*depth*height;
    int wxh  = width * height;
    int total_ones = 0;

    //must iterate over voxels in the order of the binvox file, i.e. y,z,x

    while (index < size)
    {
        value = f[index];
        byte nvalue = value;
        count = 0;
        while((index < size) && (count < 255) && (nvalue==value))
        {
            index++;
            count++;
            nvalue = f[index];
        }
        //    value = 1 - value;
        
        if (value) total_ones += count;
        *output << value << count;  // inverted...
        bytes_written += 2;
    }

    output->close();
    delete output;
}

