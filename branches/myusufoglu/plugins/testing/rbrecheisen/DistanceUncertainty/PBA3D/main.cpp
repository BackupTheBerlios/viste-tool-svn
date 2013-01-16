#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <map>
#include <string>
#include <GLUT/glut.h>

#include "gpudefs.h"
#include "readers.h"
#include "utils.h"
#include "myrandom.h"
#include "pba/pba3D.h"
#include "dtengine.h"

using namespace std;


int nVertices    = 500000;					//Max # input points we can process

enum MODE									//Visualization-related parameters:
{ VORONOI, EDT, CMAP, SEGMENT, MODE_END };

float		   LMAX	= -0.5;					//Segmentation threshold for inflation
MODE		   mode = VORONOI;				//What info we currently draw in the viewer
int			   slice = 1;					//Position of slice along the slicing axis
int			   slice_dir = 2;				//Axis along which we slice (0=X,1=Y,2=Z). The 3D model is viewed along that axis
int			   winSizeX = 512,				//Sizes of the viewing window
			   winSizeY = 512;
map<int,Color> m,m2;						//Different colors for the Voronoi regions
bool		   show_points = true;			//If true, sites are shown too
float*		   tex2D=0;						//Client-side buffer for the current slice
unsigned char* outputSeg=0;					//Segmentation of the EDT for given threshold LMAX
GLuint		   texture;						//Texture-id for the slice texture




void generateTexture(float* voronoi,MODE mode,float alpha,float gray) 
// Generate 2D texture for current 'slice', depending on 'mode'
{
    int tx, ty, tz; 
	float D_max = 0;

	for (int ii = 0; ii < fboSizeX; ii++)				//x coord in 2D tex
	   for (int jj = 0; jj < fboSizeY; jj++)			//y coord in 2D tex
	   {
		int i,j,k;
		slice2volume(ii,jj,i,j,k,slice_dir,slice);		//find 3D voxel i,j,k corresponding to 2D slice pixel ii,jj
		Color rgb;			
		int id = (i>=fboSizeX || j>= fboSizeY || k >= fboSizeZ)? MARKER : TOID(i,j,k,fboSizeX,fboSizeY);			//Find closest-side-it to i,j,k
		int iid = jj * fboSizeX + ii;					//offset in 2D texture of current pixel to color
		
		if (id!=MARKER)
		{
		  if (mode==SEGMENT)								//Visualize segmentation: use simple b/w colormap
		  {
		    unsigned char v = outputSeg[id];
		    rgb.r = rgb.g = rgb.b = float(v);
		  }
		  else //mode!=SEGMENT
		  {	
		    DECODE(outputVoronoi[id],tx,ty,tz);			//Get coords of closest-site to current point i,j,k

		    if (mode==VORONOI)							//Visualize Voronoi diagram or FT
		    { 								
		      int vid = TOID(tx,ty,tz,fboSizeX,fboSizeY);	//id for closest site
		      map<int,Color>::iterator it = m.find(vid);	//map id-of-closest-site to unique color for a site
		      if (it==m.end())
		      {
			   rgb = Color(myrandom(),myrandom(),myrandom());
			   m.insert(make_pair(vid,rgb));
		      }  else rgb = it->second;
		    }
		    else if (mode==EDT || mode==CMAP)				//Visualize DT (gray or colormap)
		    {
			  //int dx = tx - i, dy = ty - j, dz = tz - k; 
			  //float D = sqrt(dx * dx + dy * dy + dz * dz);
			  float D = outputDT[id];
			  rgb.r = rgb.g = rgb.b = D;
			  if (D>D_max) D_max=D;
		    }
		  }
		}
		else rgb = Color(0,0,0);						//We're at a marker voxel: show it black
		
		voronoi[iid * 4 + 0] = rgb.r; 
		voronoi[iid * 4 + 1] = rgb.g; 
		voronoi[iid * 4 + 2] = rgb.b; 
		voronoi[iid * 4 + 3] = 1; 
	   }
	   
    if (mode==EDT)											//Visualize the DT or FT: use gray colormap
        for (int ii = 0; ii < fboSizeX; ii++)				//x coord in 2D tex
           for (int jj = 0; jj < fboSizeY; jj++)			//y coord in 2D tex
		   {
            int iid = jj * fboSizeX + ii;					//offset in 2D texture of current pixel to color
			float D = voronoi[iid * 4 + 0];					//use max intensity for visualization...
			float a = 1;
			if (LMAX>=0)
			{
			   D = (D>LMAX)? 1:gray;
			   a = (D==1)? alpha:1;
			}
		    else
			   D = D/float(D_max);
            voronoi[iid * 4 + 0] = D; 
            voronoi[iid * 4 + 1] = D; 
            voronoi[iid * 4 + 2] = D; 
            voronoi[iid * 4 + 3] = a; 
		   }
		
    if (mode==CMAP)											//Visualize the DT or FT: use rainbow colormap   
        for (int ii = 0; ii < fboSizeX; ii++)				//x coord in 2D tex
           for (int jj = 0; jj < fboSizeY; jj++)			//y coord in 2D tex
		   {
            int iid = jj * fboSizeX + ii;					//offset in 2D texture of current pixel to color
			float D = voronoi[iid * 4 + 0];					//map DT to color via colormap
			float rgb[3];
			D = D/D_max;
			rainbowColormap(D,rgb);
            voronoi[iid * 4 + 0] = rgb[0]; 
            voronoi[iid * 4 + 1] = rgb[1]; 
            voronoi[iid * 4 + 2] = rgb[2]; 
		   }

    // Copy image to a 2D texture (tex-id is already allocated)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fboSizeX, fboSizeY, 0, GL_RGBA, GL_FLOAT, voronoi); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}





void drawPoints() 
{
    glColor3d(1.0, 1.0, 1.0); 
    glPointSize(1);
	glBegin(GL_POINTS); 
	
    for (int i = 0; i < nVertices; i++)
	{
	    int nx,ny,nz; DECODE(inputPoints[i],nx,ny,nz);				//get the 3D coords of i-th site
		int x,y,z;
		switch (slice_dir)
		{
		  case 0: x=nz; y=ny; z=nx; break;
		  case 1: x=nz; y=nx; z=ny; break; 
		  case 2: x=nx; y=ny; z=nz; break; 
		}
		
		if (!(slice_dir==0 && nx==slice) && !(slice_dir==1 && ny==slice) && !(slice_dir==2 && nz==slice)) continue;
		{
          glVertex2d(double(x) / fboSizeX, double(y) / fboSizeY); 
		}
	}

    glEnd(); 
}



void glutDisplay() 
{
    int dX,dY;
	switch (slice_dir)
	{
	case 0: dX = fboSizeZ; dY = fboSizeY; break;
	case 1: dX = fboSizeZ; dY = fboSizeX; break;
	case 2: dX = fboSizeX; dY = fboSizeY; break;
	}
	
	float sf = myMin(dX,dY);
	
	glViewport(0, 0, winSizeX, winSizeY); 

    glClearColor(0.33, 0.0, 0.0, 0.0); 
    glClear(GL_COLOR_BUFFER_BIT); 

    glMatrixMode(GL_PROJECTION); 
    glLoadIdentity(); 
    gluOrtho2D(0.0, dX/sf, 0.0, dY/sf); 

    glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity(); 
    glScalef(dX/sf, dY/sf, 1.0);
    glTranslatef(0, 0, 0);

    glDisable(GL_LIGHTING); 
    glDisable(GL_DEPTH_TEST); 
    glEnable(GL_TEXTURE_2D); 
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glBindTexture(GL_TEXTURE_2D, texture); 

    //Test code for multi-slice vis
    int ss = slice;
	int thick_z = 1;   //!!! should depend on mode
	int smin = ss-thick_z,smax = ss+thick_z;
	for(int s=smin;s<smax;++s)
	{
	   float c = float(s-smin)/(smax-smin);
	   if (s>=0 && s<fboSizeZ)	
	   {
	    slice = s;
		generateTexture(tex2D,mode,0,c);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 0.0); 
		glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 0.0); 
		glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0); 
		glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 1.0); 		
		glEnd(); 
	   }	
	}
	slice = ss;	
			
    glBindTexture(GL_TEXTURE_2D, 0); 
    glDisable(GL_TEXTURE_2D);
	glDisable(GL_BLEND);
	
    if (show_points) drawPoints(); 

    glutSwapBuffers();     
}




void glut_kbfunc(unsigned char ch,int,int)
{
    switch(ch)
	{
	  case '+': case '=': slice++; if (slice>=fboSizeZ) slice=0; break;
	  case '-': case '_': slice--; if (slice<0) slice=fboSizeZ-1; break;
	  case 'x': slice_dir++; if (slice_dir>2) slice_dir=0;  
			  cout<<"Slicing: "<<((slice_dir==0)? "YZ" : (slice_dir==1)? "XZ" : "XY")<<endl; 
			  break;
	  case ' ': mode = (MODE)(1+mode); if (mode==MODE_END) mode = (MODE)(0); break;
	  case 'p': show_points = !show_points; break;
	  case '.': LMAX+=0.5; pba3DThreshold(outputVoronoi, outputSeg, LMAX); cout<<"Thr: "<<LMAX<<endl; break;
	  case ',': LMAX-=0.5; pba3DThreshold(outputVoronoi, outputSeg, LMAX); cout<<"Thr: "<<LMAX<<endl; break;
	  //case 'd': saveVolume(outputDT,inputVolume,fboSizeX,fboSizeY,fboSizeZ,"DT.volume"); cout<<"Volume saved"<<endl; break;
	  case 27: exit(0);
	}

	glutPostRedisplay();
}

void glut_mousefunc(int,int,int x,int y)
{
    int scale = myMin(winSizeX/fboSizeX,winSizeY/fboSizeY);					//!!!
	x /= scale; y /= scale; y = fboSizeY - y;
    int i,j,k; int tx,ty,tz;
	slice2volume(x,y,i,j,k,slice_dir,slice);

	int id = TOID(i,j,k,fboSizeX,fboSizeY);			//
    DECODE(outputVoronoi[id],tx,ty,tz);				//Get closest site tx,ty,tz to current point i,j,k
    int dx = tx - i, dy = ty - j, dz = tz - k; 
    float D = sqrt(float(dx * dx + dy * dy + dz * dz)); 				

    cout<<"x "<<x<<" y "<<y<<": "<<i<<" "<<j<<" "<<k<<" DT "<<D<<endl;	
}


// Read input data 
void readData(const string& inpf, bool fit_model,int* inputVoronoi)
{
    if (!inpf.empty())																				//Input file given? Try read it
	{
	  if (inpf.rfind(".ply")!=string::npos)
	     nVertices = readPLY(inpf.c_str(),fboSizeX,fboSizeY,fboSizeZ,inputVoronoi,inputPoints,fit_model);
      else
	  if (inpf.rfind(".binvox")!=string::npos)	  
         nVertices = readBinvox(inpf.c_str(),fboSizeX,fboSizeY,fboSizeZ,inputVolume,inputVoronoi,inputPoints,fit_model);
	  else cout<<"Error: file "<<inpf<<" of unrecognized format"<<endl;
	  if (nVertices<0)
	  {
	     cout<<"Error: file "<<inpf<<" cannot be opened"<<endl;
		 exit(0);
	  }
	}  

	if (!nVertices)																					//No input file specified? Use some random test data
	{
       randinit(0);
	   generateRandomPoints(inputVoronoi,inputPoints,fboSizeX, fboSizeY, fboSizeZ, nVertices); 
	}
}






int main(int argc_, char **argv_)
{
    bool	fit_model = false;
	string	inpf;
	string  thr_volume,dt_volume;
	float   thr;
	int     nx=256,ny=256,nz=256;										//Computational volume: 512^3 requires 1 GB GPU RAM (or more)

    int argc = argc_-1; char** argv = argv_+1;
	for(;argc;argc--,argv++)											//Parse cmdline args
	{
	   string a = argv[0];
	   if (a=="-fit")
	      fit_model = true;
	   else if (a=="-d")
	   {
	      nx = atoi(argv[1]);
	      ny = atoi(argv[2]);
	      nz = atoi(argv[3]);
		  argv += 3; argc -= 3;
	   }
	   else if (a=="-dt")												//-dt: compute DT of input model, save it to .volume file
	   {
	      dt_volume = argv[1];
          argv += 1; argc -= 1;
	   }
	   else if (a=="-t")												//-t: compute thresholding of DT of input, save it to .volume file
	   {
	      thr        = atof(argv[1]);
	      thr_volume = argv[2];
          argv += 2; argc -= 2;
	   }
	   else if (a[0]!='-')
	      inpf = argv[0];	  
	}

    dtcuda_initialization(nVertices,nx,ny,nz);							//1. Initialize data structures (give max # sites and volume size)

    int* inputVoronoi = 0;										
	dtcuda_bindBuffer(inputVoronoi);									//Bind inputVoronoi to an allocated buffer
    readData(inpf,fit_model,inputVoronoi);								//2. Read input data

    cout<<"Computing 3D DT (size: "<<fboSizeX<<"x"<<fboSizeY<<"x"<<fboSizeZ<<", points "<<nVertices<<")..."<<endl; 
	pba3DVoronoiDiagram(inputVoronoi, outputVoronoi, phase1Band, phase2Band, phase3Band); 
																		//3. Compute Voronoi diagram

    dtcuda_bindBuffer(outputDT);										//Bind outputDT to an allocated buffer
    pba3DDT(outputVoronoi,outputDT);									//4. Compute DT from the Voronoi diagram

	if (!dt_volume.empty())												//5. Save results
	   saveVolume(outputDT,inputVolume,fboSizeX,fboSizeY,fboSizeZ,dt_volume.c_str()); 
	else if (!thr_volume.empty())
	   saveThresholdedVolume(outputDT,thr,inputVolume,fboSizeX,fboSizeY,fboSizeZ,thr_volume.c_str());  


    bool interactive = dt_volume.empty() && thr_volume.empty();			//Determines if the tool is to be run into interactive or batch mode	
	if (interactive)
	{
      glutInitWindowPosition(0, 0); 
      glutInitWindowSize(winSizeX, winSizeY); 
      glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA); 
      glutInit(&argc_,argv_);
	  tex2D	= new float[fboSizeX * fboSizeY * 4];
      outputSeg = new unsigned char[fboSizeX * fboSizeY * fboSizeZ]; 
	
	  pba3DThreshold(outputVoronoi, outputSeg, LMAX*LMAX);				//Finally, do one segmentation step
      glutCreateWindow("3D Model Analysis");
      glGenTextures(1,&texture);										//Generate stuff for visualization
      glutDisplayFunc(glutDisplay); 
      glutMouseFunc(glut_mousefunc); 
	  glutKeyboardFunc(glut_kbfunc);
      glutMainLoop(); 
	  delete[] tex2D;
	  delete[] outputSeg;
	}
	
    dtcuda_deinitialization();
	return 0;
}