#pragma once

// This file:				Various general-purpose utilities used throughout the code



struct Color																		//Visualization related stuff
{ 
	float r,g,b; 
	
	Color() {}; 
	Color(float r_,float g_,float b_):r(r_),g(g_),b(b_) {} 
};


inline float myMin(float a,float b) { return (a>b)? b:a; }
inline float myMax(float a,float b) { return (a>b)? a:b; }

inline void rainbowColormap(float value,float* rgb)                               //maps value to rainbow hue
{
   const float dx = 0.8;
   value = (value<0.0)? 0 : (value>1.0)? 1 : value;
   value = (6-2*dx)*value + dx;
   rgb[0] = myMax(0.0,(3-fabs(value-4)-fabs(value-5))/2.0);
   rgb[1] = myMax(0.0,(4-fabs(value-2)-fabs(value-4))/2.0);
   rgb[2] = myMax(0.0,(3-fabs(value-1)-fabs(value-2))/2.0);
}

inline void slice2volume(int x,int y,int& i,int& j,int& k,int slice_dir,int slice)	//Converts a 2D coord in the currently visualized slice to its 3D voxel
{
	switch (slice_dir)
	{
	  case 0: i=slice; j=y; k=x; break;
	  case 1: i=y; j=slice; k=x; break;
	  case 2: i=x; j=y; k=slice; break;
    }
}


