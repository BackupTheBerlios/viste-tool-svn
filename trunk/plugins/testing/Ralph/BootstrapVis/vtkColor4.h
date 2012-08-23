#ifndef __vtkColor4_h
#define __vtkColor4_h

class vtkColor4
{
public:

	vtkColor4() : r(0), g(0), b(0), a(255)
	{
	};

	vtkColor4( int _r, int _g, int _b, int _a ) : r(_r), g(_g), b(_b), a(_a)
	{
	};

	vtkColor4( const vtkColor4 & color )
	{
		r = color.r;
		g = color.g;
		b = color.b;
		a = color.a;
	};

	void operator = ( const vtkColor4 & color )
	{
		r = color.r;
		g = color.g;
		b = color.b;
		a = color.a;
	};

	~vtkColor4()
	{
	};

	int r;
	int g;
	int b;
	int a;
};

#endif