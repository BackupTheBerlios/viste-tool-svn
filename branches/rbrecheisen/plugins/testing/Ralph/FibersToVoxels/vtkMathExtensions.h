#ifndef __vtkMathExtensions_h
#define __vtkMathExtensions_h

#include "vtkMath.h"

class vtkMathExtensions : public vtkMath
{
public:

	static bool    IsPowerOfTwo(int number);
	static int     PowerOfTwo(int number);
	static double  UnitGaussian(double value);
	static void    MinMaxInt(int *list, int size, int &min, int &max);
	static void    MinMaxDouble(double *list, int size, double &min, double &max);
	static int     TotalInt(int *list, int size);
	static double  TotalDouble(double *list, int size);
	static void    MeanAndStandardDeviation(double *values, int size, double &mean, double &sd);
};

#endif
