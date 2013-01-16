/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "vtkMathExtensions.h"

#include <cassert>
#include <vtkMath.h>

#define PI	3.141592654
#define PI2	(2*PI)

////////////////////////////////////////////////////////////////////
bool vtkMathExtensions::IsPowerOfTwo(int number)
{
	if ( (number & (number - 1)) == 0 )
	{
		return true;
	}
	
	return false;
}

////////////////////////////////////////////////////////////////////
int vtkMathExtensions::PowerOfTwo(int number)
{
	assert(number > 0);
	assert(vtkMathExtensions::IsPowerOfTwo(number));
	
	int n = number;
	int k = 0;
	
	while ( n > 1 ) 
	{ 
		n = n >> 1;
		k++;
	}
	
	return k;
}

////////////////////////////////////////////////////////////////////
double vtkMathExtensions::UnitGaussian(double value)
{
	double sigma = 1 / sqrt(PI2);
	
	double exponent = -(value * value) / (2 * sigma * sigma);
	double res = exp(exponent);
	
	// NOT NEEDED: res *= 1 / (sqrt(PI2) * sigma);	
	return res;
}

////////////////////////////////////////////////////////////////////
void vtkMathExtensions::MinMaxInt(int *list, int size, int &min, int &max)
{
	int tmpMin =  VTK_INT_MAX;
	int tmpMax = -VTK_INT_MAX;
	
	for ( int i = 0; i < size; i++ )
	{
		if ( list[i] < tmpMin ) tmpMin = list[i];
		if ( list[i] > tmpMax ) tmpMax = list[i];
	}
	
	min = tmpMin;
	max = tmpMax;
}

////////////////////////////////////////////////////////////////////
void vtkMathExtensions::MinMaxDouble(double *list, int size, double &min, double &max)
{
	double tmpMin =  VTK_DOUBLE_MAX;
	double tmpMax = -VTK_DOUBLE_MAX;
	
	for ( int i = 0; i < size; i++ )
	{
		if ( list[i] < tmpMin ) tmpMin = list[i];
		if ( list[i] > tmpMax ) tmpMax = list[i];
	}
	
	min = tmpMin;
	max = tmpMax;
}

////////////////////////////////////////////////////////////////////
int vtkMathExtensions::TotalInt(int *list, int size)
{
	int total = 0;
	
	for ( int i = 0; i < size; i++ )
	{
		total += list[i];
	}
	
	return total;
}

////////////////////////////////////////////////////////////////////
double vtkMathExtensions::TotalDouble(double *list, int size)
{
	double total = 0;
	
	for ( int i = 0; i < size; i++ )
	{
		total += list[i];
	}
	
	return total;
}

////////////////////////////////////////////////////////////////////
void vtkMathExtensions::MeanAndStandardDeviation(double *values, int size, double &mean, double &sd)
{
	double total = 0.0;
	
	for ( int i = 0; i < size; i++ )
	{
		total += values[i];
	}

	double tmpMean = 0.0;
	if ( size > 0 )
		tmpMean = total / size;

	double variance = 0.0;
	
	for ( int i = 0; i < size; i++ )
	{
		double difference = values[i] - tmpMean;
		variance += (difference*difference);
	}

	if ( size > 0 )
		variance /= size;

	double tmpStDev = sqrt(variance);

	mean = tmpMean;
	sd = tmpStDev;
}
