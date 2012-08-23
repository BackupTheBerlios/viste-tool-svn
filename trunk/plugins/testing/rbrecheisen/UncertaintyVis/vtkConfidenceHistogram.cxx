#include <vtkConfidenceHistogram.h>
#include <vtkObjectFactory.h>
#include <assert.h>

vtkCxxRevisionMacro( vtkConfidenceHistogram, "$Revision: 1.0 $");
vtkStandardNewMacro( vtkConfidenceHistogram );

///////////////////////////////////////////////////////////////////////////
vtkConfidenceHistogram::vtkConfidenceHistogram()
{
	this->NumberOfDataValues = 0;
	this->NumberOfBins = 2;
	this->Data = 0;
	this->Probabilities = 0;
	this->DataRange[0] = 0.0f;
	this->DataRange[1] = 1.0f;
	this->Modified = true;
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceHistogram::~vtkConfidenceHistogram()
{
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceHistogram::SetNumberOfBins( int size )
{
	this->NumberOfBins = size;
	this->Modified = true;
}

///////////////////////////////////////////////////////////////////////////
int vtkConfidenceHistogram::GetNumberOfBins()
{
	return this->NumberOfBins;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceHistogram::SetDataRange( float minimum, float maximum )
{
	this->DataRange[0] = minimum;
	this->DataRange[1] = maximum;
	this->Modified = true;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceHistogram::GetDataRange()
{
	return this->DataRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceHistogram::SetData( float * data, int size )
{
	this->Data = data;
	this->NumberOfDataValues = size;
	this->Modified = true;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceHistogram::GetData()
{
	return this->Data;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceHistogram::GetMinMax()
{
	float * probabilities = this->GetProbabilities();
	float minimum = 1.0f;
	float maximum = 0.0f;
	for( int i = 0; i < this->GetNumberOfBins(); ++i )
	{
		if( probabilities[i] < minimum )
			minimum = probabilities[i];
		if( probabilities[i] > maximum )
			maximum = probabilities[i];
	}

	float * minMax = new float[2];
	minMax[0] = minimum;
	minMax[1] = maximum;
	return minMax;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceHistogram::GetProbabilities()
{
	if( this->Modified == false )
		return this->Probabilities;

	if( this->Probabilities )
		delete [] this->Probabilities;
	this->Probabilities = new float[this->NumberOfBins];
	for( int i = 0; i < this->NumberOfBins; ++i )
		this->Probabilities[i] = 0.0f;

	for( int i = 0; i < this->NumberOfDataValues; ++i )
	{
		// Skip values that are outside the data range

		float value = this->Data[i];
		if( value < this->DataRange[0] || value > this->DataRange[1] )
			continue;

		// Project value back to range between [0,1] so we can
		// compute correct bin numbers

		value = (value - this->DataRange[0]) / (this->DataRange[1] - this->DataRange[0]);
		int binNr = static_cast< int >(
			value * (this->NumberOfBins - 1) );

		// Ensure that bin number is valid

		if( binNr < 0 ) binNr = 0;
		if( binNr > this->NumberOfBins - 1 )
			binNr = this->NumberOfBins - 1;

		this->Probabilities[binNr] += 1;
	}

	for( int i = 0; i < this->NumberOfBins; ++i )
		this->Probabilities[i] /= this->NumberOfDataValues;
	this->Modified = false;

	return this->Probabilities;
}
