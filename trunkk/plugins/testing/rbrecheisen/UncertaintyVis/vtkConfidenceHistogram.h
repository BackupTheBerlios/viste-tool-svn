#ifndef __vtkConfidenceHistogram_h
#define __vtkConfidenceHistogram_h

#include <vtkObject.h>

class vtkConfidenceHistogram : public vtkObject
{
public:

	static vtkConfidenceHistogram * New();
	vtkTypeRevisionMacro( vtkConfidenceHistogram, vtkObject );

	void SetNumberOfBins( int number );
	int GetNumberOfBins();

	void SetDataRange( float min, float max );
	float * GetDataRange();

	void SetData( float * values, int number );
	float * GetData();

	float * GetMinMax();
	float * GetProbabilities();

protected:

	vtkConfidenceHistogram();
	virtual ~vtkConfidenceHistogram();

private:

	int NumberOfDataValues;
	int NumberOfBins;
	float * Data;
	float   DataRange[2];
	float * Probabilities;
	bool Modified;

private:

	vtkConfidenceHistogram( const vtkConfidenceHistogram & );
	void operator = ( const vtkConfidenceHistogram & );
};

#endif
