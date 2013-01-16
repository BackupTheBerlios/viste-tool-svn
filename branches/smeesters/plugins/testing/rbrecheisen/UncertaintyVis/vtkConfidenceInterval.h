#ifndef __vtkConfidenceInterval_h
#define __vtkConfidenceInterval_h

#include <vtkObject.h>
#include <vector>
#include <string>

class vtkConfidenceHistogram;
class vtkConfidenceIntervalProperties;

class vtkConfidenceInterval : public vtkObject
{
public:

	static vtkConfidenceInterval * New();
	vtkTypeRevisionMacro( vtkConfidenceInterval, vtkObject );

	void SetNumberOfIntervals( int number );
	int GetNumberOfIntervals();

	void SetInterval( int index, float min, float max );
	float * GetInterval( int index );

	void SetRange( float min, float max );
	float * GetRange();

	void SetHistogram( vtkConfidenceHistogram * histogram );
	vtkConfidenceHistogram * GetHistogram();

	void SetSubdivisionToEqualWidth();
	void SetSubdivisionToEqualHistogramArea();
	void SetSubdivisionToCustomWidth();

	void SetChanged( bool changed );
	bool HasChanged();

	vtkConfidenceIntervalProperties * GetProperties();

	std::string WriteToString();
	void ReadFromString( const std::string & text );

private:

	vtkConfidenceInterval();
	virtual ~vtkConfidenceInterval();

	float ComputeConfidenceFromHistogramArea( 
				vtkConfidenceHistogram * histogram, float area );

private:

	typedef struct Interval
	{
		float Min;
		float Max;
	} Interval;

	enum IntervalSubdivision
	{
		SUBDIVISION_EQUAL_WIDTH,
		SUBDIVISION_EQUAL_HISTOGRAM_AREA,
		SUBDIVISION_CUSTOM
	};

	IntervalSubdivision Subdivision; 
	bool Changed;

	float Range[2];

	std::vector< Interval > Intervals;

	vtkConfidenceIntervalProperties * Properties;
	vtkConfidenceHistogram * Histogram;
};

#endif