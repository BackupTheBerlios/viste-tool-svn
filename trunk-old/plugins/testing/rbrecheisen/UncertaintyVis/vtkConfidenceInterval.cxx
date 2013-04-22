#include <vtkConfidenceInterval.h>
#include <vtkConfidenceHistogram.h>
#include <vtkConfidenceIntervalProperties.h>

#include <vtkObjectFactory.h>

#include <sstream>

vtkCxxRevisionMacro( vtkConfidenceInterval, "$Revision: 1.0 $");
vtkStandardNewMacro( vtkConfidenceInterval );

///////////////////////////////////////////////////////////////////////////
vtkConfidenceInterval::vtkConfidenceInterval()
{
	this->Properties = vtkConfidenceIntervalProperties::New();
	this->Subdivision = SUBDIVISION_EQUAL_WIDTH;
	this->Range[0] = 0.0f;
	this->Range[1] = 1.0f;
	this->Histogram = 0;
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceInterval::~vtkConfidenceInterval()
{
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetNumberOfIntervals( int number )
{
	this->Intervals.clear();

	for( int i = 0; i < number; ++i )
	{
		Interval interval;

		interval.Min = 0.0f;
		interval.Max = 1.0f;

		this->Intervals.push_back( interval );
	}

	this->Properties->SetNumberOfIntervals( number );
	this->SetChanged( true );
}

///////////////////////////////////////////////////////////////////////////
int vtkConfidenceInterval::GetNumberOfIntervals()
{
	return this->Intervals.size();
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetRange( float min, float max )
{
	this->Range[0] = min;
	this->Range[1] = max;
	
	if( this->Histogram )
	{
		this->Histogram->SetDataRange( min, max );
	}
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceInterval::GetRange()
{
	float * range = new float[2];
	range[0] = this->Range[0];
	range[1] = this->Range[1];
	return range;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetInterval( int index, float min, float max )
{
	Interval & interval = this->Intervals.at( index );
	interval.Min = min;
	interval.Max = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceInterval::GetInterval( int index )
{
	Interval & interval = this->Intervals.at( index );
	float * values = new float[2];
	values[0] = interval.Min;
	values[1] = interval.Max;
	return values;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetHistogram( vtkConfidenceHistogram * histogram )
{
	this->Histogram = histogram;
	this->Histogram->SetDataRange( this->Range[0], this->Range[1] );
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceHistogram * vtkConfidenceInterval::GetHistogram()
{
	return this->Histogram;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetSubdivisionToEqualWidth()
{
	float increment = (this->Range[1] - this->Range[0]) /
		this->GetNumberOfIntervals();
	float next = this->Range[0];

	//float increment = 1.0f / this->GetNumberOfIntervals();
	//float next = 0.0f;
	for( int i = 0; i < this->GetNumberOfIntervals(); ++i )
	{
		Interval & interval = this->Intervals.at( i );
		interval.Min = next;
		interval.Max = next + increment;
		next += increment;
	}

	this->Subdivision = SUBDIVISION_EQUAL_WIDTH;
	this->SetChanged( true );
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetSubdivisionToEqualHistogramArea()
{
	if( this->Histogram )
	{
		float areaIncrement = 1.0f / this->GetNumberOfIntervals();
		float area = areaIncrement;
		float min = 0.0f;
		float max = this->ComputeConfidenceFromHistogramArea( this->Histogram, area );
		for( int i = 0; i < this->GetNumberOfIntervals(); ++i )
		{
			Interval & interval = this->Intervals.at( i );
			interval.Min = min;
			interval.Max = max;
			area += areaIncrement;
			min = max;
			max = this->ComputeConfidenceFromHistogramArea( this->Histogram, area );
		}

		this->Subdivision = SUBDIVISION_EQUAL_HISTOGRAM_AREA;
		this->SetChanged( true );
	}
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetSubdivisionToCustomWidth()
{
	this->Subdivision = SUBDIVISION_CUSTOM;
	this->SetChanged( true );
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceInterval::ComputeConfidenceFromHistogramArea( vtkConfidenceHistogram * histogram, float area )
{
	float * probabilities = histogram->GetProbabilities();
	int nrBins = histogram->GetNumberOfBins();

	if( probabilities == 0 || nrBins == 0 )
		return 0.0f;
	if( area > 1.0f )
		return 1.0f;

	float binSize = 1.0f / nrBins;
//	float binSize = (this->Range[1] - this->Range[0]) / nrBins;
	float accumArea = 0.0f;
	float value = 0.0f;

	for( int i = 0; i < nrBins; ++i )
	{
		accumArea += probabilities[i];
		if( accumArea > area )
		{
			float height = probabilities[i] / binSize;
			float diffArea = accumArea - area;
			float width = diffArea / height;
			value += (binSize - width);
			break;
		}
		else
		{
			value += binSize;
		}
	}

	return value;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::SetChanged( bool changed )
{
	this->Changed = changed;
}

///////////////////////////////////////////////////////////////////////////
bool vtkConfidenceInterval::HasChanged()
{
	return this->Changed;
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceIntervalProperties * vtkConfidenceInterval::GetProperties()
{
	return this->Properties;
}

///////////////////////////////////////////////////////////////////////////
std::string vtkConfidenceInterval::WriteToString()
{
	std::stringstream stream;

	int     nrIntervals = this->GetNumberOfIntervals();
	int     subdivision = this->Subdivision;
	float * range = this->Range;
	float * opacityRange = this->Properties->GetOpacityRange();
	float * outlineOpacityRange = this->Properties->GetOutlineOpacityRange();
	float * outlineThicknessRange = this->Properties->GetOutlineThicknessRange();
	float * dilationRange = this->Properties->GetDilationRange();
	float * checkerSizeRange = this->Properties->GetCheckerSizeRange();
	float * holeSizeRange = this->Properties->GetHoleSizeRange();
	float * blurringRadiusRange = this->Properties->GetBlurringRadiusRange();
	float * blurringBrightnessRange = this->Properties->GetBlurringBrightnessRange();
	float * noiseFrequencyRange = this->Properties->GetNoiseFrequencyRange();
	int     active = this->Properties->GetActiveProperty();
	bool    blurringEnabled = this->Properties->IsBlurringEnabled();
	bool    noiseEnabled = this->Properties->IsNoiseEnabled();

	stream << nrIntervals << std::endl;
	stream << subdivision << std::endl;
	stream << range[0] << std::endl;
	stream << range[1] << std::endl;
	stream << opacityRange[0] << std::endl;
	stream << opacityRange[1] << std::endl;
	stream << outlineOpacityRange[0] << std::endl;
	stream << outlineOpacityRange[1] << std::endl;
	stream << outlineThicknessRange[0] << std::endl;
	stream << outlineThicknessRange[1] << std::endl;
	stream << dilationRange[0] << std::endl;
	stream << dilationRange[1] << std::endl;
	stream << checkerSizeRange[0] << std::endl;
	stream << checkerSizeRange[1] << std::endl;
	stream << holeSizeRange[0] << std::endl;
	stream << holeSizeRange[1] << std::endl;
	stream << blurringRadiusRange[0] << std::endl;
	stream << blurringRadiusRange[1] << std::endl;
	stream << blurringBrightnessRange[0] << std::endl;
	stream << blurringBrightnessRange[1] << std::endl;
	stream << noiseFrequencyRange[0] << std::endl;
	stream << noiseFrequencyRange[1] << std::endl;
	stream << blurringEnabled << std::endl;
	stream << noiseEnabled << std::endl;
	stream << active << std::endl;

	vtkConfidenceIntervalProperties * properties = this->Properties;

	for( int i = 0; i < nrIntervals; ++i )
	{
		Interval & interval = this->Intervals.at( i );

		float min = interval.Min;
		float max = interval.Max;

		float * color = properties->GetColor( i );
		float * outlineColor = properties->GetOutlineColor( i );
		float opacity = properties->GetOpacity( i );
		float outlineOpacity = properties->GetOutlineOpacity( i );
		float outlineThickness = properties->GetOutlineThickness( i );
		float dilation = properties->GetDilation( i );
		float checkerSize = properties->GetCheckerSize( i );
		float holeSize = properties->GetHoleSize( i );
		float blurringRadius = properties->GetBlurringRadius( i );
		float blurringBrightness = properties->GetBlurringBrightness( i );
		float noiseFrequency = properties->GetNoiseFrequency( i );
		bool enabled = properties->IsEnabled( i );

		stream << i << std::endl;
		stream << min << std::endl;
		stream << max << std::endl;
		stream << color[0] << std::endl;
		stream << color[1] << std::endl;
		stream << color[2] << std::endl;
		stream << outlineColor[0] << std::endl;
		stream << outlineColor[1] << std::endl;
		stream << outlineColor[2] << std::endl;
		stream << opacity << std::endl;
		stream << outlineOpacity << std::endl;
		stream << outlineThickness << std::endl;
		stream << dilation << std::endl;
		stream << checkerSize << std::endl;
		stream << holeSize << std::endl;
		stream << blurringRadius << std::endl;
		stream << blurringBrightness << std::endl;
		stream << noiseFrequency << std::endl;
		stream << enabled << std::endl;
	}

	return stream.str();
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceInterval::ReadFromString( const std::string & text )
{
	int   nrIntervals;
	int   subdivision;
	float range[2];
	float opacityRange[2];
	float outlineOpacityRange[2];
	float outlineThicknessRange[2];
	float dilationRange[2];
	float checkerSizeRange[2];
	float holeSizeRange[2];
	float blurringRadiusRange[2];
	float blurringBrightnessRange[2];
	float noiseFrequencyRange[2];
	bool  blurringEnabled;
	bool  noiseEnabled;
	int   active;

	std::stringstream stream( text );

	stream >> nrIntervals;
	stream >> subdivision;
	stream >> range[0];
	stream >> range[1];
	stream >> opacityRange[0];
	stream >> opacityRange[1];
	stream >> outlineOpacityRange[0];
	stream >> outlineOpacityRange[1];
	stream >> outlineThicknessRange[0];
	stream >> outlineThicknessRange[1];
	stream >> dilationRange[0];
	stream >> dilationRange[1];
	stream >> checkerSizeRange[0];
	stream >> checkerSizeRange[1];
	stream >> holeSizeRange[0];
	stream >> holeSizeRange[1];
	stream >> blurringRadiusRange[0];
	stream >> blurringRadiusRange[1];
	stream >> blurringBrightnessRange[0];
	stream >> blurringBrightnessRange[1];
	stream >> noiseFrequencyRange[0];
	stream >> noiseFrequencyRange[1];
	stream >> blurringEnabled;
	stream >> noiseEnabled;
	stream >> active;

	this->SetRange( range[0], range[1] );
	this->SetNumberOfIntervals( nrIntervals );

	vtkConfidenceIntervalProperties * properties = this->Properties;

	properties->SetOpacityRange( opacityRange[0], opacityRange[1] );
	properties->SetOutlineOpacityRange( outlineOpacityRange[0], outlineOpacityRange[1] );
	properties->SetOutlineThicknessRange( outlineThicknessRange[0], outlineThicknessRange[1] );
	properties->SetDilationRange( dilationRange[0], dilationRange[1] );
	properties->SetCheckerSizeRange( checkerSizeRange[0], checkerSizeRange[1] );
	properties->SetHoleSizeRange( holeSizeRange[0], holeSizeRange[1] );
	properties->SetBlurringRadiusRange( blurringRadiusRange[0], blurringRadiusRange[1] );
	properties->SetBlurringBrightnessRange( blurringBrightnessRange[0], blurringBrightnessRange[1] );
	properties->SetNoiseFrequencyRange( noiseFrequencyRange[0], noiseFrequencyRange[1] );
	properties->SetBlurringEnabled( blurringEnabled );
	properties->SetNoiseEnabled( noiseEnabled );
	properties->SetActiveProperty( active );

	for( int i = 0; i < nrIntervals; ++i )
	{
		int   index;
		float min;
		float max;
		float color[3];
		float outlineColor[3];
		float opacity;
		float outlineOpacity;
		float outlineThickness;
		float dilation;
		float checkerSize;
		float holeSize;
		float blurringRadius;
		float blurringBrightness;
		float noiseFrequency;
		bool enabled;

		stream >> index;
		stream >> min;
		stream >> max;
		stream >> color[0];
		stream >> color[1];
		stream >> color[2];
		stream >> outlineColor[0];
		stream >> outlineColor[1];
		stream >> outlineColor[2];
		stream >> opacity;
		stream >> outlineOpacity;
		stream >> outlineThickness;
		stream >> dilation;
		stream >> checkerSize;
		stream >> holeSize;
		stream >> blurringRadius;
		stream >> blurringBrightness;
		stream >> noiseFrequency;
		stream >> enabled;

		Interval & interval = this->Intervals.at( i );

		interval.Min = min;
		interval.Max = max;

		properties->SetColor( i, color[0], color[1], color[2] );
		properties->SetOutlineColor( i, outlineColor[0], outlineColor[1], outlineColor[2] );
		properties->SetOpacity( i, opacity );
		properties->SetOutlineOpacity( i, outlineOpacity );
		properties->SetOutlineThickness( i, outlineThickness );
		properties->SetDilation( i, dilation );
		properties->SetCheckerSize( i, checkerSize );
		properties->SetHoleSize( i, holeSize );
		properties->SetBlurringRadius( i, blurringRadius );
		properties->SetBlurringBrightness( i, blurringBrightness );
		properties->SetNoiseFrequency( i, noiseFrequency );
		properties->SetEnabled( i, enabled );
	}

	//if( subdivision == 0 )
	//	this->SetSubdivisionToEqualWidth();
	//else if( subdivision == 1 )
	//	this->SetSubdivisionToEqualHistogramArea();
	//else if( subdivision == 2 )
	//	this->SetSubdivisionToCustomWidth();
	//else
	//	std::cout << "unknown subdivision" << std::endl;

	this->SetSubdivisionToCustomWidth();
}
