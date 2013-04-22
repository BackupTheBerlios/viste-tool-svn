#include <vtkConfidenceIntervalProperties.h>
#include <vtkObjectFactory.h>

#include <sstream>

vtkCxxRevisionMacro( vtkConfidenceIntervalProperties, "$Revision: 1.0 $");
vtkStandardNewMacro( vtkConfidenceIntervalProperties );

///////////////////////////////////////////////////////////////////////////
vtkConfidenceIntervalProperties::vtkConfidenceIntervalProperties()
{
	this->OpacityRange[0] = 0.0f;
	this->OpacityRange[1] = 1.0f;
	this->OutlineOpacityRange[0] = 0.0f;
	this->OutlineOpacityRange[1] = 1.0f;
	this->OutlineThicknessRange[0] = 0.0f;
	this->OutlineThicknessRange[1] = 4.0f;
	this->DilationRange[0] = 0.0f;
	this->DilationRange[1] = 8.0f;
	this->CheckerSizeRange[0] = 0.0f;
	this->CheckerSizeRange[1] = 16.0f;
	this->HoleSizeRange[0] = 0.0f;
	this->HoleSizeRange[1] = 16.0f;
	this->BlurringRadiusRange[0] = 0.0f;
	this->BlurringRadiusRange[1] = 16.0f;
	this->BlurringBrightnessRange[0] = 0.0f;
	this->BlurringBrightnessRange[1] = 4.0f;
	this->NoiseFrequencyRange[0] = 0.0f;
	this->NoiseFrequencyRange[1] = 2.0f;

	this->BlurringEnabled = false;
	this->NoiseEnabled = false;

	this->Active = PROPERTY_OPACITY;
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceIntervalProperties::~vtkConfidenceIntervalProperties()
{
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetNumberOfIntervals( int number )
{
	this->Properties.clear();

	for( int i = 0; i < number; ++i )
	{
		IntervalProperties properties;

		for( int i = 0; i < 3; ++i )
			properties.Color[i] = 1.0f;

		for( int i = 0; i < 3; ++i )
			properties.OutlineColor[i] = 1.0f;

		properties.Opacity = 1.0f;
		properties.OutlineOpacity = 1.0f;
		properties.OutlineThickness = 0.5f;
		properties.Dilation = 0.5f;
		properties.CheckerSize = 0.5f;
		properties.HoleSize = 0.25f;
		properties.BlurringRadius = 0.5f;
		properties.BlurringBrightness = 0.25f;
		properties.NoiseFrequency = 0.5f;
		properties.Enabled = true;

		this->Properties.push_back( properties );
	}
}

///////////////////////////////////////////////////////////////////////////
int vtkConfidenceIntervalProperties::GetNumberOfIntervals()
{
	return this->Properties.size();
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetEnabled( int index, bool enabled )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.Enabled = enabled;
}

///////////////////////////////////////////////////////////////////////////
bool vtkConfidenceIntervalProperties::IsEnabled( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.Enabled;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetColor( int index, float r, float g, float b )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.Color[0] = r;
	properties.Color[1] = g;
	properties.Color[2] = b;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetColor( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	float * color = new float[3];
	for( int i = 0; i < 3; ++i )
		color[i] = properties.Color[i];
	return color;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetOutlineColor( int index, float r, float g, float b )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.OutlineColor[0] = r;
	properties.OutlineColor[1] = g;
	properties.OutlineColor[2] = b;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetOutlineColor( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	float * color = new float[3];
	for( int i = 0; i < 3; ++i )
		color[i] = properties.OutlineColor[i];
	return color;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetOpacityRange( float min, float max )
{
	this->OpacityRange[0] = min;
	this->OpacityRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetOpacityRange()
{
	return this->OpacityRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetOpacity( int index, float opacity )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.Opacity = opacity;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetOpacity( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.Opacity;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetOutlineOpacityRange( float min, float max )
{
	this->OutlineOpacityRange[0] = min;
	this->OutlineOpacityRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetOutlineOpacityRange()
{
	return this->OutlineOpacityRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetOutlineOpacity( int index, float opacity )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.OutlineOpacity = opacity;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetOutlineOpacity( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.OutlineOpacity;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetOutlineThicknessRange( float min, float max )
{
	this->OutlineThicknessRange[0] = min;
	this->OutlineThicknessRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetOutlineThicknessRange()
{
	return this->OutlineThicknessRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetOutlineThickness( int index, float thickness )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.OutlineThickness = thickness;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetOutlineThickness( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.OutlineThickness;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetDilationRange( float min, float max )
{
	this->DilationRange[0] = min;
	this->DilationRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetDilationRange()
{
	return this->DilationRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetDilation( int index, float dilation )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.Dilation = dilation;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetDilation( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.Dilation;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetCheckerSizeRange( float min, float max )
{
	this->CheckerSizeRange[0] = min;
	this->CheckerSizeRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetCheckerSizeRange()
{
	return this->CheckerSizeRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetCheckerSize( int index, float size )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.CheckerSize = size;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetCheckerSize( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.CheckerSize;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetHoleSizeRange( float min, float max )
{
	this->HoleSizeRange[0] = min;
	this->HoleSizeRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetHoleSizeRange()
{
	return this->HoleSizeRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetHoleSize( int index, float size )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.HoleSize = size;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetHoleSize( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.HoleSize;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetBlurringRadiusRange( float min, float max )
{
	this->BlurringRadiusRange[0] = min;
	this->BlurringRadiusRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetBlurringRadiusRange()
{
	return this->BlurringRadiusRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetBlurringRadius( int index, float radius )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.BlurringRadius = radius;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetBlurringRadius( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.BlurringRadius;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetBlurringBrightnessRange( float min, float max )
{
	this->BlurringBrightnessRange[0] = min;
	this->BlurringBrightnessRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetBlurringBrightnessRange()
{
	return this->BlurringBrightnessRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetBlurringBrightness( int index, float brightness )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.BlurringBrightness = brightness;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetBlurringBrightness( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.BlurringBrightness;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetNoiseFrequencyRange( float min, float max )
{
	this->NoiseFrequencyRange[0] = min;
	this->NoiseFrequencyRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceIntervalProperties::GetNoiseFrequencyRange()
{
	return this->NoiseFrequencyRange;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetNoiseFrequency( int index, float frequency )
{
	IntervalProperties & properties = this->Properties.at( index );
	properties.NoiseFrequency = frequency;
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetNoiseFrequency( int index )
{
	IntervalProperties & properties = this->Properties.at( index );
	return properties.NoiseFrequency;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetBlurringEnabled( bool enabled )
{
	this->BlurringEnabled = enabled;
}

///////////////////////////////////////////////////////////////////////////
bool vtkConfidenceIntervalProperties::IsBlurringEnabled()
{
	return this->BlurringEnabled;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetNoiseEnabled( bool enabled )
{
	this->NoiseEnabled = enabled;
}

///////////////////////////////////////////////////////////////////////////
bool vtkConfidenceIntervalProperties::IsNoiseEnabled()
{
	return this->NoiseEnabled;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToOpacity()
{
	this->Active = PROPERTY_OPACITY;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToOutlineOpacity()
{
	this->Active = PROPERTY_OUTLINE_OPACITY;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToOutlineThickness()
{
	this->Active = PROPERTY_OUTLINE_THICKNESS;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToDilation()
{
	this->Active = PROPERTY_DILATION;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToCheckerSize()
{
	this->Active = PROPERTY_CHECKER_SIZE;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToHoleSize()
{
	this->Active = PROPERTY_HOLE_SIZE;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToBlurringRadius()
{
	this->Active = PROPERTY_BLURRING_RADIUS;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToBlurringBrightness()
{
	this->Active = PROPERTY_BLURRING_BRIGHTNESS;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActivePropertyToNoiseFrequency()
{
	this->Active = PROPERTY_NOISE_FREQUENCY;
}

///////////////////////////////////////////////////////////////////////////
int vtkConfidenceIntervalProperties::GetActiveProperty()
{
	return this->Active;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetActiveProperty( int index )
{
	if( index == 0 )
		this->Active = PROPERTY_OPACITY;
	else if( index == 1 )
		this->Active = PROPERTY_OUTLINE_OPACITY;
	else if( index == 2 )
		this->Active = PROPERTY_OUTLINE_THICKNESS;
	else if( index == 3 )
		this->Active = PROPERTY_DILATION;
	else if( index == 4 )
		this->Active = PROPERTY_CHECKER_SIZE;
	else if( index == 5 )
		this->Active = PROPERTY_HOLE_SIZE;
	else if( index == 6 )
		this->Active = PROPERTY_BLURRING_RADIUS;
	else if( index == 7 )
		this->Active = PROPERTY_BLURRING_BRIGHTNESS;
	else if( index == 8 )
		this->Active = PROPERTY_NOISE_FREQUENCY;
	else
		std::cout << "SetActiveProperty() unknown index" << std::endl;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceIntervalProperties::SetValue( int index, float value )
{
	if( this->Active == PROPERTY_OPACITY )
	{
		this->SetOpacity( index, value );
	}
	else if( this->Active == PROPERTY_OUTLINE_OPACITY )
	{
		this->SetOutlineOpacity( index, value );
	}
	else if( this->Active == PROPERTY_OUTLINE_THICKNESS )
	{
		this->SetOutlineThickness( index, value );
	}
	else if( this->Active == PROPERTY_DILATION )
	{
		this->SetDilation( index, value );
	}
	else if( this->Active == PROPERTY_CHECKER_SIZE )
	{
		this->SetCheckerSize( index, value );
	}
	else if( this->Active == PROPERTY_HOLE_SIZE )
	{
		this->SetHoleSize( index, value );
	}
	else if( this->Active == PROPERTY_BLURRING_RADIUS )
	{
		this->SetBlurringRadius( index, value );
	}
	else if( this->Active == PROPERTY_BLURRING_BRIGHTNESS )
	{
		this->SetBlurringBrightness( index, value );
	}
	else if( this->Active == PROPERTY_NOISE_FREQUENCY )
	{
		this->SetNoiseFrequency( index, value );
	}
	else
	{
	}
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceIntervalProperties::GetValue( int index )
{
	if( this->Active == PROPERTY_OPACITY )
	{
		return this->GetOpacity( index );
	}
	else if( this->Active == PROPERTY_OUTLINE_OPACITY )
	{
		return this->GetOutlineOpacity( index );
	}
	else if( this->Active == PROPERTY_OUTLINE_THICKNESS )
	{
		return this->GetOutlineThickness( index );
	}
	else if( this->Active == PROPERTY_DILATION )
	{
		return this->GetDilation( index );
	}
	else if( this->Active == PROPERTY_CHECKER_SIZE )
	{
		return this->GetCheckerSize( index );
	}
	else if( this->Active == PROPERTY_HOLE_SIZE )
	{
		return this->GetHoleSize( index );
	}
	else if( this->Active == PROPERTY_BLURRING_RADIUS )
	{
		return this->GetBlurringRadius( index );
	}
	else if( this->Active == PROPERTY_BLURRING_BRIGHTNESS )
	{
		return this->GetBlurringBrightness( index );
	}
	else if( this->Active == PROPERTY_NOISE_FREQUENCY )
	{
		return this->GetNoiseFrequency( index );
	}
	else
	{
	}

	return 0.0f;
}
