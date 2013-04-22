#ifndef __vtkConfidenceIntervalProperties_h
#define __vtkConfidenceIntervalProperties_h

#include <vtkObject.h>

#include <vector>
#include <string>

class vtkConfidenceIntervalProperties : public vtkObject
{
public:

	static vtkConfidenceIntervalProperties * New();
	vtkTypeRevisionMacro( vtkConfidenceIntervalProperties, vtkObject );

	void SetNumberOfIntervals( int number );
	int GetNumberOfIntervals();

	void SetColor( int index, float r, float g, float b );
	float * GetColor( int index );

	void SetOutlineColor( int index, float r, float g, float b );
	float * GetOutlineColor( int index );

	void SetOpacityRange( float min, float max );
	float * GetOpacityRange();
	void SetOpacity( int index, float opacity );
	float GetOpacity( int index );

	void SetOutlineOpacityRange( float min, float max );
	float * GetOutlineOpacityRange();
	void SetOutlineOpacity( int index, float opacity );
	float GetOutlineOpacity( int index );

	void SetOutlineThicknessRange( float min, float max );
	float * GetOutlineThicknessRange();
	void SetOutlineThickness( int index, float thickness );
	float GetOutlineThickness( int index );

	void SetDilationRange( float min, float max );
	float * GetDilationRange();
	void SetDilation( int index, float dilation );
	float GetDilation( int index );

	void SetCheckerSizeRange( float min, float max );
	float * GetCheckerSizeRange();
	void SetCheckerSize( int index, float size );
	float GetCheckerSize( int index );

	void SetHoleSizeRange( float min, float max );
	float * GetHoleSizeRange();
	void SetHoleSize( int index, float size );
	float GetHoleSize( int index );

	void SetBlurringRadiusRange( float min, float max );
	float * GetBlurringRadiusRange();
	void SetBlurringRadius( int index, float radius );
	float GetBlurringRadius( int index );

	void SetBlurringBrightnessRange( float min, float max );
	float * GetBlurringBrightnessRange();
	void SetBlurringBrightness( int index, float brightness );
	float GetBlurringBrightness( int index );

	void SetNoiseFrequencyRange( float min, float max );
	float * GetNoiseFrequencyRange();
	void SetNoiseFrequency( int index, float frequency );
	float GetNoiseFrequency( int index );

	void SetEnabled( int index, bool enabled );
	bool IsEnabled( int index );

	void SetBlurringEnabled( bool enabled );
	bool IsBlurringEnabled();

	void SetNoiseEnabled( bool enabled );
	bool IsNoiseEnabled();

	void SetActivePropertyToOpacity();
	void SetActivePropertyToOutlineOpacity();
	void SetActivePropertyToOutlineThickness();
	void SetActivePropertyToDilation();
	void SetActivePropertyToCheckerSize();
	void SetActivePropertyToHoleSize();
	void SetActivePropertyToBlurringRadius();
	void SetActivePropertyToBlurringBrightness();
	void SetActivePropertyToNoiseFrequency();

	void SetActiveProperty( int index );
	int GetActiveProperty();

	void SetValue( int index, float value );
	float GetValue( int index );

private:

	vtkConfidenceIntervalProperties();
	virtual ~vtkConfidenceIntervalProperties();

private:

	typedef struct IntervalProperties
	{
		float Color[3];
		float OutlineColor[3];
		float Opacity;
		float OutlineOpacity;
		float OutlineThickness;
		float Dilation;
		float CheckerSize;
		float HoleSize;
		float BlurringRadius;
		float BlurringBrightness;
		float NoiseFrequency;
		bool Enabled;
	} IntervalProperties;

	enum ActiveProperty
	{
		PROPERTY_OPACITY,
		PROPERTY_OUTLINE_OPACITY,
		PROPERTY_OUTLINE_THICKNESS,
		PROPERTY_DILATION,
		PROPERTY_CHECKER_SIZE,
		PROPERTY_HOLE_SIZE,
		PROPERTY_BLURRING_RADIUS,
		PROPERTY_BLURRING_BRIGHTNESS,
		PROPERTY_NOISE_FREQUENCY
	};

	std::vector< IntervalProperties > Properties;
	ActiveProperty Active;

	float OpacityRange[2];
	float OutlineOpacityRange[2];
	float OutlineThicknessRange[2];
	float DilationRange[2];
	float CheckerSizeRange[2];
	float HoleSizeRange[2];
	float BlurringRadiusRange[2];
	float BlurringBrightnessRange[2];
	float NoiseFrequencyRange[2];

	bool BlurringEnabled;
	bool NoiseEnabled;
};

#endif