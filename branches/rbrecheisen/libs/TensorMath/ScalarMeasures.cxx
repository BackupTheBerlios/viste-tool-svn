/**
 * ScalarMeasures.h
 *
 * 2006-12-26	Tim Peeters
 * - First version
 *
 * 2007-04-24	Tim Peeters
 * - Add distance measures
 *
 * 2010-12-16	Evert van Aart
 * - First version for DTITool3.
 *
 */


/** Includes */

#include "ScalarMeasures.h"


namespace bmia {


namespace ScalarMeasures {


//-----------------------------[ GetLongName ]-----------------------------\\

const char * GetLongName(int measure)
{
	// Check if the measure index is in range
	assert(!(measure < 0 || measure >= numberOfScalarMeasures));

	// Anisotropy measures
	if (measure < AnisotropyMeasures::numberOfMeasures) 
	{
		return AnisotropyMeasures::GetLongName(measure);
	}

	// Distance measures
	if (measure < AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures)
	{
		return Distance::longNames[measure - AnisotropyMeasures::numberOfMeasures];
	}

	// Invariants
	if(measure < AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures + Invariants::numberOfMeasures)
	{
		return Invariants::longNames[measure - AnisotropyMeasures::numberOfMeasures - Distance::numberOfMeasures];
	}

	// Measures exclusive to this namespace
	return ScalarMeasures::longNames[measure - AnisotropyMeasures::numberOfMeasures - Distance::numberOfMeasures - Invariants::numberOfMeasures];
}


//-----------------------------[ GetShortName ]----------------------------\\

const char * GetShortName(int measure)
{
	// Check if the measure index is in range
	assert(!(measure < 0 || measure >= numberOfScalarMeasures));

	// Anisotropy measures
	if (measure < AnisotropyMeasures::numberOfMeasures) 
	{
		return AnisotropyMeasures::GetShortName(measure);
	}

	// Distance measures
	if (measure < AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures)
	{
		return Distance::shortNames[measure - AnisotropyMeasures::numberOfMeasures];
	}

	// Invariants
	if(measure < AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures + Invariants::numberOfMeasures)
	{
		return Invariants::shortNames[measure - AnisotropyMeasures::numberOfMeasures - Distance::numberOfMeasures];
	}

	// Measures exclusive to this namespace
	return ScalarMeasures::shortNames[measure - AnisotropyMeasures::numberOfMeasures - Distance::numberOfMeasures - Invariants::numberOfMeasures];
}


//-------------------------[ IsAnisotropyMeasure ]-------------------------\\

bool IsAnisotropyMeasure(int measure)
{
	// Check if the measure index is in the correct range
	if( (measure >= 0) && (measure < AnisotropyMeasures::numberOfMeasures))
	{
		return true;
	}

	return false;
}


//--------------------------[ IsDistanceMeasure ]--------------------------\\

bool IsDistanceMeasure(int measure)
{
	// Check if the measure index is in the correct range
	int m = measure - AnisotropyMeasures::numberOfMeasures;
	if( (measure >= 0) && (m < Distance::numberOfMeasures))
	{
		return true;
	}

	return false;
}


//-----------------------------[ IsInvariant ]-----------------------------\\

bool IsInvariant(int measure)
{
	// Check if the measure index is in the correct range
	int m = measure - AnisotropyMeasures::numberOfMeasures - Distance::numberOfMeasures;
	if( (measure >= 0) && (m < Invariants::numberOfMeasures))
	{
		return true;
	}

	return false;
}


} // namespace ScalarMeasures


} // namespace bmia
