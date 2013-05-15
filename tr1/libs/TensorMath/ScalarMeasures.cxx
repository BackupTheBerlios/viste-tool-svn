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
