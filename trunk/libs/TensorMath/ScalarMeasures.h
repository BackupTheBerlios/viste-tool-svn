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


#ifndef bmia_ScalarMeasures_h
#define bmia_ScalarMeasures_h


/** Includes - Custom Files */

#include "AnisotropyMeasures.h"
#include "DistanceMeasures.h"
#include "Invariants.h"

/** Includes - C++ */

#include <assert.h>


namespace bmia {


namespace ScalarMeasures {


using namespace AnisotropyMeasures;

	/** Add three measures that are not part of one of the other classes/namespaces. */
	static const int OOP = AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures + Invariants::numberOfMeasures + 0;	// Out-of-Plane Component 
	static const int MRI = AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures + Invariants::numberOfMeasures + 1;    // MRI
	static const int AD1 = AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures + Invariants::numberOfMeasures + 2;	// Anatomical Data

	/** Compute the total number of scalar measures. */
	static const int numberOfScalarMeasures = AnisotropyMeasures::numberOfMeasures + Distance::numberOfMeasures + Invariants::numberOfMeasures + 3;

	/** Long names of the measures exclusive to this namespace. */
	static const char * longNames[] = 
	{
		"Out-of-plane component",
		"MRI",
		"Anatomical Data"
	};

	/** Short names of these measures. */
	static const char * shortNames[] = 
	{
		"OOP",
		"MRI",
		"AD1"
	};

	/** Return the long/short name of the specified measure.
		@param measure	Desired measure. */

	const char * GetLongName (int measure);
	const char * GetShortName(int measure);


	/** Returns true if the specified measure is an invariant, distance measure,
		or anisotropy measure, respectively. Returns false otherwise.
		@param measure	Desired measure. */

	bool IsInvariant(int measure);
	bool IsDistanceMeasure(int measure);
	bool IsAnisotropyMeasure(int measure);


} // namespace ScalarMeasures


} // namespace bmia


#endif // bmia_ScalarMeasures_h
