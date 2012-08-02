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
