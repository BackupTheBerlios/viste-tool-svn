/*
 * vtkEigenvaluesAnisotropyFilter.cxx
 *
 * 2006-12-26	Tim Peeters
 * - First version.
 *
 * 2011-03-11	Evert van Aart
 * - Added additional comments.
 *
 */


/** Includes */

#include "vtkEigenvaluesToAnisotropyFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkEigenvaluesToAnisotropyFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkEigenvaluesToAnisotropyFilter::vtkEigenvaluesToAnisotropyFilter()
{
	// Set default measure to FA
	this->Measure = AnisotropyMeasures::FA;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkEigenvaluesToAnisotropyFilter::~vtkEigenvaluesToAnisotropyFilter()
{

}


//----------------------------[ ComputeScalar ]----------------------------\\

double vtkEigenvaluesToAnisotropyFilter::ComputeScalar(double eVals[3])
{
	// Use the library functions to compute the AI
	return AnisotropyMeasures::AnisotropyMeasure(this->Measure, eVals);
} 


} // namespace bmia
