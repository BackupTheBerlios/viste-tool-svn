/**
 * vtkTensorToInvariantFilter.cxx
 *
 * 2012-03-16	Ralph Brecheisen
 * - First version.
 *
 */

// Includes

#include "vtkTensorToInvariantFilter.h"

// Includes VTK

#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkTensorToInvariantFilter);

//-----------------------------[ Constructor ]-----------------------------\\

vtkTensorToInvariantFilter::vtkTensorToInvariantFilter()
{
    // Set default invariant measure
    this->Invariant = Invariants::K1;
}

//-----------------------------[ Destructor ]------------------------------\\

vtkTensorToInvariantFilter::~vtkTensorToInvariantFilter()
{
}

double vtkTensorToInvariantFilter::ComputeScalar(double tensor[])
{
    // Use library functions to compute the actual invariant measure
    return Invariants::computeInvariant(this->Invariant, tensor);
}

} // namespace bmia
