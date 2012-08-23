/**
 * vtkTensorToInvariantFilter.h
 *
 * 2012-03-16	Ralph Brecheisen
 * - First version.
 *
 */

#ifndef bmia_vtkTensorToInvariantFilter_h
#define bmia_vtkTensorToInvariantFilter_h

// Includes

#include "vtkTensorToScalarFilter.h"

// Includes tensor math

#include "TensorMath/Invariants.h"

namespace bmia {

/** This class computes scalar invariants based on input tensor data. Looping
    through all voxels is done in the parent class 'vtkTensorToScalarFilter'.
    The actual invariant computations are done in the 'Invariants' class. */

class vtkTensorToInvariantFilter : public vtkTensorToScalarFilter
{
public:

    /** Constructor call. */

    static vtkTensorToInvariantFilter * New();

    /** Specify the invariant measure for the output. */

    vtkSetClampMacro(Invariant, int, 0, Invariants::numberOfMeasures);

    /** Get the current invariant measure. */

    vtkGetMacro(Invariant, int);

protected:

    /** Constructor. */

    vtkTensorToInvariantFilter();

    /** Destructor. */

    ~vtkTensorToInvariantFilter();

    /** Compute the scalar invariant based on the given tensor value.
        @param tensor   The tensor value. */

    virtual double ComputeScalar(double tensor[]);

private:

    /** Current invariant measure. */

    int Invariant;

}; // class vtkTensorToInvariantFilter

} // namespace bmia

#endif // bmia_vtkTensorToInvariantFilter_h
