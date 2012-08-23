/**
 * vtkTensorToScalarFilter.h
 *
 * 2012-03-16	Ralph Brecheisen
 * - First version.
 *
 */

#ifndef bmia_vtkTensorToScalarFilter_h
#define bmia_vtkTensorToScalarFilter_h

// Includes VTK

#include <vtkSimpleImageToImageFilter.h>
#include <vtkImageData.h>

namespace bmia {

/** Abstract class that takes a tensor volume and produces a scalar volume
    as output. Subclasses must implement the 'ComputeScalar' function, which
    computes a scalar value for each tensor value.
*/

class vtkTensorToScalarFilter : public vtkSimpleImageToImageFilter
{
public:

protected:

    /** Constructor. */

	vtkTensorToScalarFilter()
	{
	};

    /** Destructor. */

	~vtkTensorToScalarFilter()
	{
	};

    /** Execute the filter.
        @param input    Input tensor data.
        @param output   Output scalar data. */

	virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

    /** Compute a scalar value from the given tensor value. Implemented
        in subclasses.
        @param tensor   The tensor value. */

	virtual double ComputeScalar(double tensor[6]) = 0;

private:

}; // class vtkTensorToScalarFilter

} // namespace bmia

#endif // bmia_vtkTensorToScalarFilter_h
