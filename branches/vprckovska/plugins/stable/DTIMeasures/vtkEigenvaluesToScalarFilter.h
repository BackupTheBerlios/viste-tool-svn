/*
 * vtkEigenvaluesToScalarFilter.h
 *
 * 2006-02-22	Tim Peeters
 * - First version.
 *
 * 2006-05-12	Tim Peeters
 * - Add progress updates
 *
 * 2011-03-10	Evert van Aart
 * - Added additional comments.
 *
 */


#ifndef bmia_vtkEigenvaluesToScalarFilter_h
#define bmia_vtkEigenvaluesToScalarFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>


namespace bmia {

/** Abstract class that takes a volume of eigenvalues as input and has a scalar 
	volume as output. Subclasses must implement the "ComputeScalar" function,
	which computes a scalar value from the three eigenvalues. 
*/

class vtkEigenvaluesToScalarFilter : public vtkSimpleImageToImageFilter
{
	public:

	protected:

		/** Constructor */
		
		vtkEigenvaluesToScalarFilter() 
		{

		};

		/** Destructor */
	
		~vtkEigenvaluesToScalarFilter() 
		{

		};

		/** Execute the filter.
			@param input	Input eigensystem data.
			@param output	Output scalar data. */

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

		/** Compute a scalar value from three eigenvalues, ordered in descending 
			order of magnitude. Implemented in subclasses.
			@param eVals	Three eigenvalues. */
  
		virtual double ComputeScalar(double eVals[3]) = 0;

	private:


}; // class vtkEigenvaluesToScalarFilter


} // namespace bmia


#endif // bmia_vtkEigenvaluesToScalarFilter_h
