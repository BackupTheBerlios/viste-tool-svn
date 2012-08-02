/*
 * vtkEigenvaluesToAnisotropyFilter.h
 *
 * 2006-12-26	Tim Peeters
 * - First version.
 *
 * 2011-03-11	Evert van Aart
 * - Added additional comments.
 *
 */


#ifndef bmia_vtkEigenvaluesToAnisotropyFilter_h
#define bmia_vtkEigenvaluesToAnisotropyFilter_h


/** Includes - Custom Files */

#include "vtkEigenvaluesToScalarFilter.h"
#include "TensorMath/AnisotropyMeasures.h"

/** Includes - VTK */

#include <vtkObjectFactory.h>


namespace bmia {


/** This class computes an anisotropy measure value using the three eigenvalues
	of a tensor as input. Looping through all input voxels is done in the parent
	class, "vtkEigenvaluesToScalarFilter". The actual measure computations are
	done in the "AnisotropyMeasures" class. 
*/

class vtkEigenvaluesToAnisotropyFilter: public vtkEigenvaluesToScalarFilter
{
	public:
  
		/** Constructor Call */

		static vtkEigenvaluesToAnisotropyFilter * New();

		/** Specify the anisotropy measure for the output. */

		vtkSetClampMacro(Measure, int, 0, AnisotropyMeasures::numberOfMeasures);
		
		/** Get the current measure. */

		vtkGetMacro(Measure, int);

	protected:
	
		/** Constructor */

		vtkEigenvaluesToAnisotropyFilter();

		/** Destructor */

		~vtkEigenvaluesToAnisotropyFilter();

		/** Compute a scalar value, based on the three eigenvalues of a tensor. 
			@param eVals	Eigenvalues. */
		
		virtual double ComputeScalar(double eVals[3]);

		/** Current anisotropy measure. */
  
		int Measure;

}; // class vtkEigenvaluesToAnisotropyFilter


} // namespace bmia


#endif // bmia_vtkEigenvaluesToAnisotropyFilter_h
