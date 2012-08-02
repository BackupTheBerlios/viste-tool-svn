/**
 * vtkEigensystemToGPU.h
 * by Tim Peeters
 *
 * 2007-01-10	Tim Peeters
 * - First version. Copy of vtkTensorToEigensystemFilter.h
 */

#ifndef bmia_vtkEigensystemToGPU_h
#define bmia_vtkEigensystemToGPU_h

#include <vtkSimpleImageToImageFilter.h>

namespace bmia {

/**
 * Input is tensor data.
 *
 * Computes eigensystem from tensor data, and shift/scales the components of
 * the eigenvalues to 0..1 instead of -1..1 so that they can be put in a texture.
 * Also, Cl, Cp, Cs is added as a fourth component
 */
class vtkEigensystemToGPU : public vtkSimpleImageToImageFilter
{
public:
  static vtkEigensystemToGPU *New();

protected:

  vtkEigensystemToGPU() {};
  ~vtkEigensystemToGPU() {};

  virtual void SimpleExecute(vtkImageData* input, vtkImageData* output);

private:

}; // class vtkEigensystemToGPU

} // namespace bmia

#endif // bmia_vtkEigensystemToGPU_h
