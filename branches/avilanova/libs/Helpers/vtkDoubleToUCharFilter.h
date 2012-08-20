/**
 * vtkDoubleToUCharFilter.h
 * by Tim Peeters
 *
 * 2005-03-29	Tim Peeters
 * - First version
 *
 * 2005-06-03	Tim Peeters
 * - Use bmia namespace
 */

#ifndef bmia_vtkDoubleToUCharFilter_h
#define bmia_vtkDoubleToUCharFilter_h

#include <vtkSimpleImageToImageFilter.h>

namespace bmia {

/**
 * Datasets containing double data in the range <0..1> are converted
 * to unsigned char datasets with values from 0 to 255.
 *
 * vtkImageShiftScale is not used because that always assumes scalar
 * data with one component per tuple, while this is not always the case.
 * I didn't try to patch that one because it already has changes in
 * CVS version and upgrading to CVS is not practical for me.
 */
class vtkDoubleToUCharFilter : public vtkSimpleImageToImageFilter
{
public:
  static vtkDoubleToUCharFilter* New();

protected:
  vtkDoubleToUCharFilter();
  ~vtkDoubleToUCharFilter();

  virtual void SimpleExecute(vtkImageData* input, vtkImageData* output);

private:
  vtkDoubleToUCharFilter(const vtkDoubleToUCharFilter&); // Not implemented
  void operator=(const vtkDoubleToUCharFilter&); // Not implemented
};

} // namespace bmia

#endif
