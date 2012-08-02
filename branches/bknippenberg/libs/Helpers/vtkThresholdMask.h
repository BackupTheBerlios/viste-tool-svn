/**
 * vtkThresholdMask.h
 * by Tim Peeters
 *
 * 2009-03-21	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkThresholdMask_h
#define bmia_vtkThresholdMask_h

#include <vtkSimpleImageToImageFilter.h>

namespace bmia {

/**
 * Class for masking a ImageData using a threshold on a
 * second ImageData with the same dimensions.
 * The output data will be a copy of input data, but with all voxels where
 * in the ThresholdInput the value is smaller than Threshold, the output data
 * will have all zeroes.
 */
class vtkThresholdMask : public vtkSimpleImageToImageFilter
{
public:
  static vtkThresholdMask* New();

  /**
   * Set the scalar volume that is used for masking
   * It must have a scalar array filled with double values.
   * If ThresholdInput is NULL then the output data will be an unmasked
   * copy of the input data.
   */
  void SetThresholdInput(vtkImageData* w);
  vtkGetObjectMacro(ThresholdInput, vtkImageData);

  vtkSetMacro(Threshold, double);
  vtkGetMacro(Threshold, double);

protected:

  vtkThresholdMask();
  ~vtkThresholdMask();

  vtkImageData* ThresholdInput;
  double Threshold;

  virtual void SimpleExecute(vtkImageData* input, vtkImageData* output);

private:

}; // class


} // namespace bmia
#endif // bmia_vtkThresholdMask_h
