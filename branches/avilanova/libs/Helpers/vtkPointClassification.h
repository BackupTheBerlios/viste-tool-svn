/**
 * vtkPointClassification.h
 * by Tim Peeters
 *
 * 2009-03-24	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkPointClassification_h
#define bmia_vtkPointClassification_h

#include <vtkPointSetAlgorithm.h>

class vtkImageData;
class vtkPointSet;
class vtkAlgorithmOutput;
class vtkPoints;

namespace bmia {

/**
 * Class for classifying points in a vtkPointSet depending on
 * an input scalar volume and two threshold values for the
 * scalar volume. The two thresholds divide the values in the scalar
 * volume in 3 bins (lower, middle and upper). For each of the points
 * in the input PointSet, the scalar value at the location of that point
 * is looked up in the scalar volume, and depending on the bin that the
 * scalar value goes into, the point is classified in one of the respective
 * output point sets.
 */
class vtkPointClassification : public vtkPointSetAlgorithm
{
public:
  static vtkPointClassification* New();

  vtkSetMacro(UpperThreshold, double);
  vtkGetMacro(UpperThreshold, double);
  vtkSetMacro(LowerThreshold, double);
  vtkGetMacro(LowerThreshold, double);

  vtkPointSet* GetOutputLower() { return this->GetOutput(0); };
  vtkPointSet* GetOutputMiddle() { return this->GetOutput(1); };
  vtkPointSet* GetOutputUpper() { return this->GetOutput(2); };
  vtkAlgorithmOutput* GetOutputPortLower() { return this->GetOutputPort(0); };
  vtkAlgorithmOutput* GetOutputPortMiddle() { return this->GetOutputPort(1); };
  vtkAlgorithmOutput* GetOutputPortUpper() { return this->GetOutputPort(2); };

  // TODO: set this input the proper way.
  void SetInputScalarImage(vtkImageData* image);
  vtkGetObjectMacro(InputScalarImage, vtkImageData);

  /**
   * Set the thresholds such that all points in the given set have scalar values
   * between the thresholds.
   */
  void OptimizeThresholds(vtkPointSet*);
  void OptimizeThresholds(vtkPointSet* pos, vtkPointSet* neg);

protected:
  vtkPointClassification();
  ~vtkPointClassification();

  virtual int RequestData(vtkInformation*, 
                          vtkInformationVector**, 
                          vtkInformationVector*);

  double UpperThreshold;
  double LowerThreshold;

  vtkImageData* InputScalarImage; // input

  /**
   * Get values in this->InputScalarImage in the given points and put them
   * in the values array. values must have length num_points.
   * The number of added values (<= num_points) is returned.
   */
  int GetValuesInPoints(vtkPoints* points, double* values, int num_points);

  void BubbleSort(double* list, int len);

private:

}; // class vtkPointClassification
} // namespace bmia
#endif // bmia_vtkPointClassification_h
