/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
