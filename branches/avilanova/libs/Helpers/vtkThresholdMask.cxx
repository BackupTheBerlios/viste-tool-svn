/**
 * vtkThresholdMask.cxx
 * by Tim Peeters
 *
 * 2009-03-21	Tim Peeters
 * - First version
 */

#include "vtkThresholdMask.h"
#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>

namespace bmia {

vtkStandardNewMacro(vtkThresholdMask);
vtkCxxSetObjectMacro(vtkThresholdMask, ThresholdInput, vtkImageData);

vtkThresholdMask::vtkThresholdMask()
{
  this->ThresholdInput = NULL;
  this->Threshold = 0.0;
}

vtkThresholdMask::~vtkThresholdMask()
{
  // nothing to do.
}

void vtkThresholdMask::SimpleExecute(vtkImageData* input, vtkImageData* output)
{
  this->SetProgressText("Masking data...");
  this->UpdateProgress(0.0);

  if (!input) vtkErrorMacro("No input!");
  if (!output) vtkErrorMacro("No output!");

  // copy intput to output
  output->CopyStructure(input);
  output->DeepCopy(input);
  
  if (!this->ThresholdInput)
    {
    vtkDebugMacro(<<"No threshold input. Output will be unmasked copy of input.");
    return;
    }

  vtkPointData* inPD = input->GetPointData();
  if (!inPD) vtkErrorMacro("No input point data!");

  vtkPointData* outPD = output->GetPointData();
  if (!outPD) vtkErrorMacro("No output point data!");

  vtkPointData* tPD = this->ThresholdInput->GetPointData();
  if (!tPD) vtkErrorMacro("No threshold point data!");

  vtkDataArray* tArray = tPD->GetScalars();
  if (!tArray) vtkErrorMacro("No threshold scalar array!");

  int numArrays = outPD->GetNumberOfArrays();

  int numComponents;
  int numTuples = tArray->GetNumberOfTuples();
  vtkDataArray* outArray = NULL;

  vtkIdType ptId; float v;
  double* emptyValue = NULL;

  for (int i = 0; i < numArrays; i++)
    {
    outArray = outPD->GetArray(i);
	outArray->SetName(inPD->GetArray(i)->GetName());

    if (numTuples != outArray->GetNumberOfTuples())
      {
      vtkErrorMacro("Number of tuples in input array differs from number of tuples in output array.");
      return;
      } // if numTuples

    cout<<"masking with threshold " <<this->Threshold<<endl;
    int numMasked = 0;

    numComponents = outArray->GetNumberOfComponents();
    emptyValue = new double[numComponents];
    for (int j = 0; j < numComponents; j++) emptyValue[j] = 0.0;

    for (ptId = 0; ptId < numTuples; ptId++)
      {
      v = tArray->GetTuple1(ptId);
      if (v < this->Threshold)
        {
        outArray->SetTuple(ptId, emptyValue);
	numMasked++;
	}
      if (ptId % 50000 == 0) this->UpdateProgress((float)(ptId+i*numArrays) / (float)(numTuples*numArrays));
      //else outArray->SetTuple(ptId, input->GetPointData()->GetArray(i)->GetTuple(ptId));
      } // for ptId
    cout<<"------------------- Masked "<<numMasked<<" of "<<numTuples<<" voxels"<<endl;
    this->UpdateProgress(1.0);
    } // for i

} // SimpleExecute


} // namespace bmia
