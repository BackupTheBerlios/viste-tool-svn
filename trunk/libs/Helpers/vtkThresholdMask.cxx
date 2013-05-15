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
