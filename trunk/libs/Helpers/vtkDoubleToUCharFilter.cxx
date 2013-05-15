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
 * vtkDoubleToUCharFilter.cxx
 * by Tim Peeters
 *
 * 2005-03-29	Tim Peeters
 * - First version
 *
 * 2005-05-19	Tim Peeters
 * - Replaced asserts by vtkErrorMacros and vtkWarningMacros.
 *
 * 2005-06-03	Tim Peeters
 * - Use bmia namespace
 */

#include "vtkDoubleToUCharFilter.h"

#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>

namespace bmia {

vtkStandardNewMacro(vtkDoubleToUCharFilter);

vtkDoubleToUCharFilter::vtkDoubleToUCharFilter()
{
  // nothing
}

vtkDoubleToUCharFilter::~vtkDoubleToUCharFilter()
{
  // nothing
}

void vtkDoubleToUCharFilter::SimpleExecute(vtkImageData* input,
					     vtkImageData* output)
{
  vtkDebugMacro(<<"starting Execute()");
  if (this->GetInput() == NULL)
    {
    vtkErrorMacro(<<"No input!");
    return;
    } // if

  vtkPointData* inPD = input->GetPointData();

  if (inPD == NULL)
    {
    vtkErrorMacro(<<"Input point data is NULL!");
    return;
    }
  vtkDoubleArray* inScalars = (vtkDoubleArray*) inPD->GetScalars();
  if (inScalars == NULL)
    {
    vtkErrorMacro(<<"Input scalars is NULL!");
    return;
    }

  int numPts = input->GetNumberOfPoints();
  if (numPts != inScalars->GetNumberOfTuples())
    {
    vtkErrorMacro(<<"Number of scalars/tuples in input data and input scalars"
	<<" does not match!");
    return;
    }

  int numComps = inScalars->GetNumberOfComponents();
  if (numComps < 1)
    {
    vtkErrorMacro(<<"Number of components of input scalars must be"
		  <<" at least ONE!");
    return;
    }

  int numValues = numPts * numComps;
 
  if ( !inScalars || numPts < 1 )
    {
    vtkErrorMacro(<<"No data to convert!");
    return;
    } // if

  if (output == NULL)
    {
    vtkErrorMacro(<<"Output is NULL!");
    return;
    }
  vtkPointData* outPD = output->GetPointData();

  if (outPD == NULL)
    {
    vtkErrorMacro(<<"Output point data is NULL!");
    return;
    }

  output->CopyStructure(input);
  output->SetScalarType(VTK_UNSIGNED_CHAR);
  output->SetUpdateExtentToWholeExtent();
  
  vtkUnsignedCharArray* outScalars = vtkUnsignedCharArray::New();
  outScalars->SetNumberOfComponents(numComps);
  outScalars->SetNumberOfTuples(numPts);
  outScalars->SetNumberOfValues(numValues);
  if (input->GetNumberOfScalarComponents() != numComps)
    {
    vtkErrorMacro("Number of scalar components of input is "
		<< input->GetNumberOfScalarComponents() << " while "
		<<"input scalars number of components is "
		<< numComps <<". They should be equal!");
    }

  output->SetNumberOfScalarComponents(numComps);

  outPD->SetScalars(outScalars);

  int comp;
  double in_double; double out_double; unsigned char out_uchar;
  unsigned char* out_array = new unsigned char[numValues];
  vtkDebugMacro(<<"Looping from 0 to "<<numValues);

  for (vtkIdType ptId = 0; ptId < numPts; ptId++)
    {
    for (comp = 0; comp < numComps; comp++)
      {
      in_double = inScalars->GetComponent(ptId, comp);
      if (in_double <= 0.0)
        out_uchar = 0;
      else if (in_double >= 1.0)
        out_uchar = 255;
      else
        {
        out_double = in_double * 255.0;
        out_uchar = (unsigned char) out_double;
	// <-- incorrect rounding. this clamps the value to the largest integer
	// value that is at most out_double.
        }
      //if (in_double < 0.0) in_double = 0.0;
      //else if (in_double > 1.0) in_double = 1.0;
      //out_double = in_double * 255.0;
      //out_uchar = (unsigned char) out_double;
      out_array[ptId*numComps + comp] = out_uchar;
/*
cout<<"in_double = "<<in_double<<", ";
cout<<"out_double = "<<out_double<<", ";
cout<<"out_uchar = "<<(unsigned short)out_uchar<<endl;
*/
      }
    } // for
  outScalars->SetArray(out_array, numValues, 0);
  out_array = NULL; // will be deleted by outScalars
} // Execute();

} // namespace bmia
