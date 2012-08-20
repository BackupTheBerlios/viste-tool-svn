/**
 * vtkUniformFloatArray.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version, not finished/working yet!
 */

#include "vtkUniformFloatArray.h"
#include <vtkObjectFactory.h>
#include <vtkFloatArray.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformFloatArray);
vtkCxxSetObjectMacro(vtkUniformFloatArray, Value, vtkFloatArray);

vtkUniformFloatArray::vtkUniformFloatArray()
{
  //this->Count = 0;
  //this->Value = NULL;
  this->Value = NULL;
}

vtkUniformFloatArray::~vtkUniformFloatArray()
{
  if (this->Value)
    {
    this->Value->UnRegister(this);
    this->Value = NULL;
    }
}

void vtkUniformFloatArray::SetGlUniformSpecific()
{
  //glUniform1fvARB(this->Location, this->Count, this->Value);

  if (!this->Value)
    {
    vtkWarningMacro(<<"Not passing uniform float array with value NULL.");
    return;
    }

  int components = this->Value->GetNumberOfComponents();
  int tuples = this->Value->GetNumberOfTuples();

  float* array = this->Value->....//HIER VERDER
  switch(tuples)
    {
    case 1:
      {
      glUniform1fv
      break;
      }
    case 2:
      {
      glUniform2fv
      break;
      }
    case 3:
      {
      glUniform3fv
      break;
      }
    case 4:
      {
      glUniform4fv
      break;
      }
    default:
      {
      glUniform1fv
      break;
      }
}

} // namespace bmia
