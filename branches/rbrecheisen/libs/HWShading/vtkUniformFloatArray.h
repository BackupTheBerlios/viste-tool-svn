/**
 * vtkUniformFloatArray.h
 *
 * 2005-05-17	Tim Peeters
 * - First version, not finished/working yet!
 */

#ifndef bmia_vtkUniformFloatArray_h
#define bmia_vtkUniformFloatArray_h

#include "vtkShaderUniform.h"

class vtkFloatArray;

namespace bmia {

/**
 * Class for representing uniform float arrays.
 */
class vtkUniformFloatArray: public vtkShaderUniform
{
public:
  static vtkUniformFloatArray* New();

  //vtkSetVector3Macro(Value, float);
  //vtkGetVector3Macro(Value, float);
  vtkSetValue(vtkFloatArray* array);
  vtkFloatArray* vtkObjectGetMacro();

protected:
  vtkUniformFloatArray();
  ~vtkUniformFloatArray();

  virtual void SetGlUniformSpecific();

private:
  vtkFloatArray* Value;
};

} // namespace bmia

#endif
