/**
 * vtkUniformFloat.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkUniformFloat_h
#define bmia_vtkUniformFloat_h

#include "vtkShaderUniform.h"

namespace bmia {

/**
 * Class for representing uniform float variables.
 */
class vtkUniformFloat : public vtkShaderUniform
{
public:
  static vtkUniformFloat* New();

  vtkSetMacro(Value, float);
  vtkGetMacro(Value, float);

protected:
  vtkUniformFloat();
  ~vtkUniformFloat();

  virtual void SetGlUniformSpecific();

private:
  float Value;
};

} // namespace bmia

#endif
