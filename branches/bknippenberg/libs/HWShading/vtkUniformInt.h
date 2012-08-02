/**
 * vtkUniformInt.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkUniformInt_h
#define bmia_vtkUniformInt_h

#include "vtkShaderUniform.h"

namespace bmia {

/**
 * Class for representing uniform int variables.
 */
class vtkUniformInt : public vtkShaderUniform
{
public:
  static vtkUniformInt* New();

  vtkSetMacro(Value, int);
  vtkGetMacro(Value, int);

protected:
  vtkUniformInt();
  ~vtkUniformInt();

  virtual void SetGlUniformSpecific();

private:
  int Value;
};

} // namespace bmia

#endif
