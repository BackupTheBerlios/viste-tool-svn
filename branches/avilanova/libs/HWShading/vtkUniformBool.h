/**
 * vtkUniformBool.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Renamed glSpecificUniform() to SetGlUniformSpecific().
 */

#ifndef bmia_vtkUniformBool_h
#define bmia_vtkUniformBool_h

#include "vtkShaderUniform.h"

namespace bmia {

/**
 * Class for representing uniform bool variables.
 */
class vtkUniformBool: public vtkShaderUniform
{
public:
  static vtkUniformBool* New();

  vtkSetMacro(Value, bool);
  vtkGetMacro(Value, bool);
  vtkBooleanMacro(Value, bool);

protected:
  vtkUniformBool();
  ~vtkUniformBool();

  virtual void SetGlUniformSpecific();

private:
  bool Value;
};

} // namespace bmia

#endif
