/**
 * vtkShadowMappingHelperLines.h
 * by Tim Peeters
 *
 * 2005-11-22	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkShadowMappingHelperLines_h
#define bmia_vtkShadowMappingHelperLines_h

#include "vtkShadowMappingHelper.h"

namespace bmia {

/**
 * Shadow mapping helper with a slightly different shader for building
 * the shadow map that is used by vtkFiberMapper.
 */
class vtkShadowMappingHelperLines : public vtkShadowMappingHelper
{
public:
  static vtkShadowMappingHelperLines* New();

protected:
  vtkShadowMappingHelperLines();
  ~vtkShadowMappingHelperLines();
private:

}; // class vtkShadowMappingHelperLines
} // namespace bmia

#endif // bmia_vtkShadowMappingHelperLines_h
