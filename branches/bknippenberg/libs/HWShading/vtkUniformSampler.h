/**
 * vtkUniformSampler.h
 *
 * 2005-05-18	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkUniformSampler_h
#define bmia_vtkUniformSampler_h

#include "vtkUniformInt.h"

namespace bmia {

/**
 * Class for representing uniform sampler variables.
 * Samplers are used to pass the name/index of a texture to use to a shader.
 *
 * Same implementation as vtkUniformInt. Here, there is no distinction between
 * sampler1D, sampler2D, sampler3D, samplerCube, sampler1DShadow or
 * sampler2DShadow. Note that the corresponding texture must be of the
 * type matching the sampler type.
 *
 * This class may be modified in the future to also handle the loading of the
 * texture and/or check for restrictions imposed on samplers.
 */
class vtkUniformSampler : public vtkUniformInt
{
public:
  static vtkUniformSampler* New();

protected:
  vtkUniformSampler();
  ~vtkUniformSampler();

private:

};

} // namespace bmia

#endif // bmia_vtkUniformSampler_h
