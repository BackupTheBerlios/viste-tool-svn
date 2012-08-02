/**
 * vtkShaderBase.cxx
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-05-17	Tim Peeters
 * - Removed most implementations because they were moved to
 *   vktShaderBaseHandle subclass.
 *
 * 2005-07-14	Tim Peeters
 * - Added SupportsOpenGLVersion() function.
 */

#include "vtkShaderBase.h"

namespace bmia {

vtkShaderBase::vtkShaderBase()
{
  // nothing
}

vtkShaderBase::~vtkShaderBase()
{
  // nothing was created by this class, so don't destroy anything either.
  // do that in the subclasses that were creating things.
}

// from http://developer.nvidia.com/object/nv_ogl2_support.html
bool vtkShaderBase::SupportsOpenGLVersion(int atLeastMajor, int atLeastMinor)
{
  const char* version;
  int major, minor;

  //glewInit();
  version = (const char *) glGetString(GL_VERSION);
  cout<<"OpenGL version is "<<version<<endl;
  //vtkDebugMacro(<<"OpenGL version is "<<version);

  if (sscanf(version, "%d.%d", &major, &minor) == 2) {
  if (major > atLeastMajor)
    return true;
  if (major == atLeastMajor && minor >= atLeastMinor)
    return true;
  } else {
    /* OpenGL version string malformed! */
  }
  return false;
}

} // namespaced bmia
