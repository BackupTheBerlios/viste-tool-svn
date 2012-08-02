/**
 * vtkShaderBase.h
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-05-17	Tim Peeters
 * - Removed functionality dealing with GLhandles. This was moved to  a new
 *   vtkShaderBaseHandle subclass.
 *
 * 2005-06-06	Tim Peeters
 * - Switched to OpenGL 2.0
 *
 * 2005-07-04	Tim Peeters
 * - Instead of using glew, set GL_GLEXT_PROTOTYPES and then include gl.h
 *
 * 2005-07-14	Tim Peeters
 * - Use glew again. On Windows it seems to be needed.
 * - Add SupportsOpenGLVersion() function.
 *
 * 2006-01-30	Tim Peeters
 * - Removed #include <GL/glew.h>
 */

#ifndef bmia_vtkShaderBase_h
#define bmia_vtkShaderBase_h

//#include <GL/glew.h> // for OpenGL types and some functions
		     // TODO: can this be done without glew?
#include <vtkgl.h>

#include <vtkObject.h>

//#define GL_GLEXT_PROTOTYPES 1
//#include "GL/gl.h"

namespace bmia {

/**
 * Base class for all GLSL shader related subclasses. Implements
 * printing of info logs, etc.
 * NOTE: include this header file before including any rendering header
 * files because glew.h must be included before gl.h.
 */
class vtkShaderBase : public vtkObject
{
public:

  /**
   * Returns true if the specified version of OpenGL is supported, and
   * false otherwise.
   */
  static bool SupportsOpenGLVersion(int atLeastMajor, int atLeastMinor);

protected:
  vtkShaderBase();
  ~vtkShaderBase();

private:

};

} // namespace bmia

#endif // bmia_vtkShaderBase_h
