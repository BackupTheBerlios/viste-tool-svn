/**
 * vtkShaderBaseHandle.h
 * by Tim Peeters
 *
 * 2005-05-17	Tim Peeters
 * - First version, based on the (old) vtkShaderBase.h
 *
 * 2006-06-06	Tim Peeters
 * - Switched to OpenGL 2.0 (removed ARB in function calls and types)
 * - Removed glDelete() function. Now implemented in vtkShaderObject
 *   and vtkShaderProgram as DeleteGlShader() and DeleteGlProgram().
 */

#ifndef bmia_vtkShaderBaseHandle_h
#define bmia_vtkShaderBaseHandle_h

#include "vtkShaderBase.h"

namespace bmia {

/**
 * Base class for GLSL shader related subclasses with GLhandles.
 */
class vtkShaderBaseHandle : public vtkShaderBase
{
public:
  /**
   * Gets/Sets the handle for the current shader object or program.
   */
  GLuint GetHandle();
  void SetHandle(GLuint newhandle);
  void UnsetHandle();

  /**
   * True if a handle was given a value using SetHandle() and UnsetHandle()
   * was not called afterwards.
   */
  bool GetHandleValid();

protected:
  vtkShaderBaseHandle();
  ~vtkShaderBaseHandle();

  /**
   * Deletes the shader program or object associated with this->Handle
   * if this->Handle is valid (GetHandleValid()).
   */
  //void glDelete();

private:

  /**
   * Handle for shader or vertex object. Initialize in subclasses!
   */
  GLuint Handle;

  /**
   * True if this->Handle was given a value using SetHandle and UnsetHandle
   * was not called afterwards.
   */
  bool HandleValid;

};

} // namespace bmia

#endif // __vtkShaderBaseHandle_h
