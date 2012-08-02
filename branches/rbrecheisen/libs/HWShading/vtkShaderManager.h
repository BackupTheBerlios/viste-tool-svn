/**
 * vtkShaderManager.h
 * by Tim Peeters
 *
 * 2005-05-04	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Added GetGlVersion()
 * - Renamed glUseProgram() to UseGlProgram().
 *
 * 2005-07-04	Tim Peeters
 * - DEPRECATED this class in favor of vtkShaderProgramReader
 *   and the vtkShaderProgram->Activate() function.
 */

#ifndef bmia_vtkShaderManager_h
#define bmia_vtkShaderManager_h

#include "vtkShaderBase.h"

namespace bmia {

class vtkMyShaderProgram;

/**
 * Class for setting the currently active shader.
 * DEPRECATED. It still works, but don't use it anyway. Shader programs can
 * be read using vtkMyShaderProgramReader. Shader programs can be activated
 * with vtkMyShaderProgram->Activate(). Calling of the Initialize() function of
 * the shader manager is no longer needed if the renderwindow is correctly
 * initialized.
 */
class vtkShaderManager: public vtkShaderBase
// shader manager doesn't have/need a handle, but I made it a subclass of
// vtkShaderBase anyway for convenience (include glew.h etc).
{
public:
  static vtkShaderManager *New();

  /**
   * Inititialize glew and check for the extensions required
   * for hardware shading.
   */
  void Initialize();

  /**
   * Returns true if the specified version of OpenGL is supported, and
   * false otherwise.
   */
  static bool SupportsOpenGLVersion(int atLeastMajor, int atLeastMinor);

  /**
   * Returns true if GLSL is supported and false otherwise.
   */
  bool GetHWShaderSupport();

  /**
   * Set/Get the shader to be used when rendering.
   * Value of shader may be NULL to disable custom shaders
   * and to enable the OpenGL fixed-pipeline functionality.
   * NOTE: This function does not (yet) start monitoring of changes
   * in the shader program or its associated shader objects. So if
   * you change any of those, execute UpdateActiveShader() again.
   */
  void UseShaderProgram(vtkMyShaderProgram* program)
    {
    this->SetActiveShaderProgram(program);
    }
  void SetActiveShaderProgram(vtkMyShaderProgram* program);
  vtkGetObjectMacro(ActiveShaderProgram, vtkMyShaderProgram);

  /**
   * Read a shader from file and activate that shader.
   */
  virtual void UseShaderFromFile(const char* filename);

  /**
   * Checks for changes in the active shader and recompiles and
   * relinks it if needed.
   */
  void UpdateActiveShaderProgram();

protected:
  vtkShaderManager();
  ~vtkShaderManager();

  bool UseGlProgram();

  bool Initialized;

private:
  vtkMyShaderProgram* ActiveShaderProgram;

  /**
   * Parse GL_VERSION and return the major and minor numbers in the supplied
   * integers.
   * If it fails for any reason, major and minor will be set to 0.
   * Assumes a valid OpenGL context.
   */
  void GetGlVersion(int *major, int *minor);
};

} // namespace bmia

#endif // bmia_vtkShaderManager_h
