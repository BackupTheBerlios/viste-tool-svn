/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * vtkMyShaderProgram.h
 * by Tim Peeters
 *
 * 2005-05-04	Tim Peeters
 * - First version
 *
 * 2005-05-17	Tim Peeters
 * - Added various functions for dealing with uniform variables.
 * - Removed functions for reading the shader from file. This is now
 *   implemented in vtkShaderProgramReader.
 *
 * 2005-06-03	Tim Peeters
 * - Use bmia namespace
 * - Removed #include <string>
 *
 * 2005-06-06	Tim Peeters
 * - Renamed glCreateProgram() to CreateGlProgram
 * - Renamed glAttach() to AttachGlShader()
 * - Renamed glDetach() to DetachGlShader()
 * - Renamed glAttachAll() to AttachAllGlShaders()
 * - Renamed glDetachAll() to DetachAllGlShaders()
 * - Renamed glLink() to LinkGlProgram()
 * - Renamed glUniform() to SetGlUniform()
 * - Renamed UniformAll() to SetAllGlUniforms()
 *
 * 2005-07-01	Tim Peeters
 * - Added Validate()
 *
 * 2005-07-04	Tim Peeters
 * - Added Activate() and Deactivate() functions. vtkShaderManager class
 *   is now no longer needed for activating a shader program.
 *
 * 2007-10-24	Tim Peeters
 * - Add functions and variables for setting vertex attribute indices.
 *
 * 2008-09-04	Tim Peeters
 * - Rename bmia::vtkShaderProgram to bmia::vtkMyShaderProgram to avoid
 *   name conflicts with the vtkShaderProgram in the new VTK 5.2
 */

#ifndef bmia_vtkMyShaderProgram_h
#define bmia_vtkMyShaderProgram_h

#include "vtkShaderBaseHandle.h"
#include <assert.h>

namespace bmia {

class vtkShaderObject;
class vtkShaderObjectCollection;
class vtkShaderUniform;
class vtkShaderUniformCollection;

/**
 * Class for representing a GLSL hardware shader program.
 * This class also takes care of linking shader objects 
 * to the shader program.
 * Subclass of vtkCollection because a shader program is a set of shader
 * objects which are linked.
 */
class vtkMyShaderProgram : public vtkShaderBaseHandle
{
public:
  static vtkMyShaderProgram *New();

  virtual void AddShaderObject(vtkShaderObject* object);
  virtual void RemoveShaderObject(vtkShaderObject* object);
  vtkShaderObjectCollection* GetShaderObjects()
    {
    return this->ShaderObjects;
    };

  virtual void AddShaderUniform(vtkShaderUniform* uniform);
  virtual void RemoveShaderUniform(vtkShaderUniform* uniform);
  vtkShaderUniformCollection* GetShaderUniforms()
    {
    return this->ShaderUniforms;
    }

  /**
   * Link the shader program. This automatically compiles associated shader
   * objects if needed. This is called by vtkShaderManager if that is used
   * for managing the shaders.
   */
  virtual void Link();
  virtual void ForceLink()
    {
    this->Linked = false;
    this->Link();
    }

  /**
   * Shortcut to do some stuff faster. Used in vtkFiberMapper.
   * TODO: Do this in a nicer way in a newer version.
   */
  virtual void ForceReLink();

  bool IsLinked()
    {
    return this->Linked;
    }
  bool GetLinked()
    {
    return this->IsLinked();
    }

  /**
   * Get the shader program's mtime plus consider its shader objects.
   */
  unsigned long int GetMTime();

  /**
   * Must be called after calling glUseShaderProgram. Applies all
   * uniform variables attached to this shader program to the actual
   * program running on the GPU.
   */
  void ApplyShaderUniforms();

  /**
   * Validate this shader program. If you want to validate your program,
   * call this function after linking and not between glBegin() and glEnd().
   * If there are problems, false is returned and a description is
   * output as text to the console. If validation is successfull, true is
   * returned.
   */
  bool Validate();

  /**
   * Deactivate all other shader programs and activate this one as the
   * active shader program.
   */
  void Activate();

  /**
   * Deactivate the use of any shader program.
   */
  static void Deactivate();

  /**
   * Set indices of vertex attributes.
   * NOTE! Add 1 to i when passing the vertex to the shader!
   */
  void SetAttrib(int i, const char* attrib);
  const char* GetAttrib(int i);
  int GetMaxNumberOfAttribs()
    {
    return this->NumAttribs;
    }

protected:
  vtkMyShaderProgram();
  ~vtkMyShaderProgram();

  bool CreateGlProgram();
  bool DeleteGlProgram();
  bool AttachGlShader(vtkShaderObject* object);
  bool DetachGlShader(vtkShaderObject* object);
  bool AttachAllGlShaders(vtkShaderObjectCollection* objects);
  bool DetachAllGlShaders(vtkShaderObjectCollection* objects);
  bool LinkGlProgram();
  bool SetGlUniform(vtkShaderUniform* uniform);
  bool SetAllGlUniforms(vtkShaderUniformCollection* uniforms);

  char* Attribs[10];
  static const int NumAttribs = 10;
 // char* Attribs[10];

private:
  vtkShaderObjectCollection* ShaderObjects;
  vtkShaderUniformCollection* ShaderUniforms;

  bool Linked;
  vtkTimeStamp LinkTime;

};

} // namespace bmia

#endif // bmia_vtkMyShaderProgram_h
