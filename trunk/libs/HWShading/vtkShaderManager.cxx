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
 * vtkShaderManager.cxx
 * by Tim Peeters
 *
 * 2005-05-04	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Switched to OpenGL 2.0
 * - Renamed glUseProgram() to UseGlProgram()
 */

#include "vtkShaderManager.h"

#include "vtkMyShaderProgram.h"
#include "vtkMyShaderProgramReader.h"

#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkShaderManager);

vtkShaderManager::vtkShaderManager()
{
  this->ActiveShaderProgram = NULL;
  this->Initialized = false;
}

vtkShaderManager::~vtkShaderManager()
{
  this->SetActiveShaderProgram(NULL);
}

void vtkShaderManager::Initialize()
{
  // TODO: check for working OpenGL environment to use.
  // otherwise segfaults will pop up when using gl*() later.
  cout<<"vtkShaderManager::Initialize(): calling glewInit()."<<endl;
  glewInit();
  if (this->GetHWShaderSupport())
    {
    vtkDebugMacro(<<"Ready for GLSL.");
    this->Initialized = true;
    }
  else
    {
    vtkWarningMacro(<<"GLSL not supported!");
    this->Initialized = false;
    }
}

bool vtkShaderManager::GetHWShaderSupport()
{
  // note: glewInit() or this->Initialize() must have been called first.
  //  return (	(glewGetExtension("GL_ARB_fragment_shader") == GL_TRUE) &&
  //		(glewGetExtension("GL_ARB_vertex_shader") == GL_TRUE)
  //	 );
  //  return true; // TODO: REALLY check :)
        //glewGetExtension("GL_ARB_shader_objects")       != GL_TRUE ||
        //glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)

  // Make sure that OpenGL 2.0 is supported by the driver
  int gl_major, gl_minor;
  this->GetGlVersion(&gl_major, &gl_minor);
  vtkDebugMacro(<<"GL_VERSION major="<<gl_major<<" minor="<< gl_minor);

  if (gl_major < 2)
  {
    vtkErrorMacro(<<"GL_VERSION major="<<gl_major<<" minor="<< gl_minor<<" "
	<<"Support for OpenGL 2.0 is required!");
    return false;
  }
  return true;
}

void vtkShaderManager::SetActiveShaderProgram(vtkMyShaderProgram* program)
{
  vtkDebugMacro(<<"Setting active shader program to "<<program);
  if (this->ActiveShaderProgram != program)
    {
    if (this->ActiveShaderProgram != NULL)
      {
      this->ActiveShaderProgram->UnRegister(this);
      }
    this->ActiveShaderProgram = program;
    if (program != NULL)
      {
      program->Register(this);
      //program->DebugOn();
      }
    this->UpdateActiveShaderProgram();
    this->Modified();
    }
}

void vtkShaderManager::UpdateActiveShaderProgram()
{
  vtkDebugMacro(<<"Updating active shader program");
  if (this->ActiveShaderProgram != NULL)
    {
    // Link() handles everything for you :)
    this->ActiveShaderProgram->Link();
    }
  this->UseGlProgram();
}

bool vtkShaderManager::UseGlProgram()
{
  if (!this->Initialized)
    {
    vtkWarningMacro(<<"Shader manager was not initialized!");
    return false;
    }

  if (this->ActiveShaderProgram == NULL)
    {
    // disable the programmable processors, and use fixed functionality
    // for both vertex and fragment processing.
    vtkDebugMacro(<<"Calling glUseProgram(0)");
    glUseProgram(0);
    }
  else
    {
    vtkDebugMacro(<<"Calling glUseProgram("<<this->ActiveShaderProgram->GetHandle()<<");");
    glUseProgram(this->ActiveShaderProgram->GetHandle());
    this->ActiveShaderProgram->ApplyShaderUniforms();
    }
  return true;
}

void vtkShaderManager::UseShaderFromFile(const char* filename)
{
  if (!this->Initialized)
    {
    vtkWarningMacro(<<"Shader manager was not initialized! Cancelling loading of shader.");
    return;
    }
  vtkDebugMacro(<<"Creating shader program...");
  vtkMyShaderProgramReader* reader = vtkMyShaderProgramReader::New();
  reader->SetFileName(filename);
  reader->Execute();
  this->SetActiveShaderProgram(reader->GetOutput());
  reader->Delete(); reader = NULL;
}

void vtkShaderManager::GetGlVersion( int *major, int *minor )
{
  const char* verstr = (const char*)glGetString( GL_VERSION );
  if( (verstr == NULL) || (sscanf( verstr, "%d.%d", major, minor ) != 2) )
  {
      *major = *minor = 0;
      vtkErrorMacro(<<"Invalid GL_VERSION format!!!");
  }
}

// from http://developer.nvidia.com/object/nv_ogl2_support.html
bool vtkShaderManager::SupportsOpenGLVersion(int atLeastMajor, int atLeastMinor)
{
  const char* version;
  int major, minor;

  version = (const char *) glGetString(GL_VERSION);
  cout<<"OpenGL version is "<<version<<endl;

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

} // namespace bmia
