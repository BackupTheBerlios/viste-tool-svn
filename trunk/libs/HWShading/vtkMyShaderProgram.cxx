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
 * vtkMyShaderProgram.cxx
 *
 * 2005-05-04	Tim Peeters
 * - First version
 *
 * 2005-05-17	Tim Peeters
 * - Simplified Link()
 * - Added support for uniform values
 * - Removed support for reading values from file. This is now implemented
 *   in vtkShaderProgramReader.
 *
 * 2005-06-03	Tim Peeters
 * - Use namespace bmia
 * - Include vtkstd/string
 *
 * 2005-06-06	Tim Peeters
 * - Switch to OpenGL 2.0
 * - Added DeleteGlProgram()
 *
 * 2005-07-01	Tim Peeters
 * - Added validation of program after linking in LinkGlProgram, and
 *   output the infolog if linking fails or debugging is enabled.
 * - Added Validate() function.
 *
 * 2006-01-30	Tim Peeters
 * - Use vtkOpenGLExtensionManager and vtkgl.h instead of GLEW.
 *
 * 2007-10-24	Tim Peeters
 * - Add functions for setting vertex attributes, instead of using the
 *   fixed ones as was done before (1 = "Tangent", 2 = "LineID").
 *
 * 2008-09-03	Tim Peeters
 * - Rename vtkShaderProgram to vtkMyShaderProgram to avoid naming conflicts
 *   with the vtkShaderProgram in (the new) VTK 5.2
 */

#include "vtkMyShaderProgram.h"

#include "vtkShaderObject.h"
#include "vtkShaderObjectCollection.h"
#include "vtkShaderUniform.h"
#include "vtkShaderUniformCollection.h"
#include <vtkObjectFactory.h>
#include <vtkstd/string>
#include <vtkgl.h>

namespace bmia {

vtkStandardNewMacro(vtkMyShaderProgram);

vtkMyShaderProgram::vtkMyShaderProgram()
{
  this->ShaderObjects = vtkShaderObjectCollection::New();
  this->ShaderUniforms = vtkShaderUniformCollection::New();
  this->Linked = false;
  for (int i=0; i < this->NumAttribs; i++) this->Attribs[i] = NULL;

}

vtkMyShaderProgram::~vtkMyShaderProgram()
{
  for (int i=0; i < this->NumAttribs; i++)
    {
    if (this->Attribs[i] != NULL) delete [] this->Attribs[i];
    this->Attribs[i] = NULL;
    }
  this->DetachAllGlShaders(this->ShaderObjects);
  this->ShaderObjects->Delete();
  this->ShaderUniforms->Delete();
  this->DeleteGlProgram();
}

void vtkMyShaderProgram::AddShaderObject(vtkShaderObject* object)
{
  vtkDebugMacro(<<"Adding shader object "<<object);
  if (object == NULL)
    {
    vtkDebugMacro(<<"Don't add NULL shader objects!");
    return;
    }
  this->ShaderObjects->AddItem(object);
  this->Modified();
}

void vtkMyShaderProgram::RemoveShaderObject(vtkShaderObject* object)
{
  this->ShaderObjects->RemoveItem(object);
  this->Modified();
}

void vtkMyShaderProgram::AddShaderUniform(vtkShaderUniform* uniform)
{
  if (uniform == NULL)
    {
    vtkDebugMacro(<<"Not adding the uniform with value NULL.");
    return;
    }
  this->ShaderUniforms->AddItem(uniform);
  this->Modified();
}

void vtkMyShaderProgram::RemoveShaderUniform(vtkShaderUniform* uniform)
{
  this->ShaderUniforms->RemoveItem(uniform);
  this->Modified();
}

unsigned long int vtkMyShaderProgram::GetMTime()
{
  unsigned long mTime = this-> vtkShaderBase::GetMTime();
  unsigned long time;

  this->ShaderObjects->InitTraversal();
  vtkShaderObject* object = this->ShaderObjects->GetNextItem();
  while (object != NULL)
    {
    time = object->GetMTime();
    mTime = ( time > mTime ? time : mTime );
    object = this->ShaderObjects->GetNextItem();
    }
  //object == NULL

  return mTime;
}

void vtkMyShaderProgram::Link()
{
  if (this->Linked && this->LinkTime > this->GetMTime())
    {
    vtkDebugMacro(<<"Link time > MTime and linked, so not re-linking.");
    }
  else
    {
    vtkDebugMacro(<<"(Re)linking.");
    this->Linked = false;

    if (!this->SupportsOpenGLVersion(2, 0))
      {
      vtkErrorMacro(<<"OpenGL 2.0 not supported, so shader program cannot "
		<<"be used. Terminating program.");
      exit(1);
      }

    if (!this->CreateGlProgram()) return;

    vtkDebugMacro(<<"Setting attribs...");

    for (int i=1; i < this->NumAttribs; i++)
      {
      //cout<<"this->Attribs = "<<this->Attribs<<endl;
      //cout<<"attribs["<<i<<"] = "<<((this->Attribs[i] != NULL)?this->Attribs[i]:"(null)")<<" ("<<this<<")"<<endl;
      if (this->Attribs[i] != NULL)
        {
	// XXX: In the past I started with handle 1. Maybe 0 is also possible. VERIFY THIS!
	//cout<<"Calling vtkgl::BindAttribLocation(this->GetHandle(), "<<i<<", "<<(this->Attribs[i]?this->Attribs[i]:"(null)")<<");"<<" ("<<this<<")"<<endl;
	vtkgl::BindAttribLocation(this->GetHandle(), i, this->Attribs[i]);
        } // if
      } // for

    if (!this->AttachAllGlShaders(this->ShaderObjects)) return;
    if (!this->LinkGlProgram()) 
      {
      return;
      }

    this->LinkTime.Modified();
    this->Linked = true;

    //this->glUniformAll(this->ShaderUniforms);
    // must be done after calling glUseProgramObject.
    // XXX: Now handled in vtkShaderManager.
  } // else

  return;
}

void vtkMyShaderProgram::ForceReLink()
{
  if (!this->LinkGlProgram()) return;
  this->LinkTime.Modified();
  this->Linked = true;
}

bool vtkMyShaderProgram::CreateGlProgram()
{
  if (this->GetHandleValid())
    {
    // no need to create a new handle.
    vtkDebugMacro(<<"Handle is already valid. Not creating a new one");
    }
  else
    {
    // create a new  handle.
    vtkDebugMacro(<<"No valid handle found. Creating new one...");
    GLuint handle = vtkgl::CreateProgram();
    vtkDebugMacro(<<"Handle created. Setting handle...");
    this->SetHandle(handle);
    }
  return true;
}

bool vtkMyShaderProgram::AttachGlShader(vtkShaderObject* object)
{
  if (!this->GetHandleValid())
    {
    vtkErrorMacro(<<"Do not try to attach shader objects to a shader program "
		<<"that does not have a handle yet!");
    return false;
    }

  if (object == NULL)
    {
    vtkWarningMacro(<<"Cannot attach a NULL shader object!");
    //return true; // why not continue with the other shader programs?
    return false;
    }

  // make sure the shader object is compiled
  object->Compile();

  vtkgl::AttachShader(this->GetHandle(), object->GetHandle());
  // TODO: check whether the attaching was successful?

  return true;  
}

bool vtkMyShaderProgram::DetachGlShader(vtkShaderObject* object)
{
  if (!this->GetHandleValid())
    {
    vtkErrorMacro(<<"How can you detach a shader object if the shader program "
		<<"does not even have a handle??");
    return false;
    }

  if (object == NULL)
    {
    vtkWarningMacro(<<"Cannot detach a NULL shader object!");
    return false; // or true, if glAttach(NULL) does this.
    }

  if (object->GetHandleValid())
    {
    vtkgl::DetachShader(this->GetHandle(), object->GetHandle());
    return true;
    }
  else
    { // !object->GetHandleValid()
    vtkErrorMacro(<<"Trying to detach a shader object that does not"
		<<" have a handle!");
    return false;
    }
}

bool vtkMyShaderProgram::AttachAllGlShaders(vtkShaderObjectCollection* objects)
{
  if (!this->GetHandleValid())
    {
    vtkErrorMacro(<<"Cannot attach shader objects to a shader program that"
		<<" does not have a handle yet!");
    return false;
    }

  if (objects == NULL)
    {
    vtkErrorMacro(<<"Don't call glAttachAll(NULL)!");
    return false;
    }

  if (objects->GetNumberOfItems() < 1)
    {
    vtkErrorMacro(<<"No shader objects specified to link to shader program!");
    return false;
    }

  bool result = true;
  this->ShaderObjects->InitTraversal();
  vtkShaderObject* object = this->ShaderObjects->GetNextItem();
  while (object != NULL)
    {
    result = (result && this->AttachGlShader(object));
    object = this->ShaderObjects->GetNextItem();
    }
  //object == NULL

  return result;  
}

bool vtkMyShaderProgram::DetachAllGlShaders(vtkShaderObjectCollection* objects)
{
  if (!this->GetHandleValid())
    {
    vtkDebugMacro(<<"Cannot detach shader objects if the shader program"
		<<" does not have a valid handle.");
    return false;
    }

  if (objects == NULL)
    {
    vtkErrorMacro(<<"Don't call glDetachAll(NULL)!");
    return false;
    }

  bool result = true;
  this->ShaderObjects->InitTraversal();
  vtkShaderObject* object = this->ShaderObjects->GetNextItem();
  while (object != NULL)
    {
    result = (this->DetachGlShader(object) && result);
    object = this->ShaderObjects->GetNextItem();
    }
  // object == NULL

  return result;
}

bool vtkMyShaderProgram::LinkGlProgram()
{
  GLint success;
  GLuint handle = this->GetHandle();
  vtkgl::LinkProgram(handle);
  //glGetShaderiv(handle, GL_LINK_STATUS, &success);
  vtkgl::GetProgramiv(handle, vtkgl::LINK_STATUS, &success);

  if (this->GetDebug() || (success != GL_TRUE))
    {
    if (success != GL_TRUE)
      {
      vtkWarningMacro(<<"Linking of shader program failed!");
      }

    GLint InfoLogLength;
    vtkgl::GetProgramiv(handle, vtkgl::INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength == 0)
      {
      vtkWarningMacro(<<"OpenGL info log has length 0!");
      return false;
      }
    else // InfoLogLength != 0
      {
		  vtkgl::GLchar* InfoLog = (vtkgl::GLchar *)malloc(InfoLogLength);
      if (InfoLog == NULL)
        {
        vtkWarningMacro(<<"Could not allocate InfoLog buffer!");
        return false;
        }
      GLint CharsWritten = 0;
      vtkgl::GetProgramInfoLog(handle, InfoLogLength, &CharsWritten, InfoLog);
      vtkWarningMacro("Shader InfoLog for shader with handle "<<handle<<":\n"<<InfoLog);
      free(InfoLog);
      }
    //this->Validate();
    if (success != GL_TRUE) return false;
    }
  vtkDebugMacro(<<"Linking of shader program succeeded.");

  if (this->GetDebug()) this->Validate();

  return true;
}

bool vtkMyShaderProgram::SetAllGlUniforms(vtkShaderUniformCollection* uniforms)
{
  if (!this->GetHandleValid())
    {
    vtkErrorMacro(<<"Cannot set uniform values for a shader program that"
		<<" does not have a handle yet!");
    return false;
    }

  if (uniforms == NULL)
    {
    vtkErrorMacro(<<"Don't call glUniformAll(NULL)!");
    return false;
    }

  if (uniforms->GetNumberOfItems() < 1)
    {
    vtkDebugMacro(<<"No uniform values specified.");
    return true;
    }

  bool result = true;
  this->ShaderUniforms->InitTraversal();
  vtkShaderUniform* uniform = this->ShaderUniforms->GetNextItem();
  while (uniform != NULL)
    {
    vtkDebugMacro("Handling uniform "<<uniform<<"...");
    //uniform->DebugOn();
    result = (this->SetGlUniform(uniform) && result);
    uniform = this->ShaderUniforms->GetNextItem();
    }
  /* uniform == NULL */

  return result;  
}

bool vtkMyShaderProgram::SetGlUniform(vtkShaderUniform* uniform)
{
  if (!uniform)
    {
    vtkErrorMacro(<<"Don't call glUniform with parameter NULL!");
    return false;
    }
  //uniform->DebugOn();
  uniform->SetHandle(this->GetHandle());
  return uniform->SetGlUniform();
}

bool vtkMyShaderProgram::DeleteGlProgram()
{
  if (!this->GetHandleValid())
    {
    vtkDebugMacro(<<"Calling DeleteGlProgram() without a valid handle!");
    // nothing to delete.
    return false;
    }
  vtkgl::DeleteProgram(this->GetHandle());
  return true;
}

bool vtkMyShaderProgram::Validate()
{
  if (!this->GetHandleValid())
    {
    vtkErrorMacro(<<"Cannot validate shader program without a handle!");
    return false;
    }

  GLint handle = this->GetHandle();
  GLint success;
  vtkgl::ValidateProgram(handle);
  vtkgl::GetProgramiv(handle, vtkgl::VALIDATE_STATUS, &success);

  if (success)
    {
    vtkWarningMacro(<<"Shader program successfully validated.");
    return true;
    }
  else // !success
    {
    GLint InfoLogLength;
    vtkgl::GetProgramiv(handle, vtkgl::INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength == 0)
      {
      vtkWarningMacro(<<"OpenGL info log has length 0!");
      }
    else // InfoLogLength != 0
      {
		  vtkgl::GLchar* InfoLog = (vtkgl::GLchar *)malloc(InfoLogLength);
      if (InfoLog == NULL)
        {
        vtkWarningMacro(<<"Could not allocate InfoLog buffer!");
        return false;
        }
      GLint CharsWritten = 0;
      vtkgl::GetProgramInfoLog(handle, InfoLogLength, &CharsWritten, InfoLog);
      vtkWarningMacro("Validation of shader program failed. InfoLog:\n"<<InfoLog);
      free(InfoLog);
      }
    }

  return false;
}

void vtkMyShaderProgram::ApplyShaderUniforms()
{
  this->SetAllGlUniforms(this->ShaderUniforms);
}

void vtkMyShaderProgram::Activate()
{
  vtkDebugMacro(<<"Activating shader program...");
  this->Link();
  vtkgl::UseProgram(this->GetHandle());
  this->ApplyShaderUniforms();
}

void vtkMyShaderProgram::Deactivate()
{
  vtkgl::UseProgram(0);
}

void vtkMyShaderProgram::SetAttrib(int i, const char* name)
{
  vtkDebugMacro(<<"Setting Attrib["<<i<<"] to: "<<(name?name:"(null)"));
  assert(0 <= i);
  assert(i < this->NumAttribs);
  if ( this->Attribs[i] == NULL && name == NULL) { return;}
  if ( this->Attribs[i] && name && (!strcmp(this->Attribs[i],name))) { return;}
  if ( this->Attribs[i]) { delete [] this->Attribs[i]; this->Attribs[i] = NULL; }
  if (name)
    {
/*
    size_t n = strlen(name) + 1;
    char *cp1 =  new char[n];
    const char *cp2 = (name);
    this->Attribs[i] = cp1;
    do { *cp1++ = *cp2++; } while ( --n );
*/

  size_t n = strlen(name) + 1;
//  char* str = new char[n];
  this->Attribs[i] = new char[n];
  strcpy(this->Attribs[i], name);
//  this->Attribs[i] = str; 

//    this->Attribs[i] = "Tangent";
    } // if
   else
    {
    this->Attribs[i] = NULL;
    } // else
  this->Modified();
}

const char* vtkMyShaderProgram::GetAttrib(int i)
{
  assert(0 <= i);
  assert(i < this->NumAttribs);
  return this->Attribs[i];
}

} // namespace bmia
