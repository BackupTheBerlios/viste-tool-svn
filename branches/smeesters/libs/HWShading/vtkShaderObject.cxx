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
 * vtkShaderObject.cxx
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-06-03	Tim Peeters
 * - Use bmia namespace
 * - Replaced including <iostream> and <fstream> by
 *   <vtkIOStream.h>
 *
 * 2005-06-06	Tim Peeters
 * - Switched to OpenGL 2.0 instead of using ARB extension for OpenGL 1.5
 * - Renamed function names (see changes in header file).
 * - Added DeleteGlShader()
 *
 * 2005-09-12	Tim Peeters
 * - Added printShaderInfoLog() from ogl2brick.
 *
 * 2006-01-30	Tim Peeters
 * - Make use of vtkgl header instead of glew.
 *   So, use vtgl::Function() instead of glFunction() for extensions.
 */

//#define GL_GLEXT_PROTOTYPES 1
//#include <GL/gl.h>
//#include "glext-tim.h"

#include "vtkShaderObject.h"

#include <vtkIOStream.h>

namespace bmia {

vtkShaderObject::vtkShaderObject()
{
  //this->SourceFileName = NULL;
  //this->SourceText = ""; //NULL;
  this->SourceText = NULL;
  this->FileName = NULL;
  this->Compiled = false;

  //this.CompileTime = vtkTimeStamp::New();
}

vtkShaderObject::~vtkShaderObject()
{
  if (this->Compiled)
    {
    this->Compiled = false;
    }

  if (this->SourceText != NULL)
    {
    delete [] this->SourceText;
    this->SourceText = NULL;
    }

  // check for HandleValid is in DeleteGlShader().
  this->DeleteGlShader();
}

void vtkShaderObject::Compile()
{
  if ( (this->Compiled) && (this->CompileTime > this->GetMTime()) )
    {
    vtkDebugMacro(<<"Compiled && CompileTime > MTime. Not recompiling.");
    // no need to recompile.
    return;
    }

  //vtkDebugMacro(<<"CompileTime not greater than MTime. Recompiling.");
  this->Compiled = false;

  bool success;
  success = this->CreateGlShader();
  if (!success)
    {
    vtkWarningMacro(<<"Could not create gl shader object!");
    return;
    }

  success = this->SetGlShaderSource();
  if (!success)
    {
    vtkWarningMacro(<<"Could not set shader source!");
    return;
    }

  if(this->FileName)
    std::cout << "Compiling shader " << this->FileName << std::endl;

  success = this->CompileGlShader();
  if (!success)
    {
//    vtkWarningMacro(<<"Could not compile shader object with source text:\n"
//		<<this->SourceText);
    vtkShaderObject::PrintShaderInfoLog(this->GetHandle());
    return;
    }

  this->CompileTime.Modified();
  this->Compiled = true;
  return;
}

bool vtkShaderObject::SetGlShaderSource()
{
  if (this->SourceText == NULL)
    {
    vtkWarningMacro(<<"No source text was specified!");
    return false;
    // OR: just compile and have an empty shader object?
    }

  const char* text = this->SourceText;
  //glShaderSource(this->GetHandle(), 1, &text, NULL);
  vtkgl::ShaderSource(this->GetHandle(), 1, &text, NULL);

  // XXX: I think/assume text only copies the pointer to SourceText,
  // so it does not need to be deleted here.
  text = NULL;

  return true;
}

bool vtkShaderObject::CompileGlShader()
{
  // this->GetHandleValid() is checked in this->GetHandle().
  //if (!this->GetHandleValid())
  //  {
  //  vtkWarningMacro(<<"No valid handle. Cancelling compilation.");
  //  return false;
  //  }

  GLint success;

  vtkgl::CompileShader(this->GetHandle());

  vtkgl::GetShaderiv(this->GetHandle(), vtkgl::COMPILE_STATUS, &success);
  if (success != GL_TRUE)
    {
    vtkWarningMacro(<<"Compilation of shader failed!");
    // TODO: find out why compilation failed and output that.
    return false;
    }

  return true;
}

void vtkShaderObject::ReadSourceTextFromFile(const char* filename)
{
  if ( !filename || (strlen(filename) == 0))
    {
    vtkErrorMacro(<< "No file specified!");
    return;
    }

  if(this->FileName)
      delete [] this->FileName;
  this->FileName = new char[strlen(filename) + 1];
  strcpy(this->FileName, filename);

  vtkDebugMacro("Reading source text from file "<<filename);
  // http://www.cplusplus.com/ref/iostream/istream/read.html

  int length;
  char * buffer;

  ifstream is;
  is.open (filename, ios::binary );

  // get length of file:
  is.seekg (0, ios::end);
  length = is.tellg();
  is.seekg (0, ios::beg);

  // allocate memory:
  buffer = new char [length+1];

  // read data as a block:
  is.read (buffer,length);

  is.close();
  //vtkDebugMacro(<<"Last 3 character read are:"<<buffer[length-3]<<buffer[length-2]<<buffer[length-1]);

  buffer[length] = '\0'; // seems to help :)
  this->SetSourceText(buffer);

  delete [] buffer;
}

bool vtkShaderObject::DeleteGlShader()
{
  if (!this->GetHandleValid())
    {
    vtkDebugMacro(<<"Calling DeleteGlProgram() without a valid handle!");
    // nothing to delete.
    return false;
    }
  vtkgl::DeleteShader(this->GetHandle());
  return true;
}

//
// Print out the information log for a shader object
//
void vtkShaderObject::PrintShaderInfoLog(GLuint shader)
{
    GLint infologLength = 0;
    GLsizei charsWritten  = 0;
    vtkgl::GLchar *infoLog;

    //printOpenGLError();  // Check for OpenGL errors

    vtkgl::GetShaderiv(shader, vtkgl::INFO_LOG_LENGTH, &infologLength);

    //printOpenGLError();  // Check for OpenGL errors

    if (infologLength > 0)
    {
	infoLog = (vtkgl::GLchar *)malloc(infologLength);
        if (infoLog == NULL)
        {
            printf("ERROR: Could not allocate InfoLog buffer\n");
            exit(1);
        }
        vtkgl::GetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
        printf("Shader InfoLog:\n%s\n\n", infoLog);
        free(infoLog);
    }
    //printOpenGLError();  // Check for OpenGL errors
}

} // namespace bmia
