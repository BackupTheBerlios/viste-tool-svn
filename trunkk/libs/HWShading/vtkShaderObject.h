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
 * vtkShaderObject.h
 * by Tim Peeters
 *
 * 2005-05-02	Tim Peeters
 * - First version
 *
 * 2005-06-03	Tim Peeters
 * - Use bmia namespace and removed old comments
 *
 * 2005-06-06	Tim Peeters
 * - Renamed glCreateObject() to CreateGlShader()
 * - Renamed glSource() to SetGlShaderSource()
 * - Renamed glCompile() to CompileGlShader()
 *
 * 2005-09-12	Tim Peeters
 * - Added static void printShaderInfoLog()
 */

#ifndef bmia_vtkShaderObject_h
#define bmia_vtkShaderObject_h

#include "vtkShaderBaseHandle.h"

namespace bmia {

/**
 * Class for representing a GLSL shader object.
 */
class vtkShaderObject : public vtkShaderBaseHandle
{
public:
  /**
   * Specify the source for this object. Either set the source text directly or
   * specify the text file which contains the source text.
   */
	virtual void SetSourceText (const char * _arg)
	{
		vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting SourceText to " << (_arg?_arg:"(null)") );

		if (this->SourceText == NULL && _arg == NULL) 
		{ 
			return;
		}

		if (this->SourceText && _arg && (!strcmp(this->SourceText,_arg))) 
		{ 
			return;
		}
	if (this->SourceText) { delete [] this->SourceText; }
	if (_arg)
	  {
	  this->SourceText = new char[strlen(_arg)+1];
	  strcpy(this->SourceText,_arg);
	  }
	else
	  {
	  this->SourceText = NULL;
	  }
	this->Modified();
	} 

  vtkGetStringMacro(SourceText);
  void ReadSourceTextFromFile(const char* filename);

  virtual void Compile();
  bool IsCompiled()
    {
    return this->Compiled;
    }

protected:
  vtkShaderObject();
  ~vtkShaderObject();

  virtual bool CreateGlShader() = 0;
  virtual bool DeleteGlShader();
  virtual bool SetGlShaderSource();
  virtual bool CompileGlShader();

private:
  /**
   * Shader source text. All lines must be \0-terminated.
   */
  char* SourceText;

  char* FileName;

  /**
   * True if Compile() was succesful.
   */
  bool Compiled;

  /**
   * Time at which the shader object was last compiled.
   */
  vtkTimeStamp CompileTime;

  /**
   * Print shader infolog (duh).
   */
  static void PrintShaderInfoLog(GLuint shader);
  
};

} // namespace bmia

#endif // bmia_vtkShaderObject_h
