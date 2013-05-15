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
 * vtkMyShaderProgramReader.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2008-09-03	Tim Peeters
 * - Rename vtkShaderProgramReader to vtkMyShaderProgramReader
 */

#ifndef bmia_vtkMyShaderProgramReader_h
#define bmia_vtkMyShaderProgramReader_h

#include "Helpers/vtkBetterDataReader.h"

namespace bmia {

class vtkMyShaderProgram;

/**
 * Class shader programs.
 */
class vtkMyShaderProgramReader: public vtkBetterDataReader
{
public:
  static vtkMyShaderProgramReader *New();

  /**
   * Set/Get the output of this reader.
   */
  void SetOutput(vtkMyShaderProgram* output);
  vtkGetObjectMacro(Output, vtkMyShaderProgram);

  /**
   * Read the shader from file. Always call this function before using the
   * output. This is not done automatically as in VTK readers that output
   * vtkDataObjects!
   */
  void Execute();

protected:
  vtkMyShaderProgramReader();
  ~vtkMyShaderProgramReader();

  void ReadUniformFromLine(string line, unsigned int linepos);

private:
  vtkMyShaderProgram* Output;

};

} // namespace bmia

#endif // bmia_vtkMyShaderProgramReader_h
