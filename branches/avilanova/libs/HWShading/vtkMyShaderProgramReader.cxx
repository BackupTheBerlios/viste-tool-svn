/**
 * vtkMyShaderProgramReader.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version.
 *
 * 2008-09-04	Tim Peeters
 * - Rename vtkShaderProgramReader to vtkMyShaderProgramReader
 */

#include "vtkMyShaderProgramReader.h"
#include "vtkMyShaderProgram.h"
#include "vtkVertexShader.h"
#include "vtkFragmentShader.h"
#include "vtkUniformFloat.h"
#include "vtkUniformVec2.h"
#include "vtkUniformVec3.h"
#include "vtkUniformVec4.h"
#include "vtkUniformInt.h"

#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkMyShaderProgramReader);
	
vtkCxxSetObjectMacro(vtkMyShaderProgramReader, Output, vtkMyShaderProgram);

vtkMyShaderProgramReader::vtkMyShaderProgramReader()
{
  this->Output = NULL;
  vtkMyShaderProgram* program = vtkMyShaderProgram::New();
  this->SetOutput(program);
  program->Delete(); program = NULL;
}

vtkMyShaderProgramReader::~vtkMyShaderProgramReader()
{
  this->SetOutput(NULL);
}

// do the actual reading of the data.
void vtkMyShaderProgramReader::Execute()
{
  vtkDebugMacro("Reading shader specification from file "<<this->FileName);

  if (!this->Output)
    {
    vtkErrorMacro(<<"No output shader program!");
    return;
    }    

  if (!this->OpenFile())
    {
    vtkErrorMacro(<<"Could not open shader specification file!");
    return;
    }

  this->ResetCurrentLineNumber();

  unsigned int linepos;
  string word;
  while (this->NextLine())
    { // another line was read to process.
    // let's start at the beginning of the line:
    vtkDebugMacro("Reading line "<<this->GetCurrentLineNumber());
    linepos = 0;

    // TODO: for reading vertex and fragment shaders, use path relative
    //       to the path of the shader program instead of relative to the
    //       path of the executable.

    word = vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);
    if (word[0] == '#')
      {
      // comment line. do nothing and proceed to the next line.
      } // (word[0] == '#')
    else if (word == "VERT")
      {
      string vertfile = "shaders/"+vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);
      vtkVertexShader* vshader = vtkVertexShader::New();
      //vshader->DebugOn();
      const char* fname = vertfile.c_str();
      vtkDebugMacro(<<"Reading vertex shader from file "<<fname);
      vshader->ReadSourceTextFromFile(fname);
      this->Output->AddShaderObject(vshader);
      vshader->Delete(); vshader = NULL;
      } // if (word == "VERT")
    else if (word == "FRAG")
      {
      string fragfile = "shaders/"+vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);
      vtkFragmentShader* fshader = vtkFragmentShader::New();
      //fshader->DebugOn();
      const char* fname = fragfile.c_str();
      vtkDebugMacro(<<"Reading fragment shader from file "<<fname);
      fshader->ReadSourceTextFromFile(fname);
      this->Output->AddShaderObject(fshader);
      fshader->Delete(); fshader = NULL;
      } // if (word == "FRAG")
    else if (word == "UNIFORM")
      {
      this->ReadUniformFromLine(this->CurrentLine, linepos);
      }
    else
      {
      vtkWarningMacro(<<"Unknown keyword " << word.c_str()
		    <<" in file " << this->FileName
		    <<", line " << this->GetCurrentLineNumber()<<"!");
      } // else
    }

  this->CloseFile();
}

void vtkMyShaderProgramReader::ReadUniformFromLine(string line, unsigned int linepos)
{
  vtkDebugMacro(<<"Reading uniform from line:\n"<<line.c_str()
	<<"\n starting at position "<<linepos);
  vtkShaderUniform* uniform = NULL;
  string type = vtkBetterDataReader::ReadWord(line, linepos, linepos);
  string name = vtkBetterDataReader::ReadWord(line, linepos, linepos);

  if (type == "float")
    {
    float val = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    uniform = vtkUniformFloat::New();
    ((vtkUniformFloat*)uniform)->SetValue(val);
    }
  else if (type == "vec2")
    {
    float val0 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    float val1 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    uniform = vtkUniformVec2::New();
    ((vtkUniformVec2*)uniform)->SetValue(val0, val1);
    }
  else if (type == "vec3")
    {
    float val0 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    float val1 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    float val2 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    uniform = vtkUniformVec3::New();
    ((vtkUniformVec3*)uniform)->SetValue(val0, val1, val2);
    }
  else if (type == "vec4")
    {
    float val0 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    float val1 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    float val2 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    float val3 = vtkBetterDataReader::ReadFloat(line, linepos, linepos);
    uniform = vtkUniformVec4::New();
    ((vtkUniformVec4*)uniform)->SetValue(val0, val1, val2, val3);
    }
  else if ((type == "int")||(type == "sampler"))
  // deal with sampler as with int. Change if sampler gets more functionality
    {
    int val = vtkBetterDataReader::ReadInt(line, linepos, linepos);
    uniform = vtkUniformInt::New();
    ((vtkUniformInt*)uniform)->SetValue(val);
    }
  else
    {
    vtkDebugMacro("Unknown type "<<type.c_str()<<" for uniform "<<name.c_str()
	<<" on line:\n" <<line.c_str());
    return;
    }

  uniform->SetName(name.c_str());
  this->Output->AddShaderUniform(uniform);
  uniform->Delete(); uniform = NULL;
}

} // namespace bmia
