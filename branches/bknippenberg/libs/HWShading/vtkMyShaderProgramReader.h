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
