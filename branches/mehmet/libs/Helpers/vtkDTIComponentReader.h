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

/*
 * vtkDTIComponentReader.h
 * by Tim Peeters
 *
 * 2006-01-03	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkDTIComponentReader_h
#define bmia_vtkDTIComponentReader_h

#include <vtkImageReader2.h>

namespace bmia {

/**
 * Class for reading one .dat component file of a BMT DTI dataset.
 * Remember to set FileName, DataScalarType, DataExtent, DataSpacing,
 * and DataOrigin.
 * Basically it is just a vtkImageReader2 with the correct dimensionality,
 * number of scalar components, and header size set.
 */
class vtkDTIComponentReader : public vtkImageReader2
{
public:
  static vtkDTIComponentReader* New();

  virtual int CanReadFile(const char* fname);

  // Description:
  // .dat
  virtual const char* GetFileExtensions()
    {
    return ".dat";
    }

  // Description:
  //
  virtual const char* GetDescription()
    {
    return "Binary DTI component file";
    }

	void ReadDimensionsOff()
	{
		this->readDimensions = false;
	}

	void ReadDimensionsOn()
	{
		this->readDimensions = true;
	}

protected:
  vtkDTIComponentReader();
  ~vtkDTIComponentReader();

  virtual void ExecuteInformation();

  bool readDimensions;

private:

}; // class vtkDTIComponentReader

} // namespace bmia

#endif // bmia_vtkDTIComponentReader_h
