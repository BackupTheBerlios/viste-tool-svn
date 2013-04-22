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
 * vtkDTIComponentReader.cxx
 * by Tim Peeters
 *
 * 2006-01-03	Tim Peeters
 * - First version
 *
 * 2006-01-05	Tim Peeters
 * - Added ExecuteInformation to read the header and set the data extent.
 *
 * 2006-02-21	Tim Peeters
 * - Fixed bug with mirrored data compared to the old DTI tool.
 *   The only change is this->FileLowerLeftOn() in the constructor.
 */

#include "vtkDTIComponentReader.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkDTIComponentReader);

vtkDTIComponentReader::vtkDTIComponentReader()
{
  this->SetFileDimensionality(3);
  this->SetNumberOfScalarComponents(1);
  //this->SetDataByteOrderToBigEndian();
  this->SetDataScalarTypeToUnsignedShort(); // Float is also possible
  // correct value must be set before reading. This is done in vtkDTIReader2.

  // when reading the date, skip the dimensions listed at the beginning
  // of the file.
  //this->SetHeaderSize(3*sizeof(unsigned short)); // computed automatically

  this->FileLowerLeftOn();
  this->ReadDimensionsOn();
}

vtkDTIComponentReader::~vtkDTIComponentReader()
{
  // nothing to do
}

int vtkDTIComponentReader::CanReadFile(const char* fname)
{
  FILE *fp = fopen(fname, "rb");
  if (!fp)
    {
    return 0;
    }
//  else
//    {

  fclose(fp);
  return 1;
    // if better checks are done, 2 or 3 can be returned.
    // Check vtkImageReader2 documentation.
//    }
}

void vtkDTIComponentReader::ExecuteInformation()
{
	//const char* fname = this->GetFileName();
	// ignore possible FilePrefix since we only have one file.

	//this->OpenFile(); // creates an input stream this->File.
	if(this->readDimensions)
	{
		unsigned short dim[3];
		FILE* in_file;

		if (!(in_file = fopen(this->GetFileName(), "rb")))
		{
			vtkErrorMacro("Could not open file "<<this->GetFileName());
			return;
		}
		////VERY BAD!!!! CHANGE IT - JUST FOR ALARD NEAT 1000
		fread(&(dim[0]), sizeof(unsigned short), 1, in_file);
		fread(&(dim[1]), sizeof(unsigned short), 1, in_file);
		fread(&(dim[2]), sizeof(unsigned short), 1, in_file);

		//dim[0]=128;
		//dim[1]=128;
		//dim[2] =20;

		// extent in all 3 dimensions i goes from 0 to dim[i].
		this->SetDataExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1);
		//this->SetDataExtent(0, 127, 0, 127, 0, 19);

		fclose(in_file);
	}
  // now set the properties (such as extent) of the output data:
  this->vtkImageReader2::ExecuteInformation();
}

} // namespace bmia 
