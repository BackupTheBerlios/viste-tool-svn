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
