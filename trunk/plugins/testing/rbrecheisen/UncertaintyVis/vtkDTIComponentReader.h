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
