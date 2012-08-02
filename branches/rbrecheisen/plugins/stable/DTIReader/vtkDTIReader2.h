/*
 * vtkDTIReader2.h
 *
 * 2006-01-05	Tim Peeters
 * - First version
 *
 * 2006-01-10	Tim Peeters
 * - Finished implementation. A lot of code is copied from the old
 *   "vtkTensorDataReader.cxx", but here the new "vtkDTIComponentReader"
 *   classes and "vtkImageReader2" functionality are used.
 *
 * 2006-01-16	Tim Peeters
 * - Hmm.. I did not finish the implementation on 2006-01-10 ;).
 * - Added code for merging of input datasets and creating an output
 *   tensor dataset.
 *
 * 2006-03-08	Tim Peeters
 * - Fixed resetting of spacing of the output data somewhere when the
 *   pipeline is executed by properly setting "DataSpacing" in "ExecuteInformation".
 * - Use "DataSpacing" from superclass instead of "VoxelSize".
 *
 * 2006-05-12	Tim Peeters
 * - Add support for a matrix to transform the data that is read with.
 *
 * 2010-09-03	Tim Peeters
 * - From now on, store the (symmetrical!) tensors in a 6-valued scalar
 *   array in the output instead of 9-valued tensor array.
 * 
 * 2010-11-18	Evert van Aart
 * - Moved "vtkBetterDataReader" and "vtkDTIComponentReader" to the libraries
 *   direction, since I need them for the new HARDI reader.
 *
 * 2011-01-14	Evert van Aart
 * - Structural set is now created as a separate "vtkImageData" object.
 *
 * 2011-03-14	Evert van Aart
 * - Changed data type of structural image to double. 
 *
 * 2011-03-31	Evert van Aart
 * - Allowed the reader to read doubles.
 *
 */


#ifndef bmia_vtkDTIReader2_h
#define bmia_vtkDTIReader2_h


/** Includes - VTK */

#include <vtkImageReader2.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>

/** Includes - Custom Files */

#include "Helpers/vtkBetterDataReader.h"
#include "Helpers/vtkDTIComponentReader.h"

/** Includes - C++ */

#include <list>

/** Includes - Qt */

#include <QFile>


namespace bmia {


class vtkDTIComponentReader;


/** New class for reading tensor data from one ".dti" file and a set of ".dat" files.
	This one is supposed to be simpler, faster, better than the old one. It uses 
	"vtkDTIComponentReader" for the reading of the ".dat" files. Tensors are transformed 
	based on the transformation matrix in the ".dti" file. The structural information
	(the "I" file) is available as a separate "vtkImageData" object. 
 */


class vtkDTIReader2 : public vtkImageReader2
{
	public:

		/** Enumeration used for the scan direction. */

		enum DTIScanDirection
		{
			BMIA_SCAN_DIRECTION_UNDEFINED	= 0,
			BMIA_SCAN_DIRECTION_TRANSVERSAL,
			BMIA_SCAN_DIRECTION_SAGITTAL,
			BMIA_SCAN_DIRECTION_CORONAL
		};

		/** Enumeration used for the different DTI components. */

		enum DTIComponent
		{
			BMIA_XX_COMPONENT_INDEX  = 0,
			BMIA_XY_COMPONENT_INDEX,
			BMIA_XZ_COMPONENT_INDEX,
			BMIA_YY_COMPONENT_INDEX,
			BMIA_YZ_COMPONENT_INDEX,
			BMIA_ZZ_COMPONENT_INDEX,
			BMIA_I_COMPONENT_INDEX
		};

		/** Total number of DTI components. */

		static const int BMIA_NUMBER_OF_COMPONENTS = 7;

		/** Constructor Call */
		
		static vtkDTIReader2 * New();

		/** Test whether the specified file can be opened. 
			@param fname	Desired filename. */

		virtual int CanReadFile(const char * fname);

		/** Return supported file extension. */
	  
		virtual const char * GetFileExtensions()
		{
			return ".dti";
		}

		/** Return an image containing the structural information. */

		vtkImageData * getStructuralInformation()
		{
			return structImage;
		}

	/** Return description of the reader. */

	virtual const char * GetDescription()
	{
		return "BMT DTI format";
	}

	protected:
  
		/** Constructor */

		vtkDTIReader2();

		/** Destructor */
  
		~vtkDTIReader2();


		/** Reads header file (".dti"). */

		virtual void ExecuteInformation();

		/** Reads all component data files (".dat").
			@param out	Output data. Not used. */
			
		virtual void ExecuteData(vtkDataObject * out);

		/** Read the next line of the input file. */

		virtual bool NextLine();

		/** Try to read the header; return true on succes. */

		bool ReadHeader();

		/** Try to read the data type; return true on succes. */

		bool ReadDataType();

		/** Try to read the scan direction; return true on succes. */

		bool ReadScanDirection();

		/** Try to read the filenames of the ".dat" files; return true on succes. */

		bool ReadComponentFileNames();

		/** Try to read the voxel size; return true on succes. */

		bool ReadVoxelSize();

		/** Try to read the tensor transformation matrix; return true on succes. */

		void ReadTensorTransformMatrix();

		/** Delete the transformation matrix, free all allocated space. */

		void CleanTensorTransformMatrix();

		/** Reset parameters, free up allocated memory. Called when any part of the 
			reading process fails, and when destroying the object. */

		void CleanUp();

		/** Transform the loaded tensor using the loaded transformation matrix.
			@param inTensor		Input tensor array. */

		void transformTensors(vtkDataArray * inTensors);

		/** Set to true if "ExecuteInformation" has completed successfully. */

		bool InformationExecuted;

		/** Current line of the input file. */

		std::string CurrentLine;

		/** Input stream used to read the file. */

		ifstream * IStream;

		/** Scan direction of the DTI data. */
  
		DTIScanDirection ScanDirection;
  		
		/** Data type of the ".dat" files. */

		int DataType;

		/** Struct containing the name of a DTI component file, and
			its associated component ID. */

		struct componentInfo
		{
			std::string name;
			DTIComponent ID;
		};

		/** List of all component file names and their IDs. Should contain
			seven elements after parsing the header file. */
	
		std::list<componentInfo> ComponentInfoList;

		/** 6x6 matrix for transforming the tensors. */
		
		float ** TensorTransformMatrix;

		/** Used during the reading of the ".dat" files. */

		vtkImageReader2 * componentReader;
		vtkImageData * componentImageData;
		vtkPointData * componentPointData;
		vtkDataArray * componentScalars;

		/** Output array of the tensor image. */
		
		vtkDataArray * outputTensorArray;

		/** Image data and scalar for the structural information image. */

		vtkImageData * structImage;
		vtkDataArray * structScalars;


}; // class vtkDTIReader2


} // namespace bmia


#endif // bmia_vtkDTIReader2_h
