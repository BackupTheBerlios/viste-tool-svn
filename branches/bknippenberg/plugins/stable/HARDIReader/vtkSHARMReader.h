/**
 * vtkSHARMReader.h
 *
 * 2010-12-01	Evert van Aart
 * - First version. Based on "vtkSHARMReader" of the old DTITool.
 *
 */


#ifndef bmia_vtkSHARMReader_h
#define bmia_vtkSHARMReader_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkImageReader2.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>

/** Includes - Custom Files */

#include "Helpers/vtkBetterDataReader.h"

/** Includes - STL */

#include <assert.h>
#include <string>

/** Includes - Qt */

#include <QFile>


namespace bmia {


/** Reader for volumes of spherical harmonics coefficients. Spherical Harmonics
	are stored as a single ".sharm" file, which is a header file, and a number of
	".dat" files, which contain the actual harmonics. For fourth-order spherical
	harmonics (which is the default), there will be 15 ".dat" files. These harmonics
	will be combined into a single VTK data array, which is stored in a "vtkImageData"
	object. The reader can also read the files "min.dat" and "max.dat", which contain
	the minimum and maximum radius per voxel. 
*/

class vtkSHARMReader : public vtkImageReader2
{
	public:
  
		/** Constructor Call */

		static vtkSHARMReader * New();

		/** Test whether the specified file can be opened. 
			@param fname	Desired filename. */

		virtual int CanReadFile(const char * fname);

		/** Return supported file extension. */
	
		virtual const char * GetFileExtensions()
		{
			return ".sharm";
		}

		/** Return description of the reader. */

		virtual const char * GetDescription()
		{
			return "Spherical harmonics coefficients";
		}

		/** Return list of parameter names. */

		std::list<std::string> getParamNames()
		{
			return paramNames;
		}

		/** Return list of parameter values (as doubles). */

		std::list<double> getParamValues()
		{
			return paramValues;
		}

	protected:

		/** File type of the header. File type determines formatting, and is
			always defined on the very first line of a file. */

		enum fileType
		{
			FileType_SHARM1 = 0,
			FileType_SHARM2,
			FileType_SHARM3,
			FileType_SHCOEFF
		};

		/** Type of the glyphs (i.e., model used to compute the	spherical harmonics.
			TODO: This should be a public enumeration in some HARDI Types class. */

		enum glyphType
		{
			GlyphType_DOT_PARAMETRIC = 0,
			GlyphType_ADC,
			GlyphType_QBALL,
			GlyphType_Unknown
		};

		/** Constructor */

		vtkSHARMReader();

		/** Destructor */

		~vtkSHARMReader();

		/** Reset parameters, free up allocated memory. Called when any part of the 
			reading process fails, and when it finishes successfully. */

		void CleanUp();

		/** Reads header file (".sharm"). */

		virtual void ExecuteInformation();

		/** Reads all component data files (".dat").
			@param out	Output data. Not used. */
			
		virtual void ExecuteData(vtkDataObject * out);

		/** Read the next line of the input file. */

		virtual bool NextLine();

		/** Set to true if "ExecuteInformation" has completed successfully. */
  
		bool InformationExecuted;

		/** Current line of the input file. */

		std::string CurrentLine;

		/** Input stream used to read the file. */

		ifstream * IStream;

		/** Dimensions of the input image. */

		int Dimensions[3];

		/** List of filenames of the ".dat" files. */

		std::list<std::string> ComponentFileNames;

		/** Try to read the header; return true on succes. */

		bool ReadHeader();

		/** Try to read the data type; return true on succes. */

		bool ReadDataType();

		/** Try to read the B-value; return true on succes. */

		bool ReadBValue();

		/** Try to read the voxel size; return true on succes. */

		bool ReadVoxelSize();

		/** Try to read the dimensions; return true on succes. */

		bool ReadDimensions();

		/** Try to read the parameters used to compute the spherical harmonics. */

		bool ReadParameters();

		/** Try to read the filenames of the ".dat" files; return true on succes. */

		bool ReadComponentFileNames();

		/** Try to read the "min.dat" and "max.dat" files, which are sometimes included
			with the ".sharm" header files. These files are not manditory, so it does not
			matter whether or not this function succeeds. 
			@param numberOfTuples	Number of voxels in the input. */

		void readMinMaxFiles(vtkIdType numberOfTuples);

		/** Number of spherical harmonic components. Default value is 15. */

		int numberOfComponents;

		/** Data type of the ".dat" files. */

		int dataType;

		/** Output data array. */

		vtkDataArray * outputArray;

		/** Array containing minimum/maximum radius per voxel. */

		vtkDataArray * minMaxArray;

		/** Used during the reading of the ".dat" files. */

		vtkImageReader2 * componentReader;
		vtkImageData * componentImageData;
		vtkPointData * componentPointData;
		vtkDataArray * componentScalars;

		/** Header file type. */

		fileType fType;

		/** List of parameter names. */

		std::list<std::string> paramNames;

		/** List of parameter values (as doubles). */

		std::list<double> paramValues;

}; // class vtkSHARMReader


} // namespace bmia


#endif // bmia_vtkSHARMReader_h
