/*
 * bmiaNiftiReader.cxx
 *
 * 2011-04-04	Evert van Aart
 * - First version. This reader is based on the old "vtkNiftiReader" class by Tim
 *   Peeters, but with some changes. The reader now supports multiple types of 
 *   input data, including triangles ("NIFTI_INTENT_TRIANGLE"), which means that
 *   its output will not always be "vtkImageData". For this reason, the reader
 *   is no longer a child of "vtkImageDataReader2". Furthermore, the reader not
 *   includes support for MiNT extensions.
 *
 * 2011-05-10	Evert van Aart
 * - Added support for spherical harmonics using MiND. 
 *
 */


#ifndef bmia_NiftiReaderPlugin_bmiaNiftiReader_h
#define bmia_NiftiReaderPlugin_bmiaNiftiReader_h


/** Includes - VTK */

#include <vtkImageReader2.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkShortArray.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkMatrix4x4.h>

/** Includes - Qt */

#include <QFile>
#include <QtDebug>
#include <QMessageBox>
#include <QList>
#include <QString>
#include <QProgressDialog>

/** Includes - NIfTI */

#include "nifti/nifti1_io.h"

/** Includes - Custom Files */

#include "HARDI/SphereTriangulator.h"

namespace bmia {


/** Class for reading files in NIfTI format. The reader supports several different
	types of NIfTI data. The type of the input file is determined based on the 
	intent code, the dimensionality of the image, and/or the MiNT extensions,
	if available. Since NIfTI files can contain data that cannot be expressed as
	an image volume - such as a triangle array, which is represented with a
	"vtkIntArray" object - a data output can be any child of "vtkObject". It is
	up the "NiftiReaderPlugin" class to create the correct type of data set for
	the output objects. 
*/

class bmiaNiftiReader
{
	public:

		/** Constructor */

		bmiaNiftiReader();

		/** Destructor */

		~bmiaNiftiReader();

		/** Check if we can read the input file. Returns one if we can, zero otherwise.
			filename	Input filename. */

		virtual int CanReadFile(const char * filename);

		/** Read a NIfTI file. If successful, the data read from the NIfTI file
			is stored in the "outData" list, and its type is stored in the 
			"imageDataType" variable. On success, an empty string is returned;
			otherwise, the return string contains an error message.
			@param filename		Name of the NIfTI file.
			@param showProgress	Should the reader create a progress bar? */

		QString readNIfTIFile(const char * filename, bool showProgress = true);

		/** Return supported file extensions. */

		virtual const char * GetFileExtensions()
		{
			return ".nii .nii.gz";
		}

		/** Get the transformation matrix read from the NIfTI file. */
	
		vtkMatrix4x4 * getTransformMatrix()
		{
			return transformMatrix;
		};

		/** Delete all temporary objects. */

		void cleanUp();

		/** Enumeration for all supported data types. To add support for a new
			type of data: 1) Add a value to this enumeration; 2) Modify the
			"determineDataType" function, so that it can correctly detect NIfTI
			files of this data type; 3) Implement a parsing function for the data
			type; 4) Add support for the new data type in "readNIfTIFile". */

		enum NIfTIDataType
		{
			NDT_Unknown = 0,		/**< Unknown data type. */
			NDT_ScalarVolume,		/**< Single-component scalar volume. */
			NDT_DTITensors,			/**< Second-order tensors with six unique elements. */
			NDT_DiscreteSphere,		/**< Discrete sphere function, incl. sphere directions and (optionally) topology. */
			NDT_SphericalHarm,		/**< Spherical harmonics. */
			NDT_Triangles,			/**< Triangles, used to describe topology. */
			NDT_GenericVector		/**< Vector of undetermined purpose. Will be split into N scalar volumes. */
		};

		/** Data type of the NIfTI image. */

		NIfTIDataType imageDataType;

		/** List of the output objects, all cast to "vtkObject" pointers. Most pointers
			will be to "vtkImageData" objects, but other data types are also possible
			(like "vtkIntArray" for "NDT_Triangles"). The final size of the list depends
			on "imageDataType": For most types it will be one, but some it will
			be more (e.g., for "NDT_GenericVectors", we will have N output objects). */

		QList<vtkObject *> outData;

	protected:

		/** The NIfTI image object constructed when reading the ".nii" file. */

		nifti_image * NiftiImage;

	private:
    
		/** Transformation matrix of the NIfTI file (if any). */
	
		vtkMatrix4x4 * transformMatrix;

		/** Determine the data type of the NIfTI image by looking at its intent code,
			its dimensionality, and/or its MiND extensions. */

		bool determineDataType();

		/** Create an image with one scalar value per voxel. Used for both 
			"NDT_ScalarVolume" and "NDT_GenericVector"; in the latter case, the
			"component" value describes which component of the input vector
			should be used to construct the scalar image volume. 
			@param component	Target output component. */

		vtkImageData * parseScalarVolume(int component = 0);

		/** Create an image containing second-order DTI tensors. */

		vtkImageData * parseDTIVolume();

		/** Create an image containing, per voxel, the radius for each of the
			spherical directions. These spherical directions, which are read
			from a MiND extension, are stored in a scalar array named 
			"Spherical Directions", which is attached to the output image.
			The topology of the output glyphs can be read from a separate NIfTI
			file (with the same name as the current file, appended with "_topo"), 
			or, failing that, constructed here. In either case, an array describing 
			the topology (triangles) is also attached to the output image. */

		vtkImageData * parseDiscreteSphereVolume();

		/** Create an image containing, for each voxel, a set of Spherical Harmonics
			coefficients. The number of coefficients (i.e., the vector length) should
			be 1 (0th order), 6 (2nd), 15 (4th), 28 (6th) or 45 (8th). The coefficients
			are stored in ascending order: First the coefficient for l = 0, then the
			five coefficients for l = 2, and so on. Uses the MiND extensions. */

		vtkImageData * parseSphericalHarmonicsVolume();

		/** Create an array of 3-element integer vectors. These integers represent
			point indices of the vertices of a discrete sphere function. Each set
			of three point indices describes a triangle; these triangles can later
			be used to create the geometry glyphs. */

		vtkIntArray * parseTriangles();

		/** Compare one string to a target string. Used for the MiND identifiers. 
			@param id			Input string.
			@param target		Target string.
			@param n			Maximum length of the comparison. */

		bool compareMiNDID(char * id, char * target, int n);

		/** Find an extension with an "ecode" identifier equal to the target ID.
			Returns a pointer to the extension on success, and NULL otherwise.
			Searching starts at the "extPos" position in the extension list, and
			on success, the position of the target extension is copied to "extPos".
			@param targetID		Desired "ecode" value.
			@param extPos		Starting location for search and output index. */

		nifti1_extension * findExtension(int targetID, int & extPos);

		/** Create an image using a specified double array.
			@param data			Double array containing the image data.
			@param numberOfComponents	Number of components in the output image. 
			@param arrayName	Desired name for the scalar array. */

		vtkImageData * createimageData(double * data, int numberOfComponents, const char * arrayName);

		/** Copy of the input filename. */

		QString filenameQ;

}; // class bmiaNiftiReader


} // namespace bmia


#endif // bmia_NiftiReaderPlugin_bmiaNiftiReader_h
