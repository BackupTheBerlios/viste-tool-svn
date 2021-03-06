/*
 * KernelNIfTIReader.cxx
 *
 * 2011-07-28	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "KernelNIfTIReader.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

KernelNIfTIReader::KernelNIfTIReader()
{
	// Initialize variables
	this->fileName				= "";
	this->dim[0]				= 0;
	this->dim[1]				= 0;
	this->dim[2]				= 0;
	this->numberOfDirections	= 0;
}


//------------------------------[ Destructor ]-----------------------------\\

KernelNIfTIReader::~KernelNIfTIReader()
{

}


//----------------------------[ getDimensions ]----------------------------\\

bool KernelNIfTIReader::getDimensions(int * outDim)
{
	// Copy the file name to both file name fields
	QByteArray fileNameBA = this->fileName.toLocal8Bit();
	char * fileNameChar	= fileNameBA.data();

	// Read the specified NIfTI file
	nifti_image * NiftiImage = nifti_image_read(fileNameChar, 1);

	if (!NiftiImage)
		return false;

	// Image should be 5D, with the fourth dimensions (time) set to one
	if (NiftiImage->dim[0] != 5	|| 
		NiftiImage->dim[4] != 1 )
	{
		nifti_image_free(NiftiImage);
		return false;
	}

	// Check if this combination of dimensions is supported
	if (!(	(NiftiImage->dim[1] == 1 && NiftiImage->dim[2] == 1 && NiftiImage->dim[3] == 1) || 
			(NiftiImage->dim[1] == 3 && NiftiImage->dim[2] == 3 && NiftiImage->dim[3] == 3) ||
			(NiftiImage->dim[1] == 5 && NiftiImage->dim[2] == 5 && NiftiImage->dim[3] == 5) || 
			(NiftiImage->dim[1] == 7 && NiftiImage->dim[2] == 7 && NiftiImage->dim[3] == 7) || 
			(NiftiImage->dim[1] == 9 && NiftiImage->dim[2] == 9 && NiftiImage->dim[3] == 9) ))
	{
		nifti_image_free(NiftiImage);
		return false;
	}
	
	// If so, copy them to the output
	outDim[0] = NiftiImage->dim[1];
	outDim[1] = NiftiImage->dim[2];
	outDim[2] = NiftiImage->dim[3];

	nifti_image_free(NiftiImage);

	return true;
}


//------------------------------[ readKernel ]-----------------------------\\

bool KernelNIfTIReader::readKernel(double * kernelData)
{
	// Copy the file name to both file name fields
	QByteArray fileNameBA = this->fileName.toLocal8Bit();
	char * fileNameChar	= fileNameBA.data();

	// Read the specified NIfTI file
	nifti_image * NiftiImage = nifti_image_read(fileNameChar, 1);

	if (!NiftiImage)
		return false;

	// Check the dimensions. Image should be 5D, with the fourth dimensions 
	// (time) set to one. Spatial dimensions should match those set by the
	// convolution filter; vector length should equal the number of directions.

	if (	NiftiImage->dim[0] != 5							|| 
			NiftiImage->dim[1] != this->dim[0]				|| 
			NiftiImage->dim[2] != this->dim[1]				|| 
			NiftiImage->dim[3] != this->dim[2]				|| 
			NiftiImage->dim[4] != 1							|| 
			NiftiImage->dim[5] != this->numberOfDirections	)
	{
		nifti_image_free(NiftiImage);
		return false;
	}

	// We only supported certain kernel sizes
	if (!(	(NiftiImage->dim[1] == 1 && NiftiImage->dim[2] == 1 && NiftiImage->dim[3] == 1) || 
			(NiftiImage->dim[1] == 3 && NiftiImage->dim[2] == 3 && NiftiImage->dim[3] == 3) ||
			(NiftiImage->dim[1] == 5 && NiftiImage->dim[2] == 5 && NiftiImage->dim[3] == 5) || 
			(NiftiImage->dim[1] == 7 && NiftiImage->dim[2] == 7 && NiftiImage->dim[3] == 7) || 
			(NiftiImage->dim[1] == 9 && NiftiImage->dim[2] == 9 && NiftiImage->dim[3] == 9) ))
	{
		nifti_image_free(NiftiImage);
		return false;
	}

	// Data type should be doubles
	if (NiftiImage->datatype != DT_DOUBLE || NiftiImage->nbyper != 8)
	{
		nifti_image_free(NiftiImage);
		return false;
	}

	// Intent name should be "MiND"
	char * intentName = &(NiftiImage->intent_name[0]);

	if (strcmp(intentName, "MiND") != 0)
	{
		nifti_image_free(NiftiImage);
		return false;
	}

	// The number of extensions should be one more than the number of directions
	if (NiftiImage->num_ext != this->numberOfDirections + 1)
	{
		nifti_image_free(NiftiImage);
		return false;
	}

	nifti1_extension ext = NiftiImage->ext_list[0];
	char * extText = (char *) ext.edata;

	// The first extension should have the "DISCSPHFUNC" string
	if (ext.ecode != 18 || ext.esize != 32 || strcmp(extText, "DISCSPHFUNC") != 0)
	{
		nifti_image_free(NiftiImage);
		return false;
	}

	// All other extensions should have size 16 and code 22
	for (int i = 0; i < this->numberOfDirections; ++i)
	{
		ext = NiftiImage->ext_list[i + 1];

		if (ext.ecode != 22 || ext.esize != 16)
		{
			nifti_image_free(NiftiImage);
			return false;
		}
	}

	// Copy the data to the output
	double * inArray = (double *) NiftiImage->data;
	int numberOfValues = this->dim[0] * this->dim[1] * this->dim[2] * this->numberOfDirections;

	for (int i = 0; i < numberOfValues; ++i)
	{
		kernelData[i] = inArray[i];
	}

	// Done!
	nifti_image_free(NiftiImage);

	return true;
}


} // namespace bmia
