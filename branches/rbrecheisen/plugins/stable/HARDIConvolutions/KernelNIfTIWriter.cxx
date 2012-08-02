/*
 * KernelNIfTIWriter.cxx
 *
 * 2011-07-26	Evert van Aart
 * - First version.
 * 2011-09-12	Ralph Brecheisen
 * - Added preprocessor directives to conditionally compile 'strcpy' or 'strcpy_s'
 *
 */


/** Includes */

#include "KernelNIfTIWriter.h"


namespace bmia {


KernelNIfTIWriter::KernelNIfTIWriter()
{
	// Initialize members
	this->fileName		= "";
	this->dim[0]		= 0;
	this->dim[1]		= 0;
	this->dim[2]		= 0;
	this->directions	= NULL;
}


KernelNIfTIWriter::~KernelNIfTIWriter()
{

}


bool KernelNIfTIWriter::writeKernel(double * kernelData)
{
	// We should have a directions list and a file name
	if (!(this->directions) || this->fileName.isEmpty())
		return false;

	// Create a new NIfTI image struct
	nifti_image * outImage = new nifti_image;

	outImage->aux_file[0]	= '\0';					// No auxiliary file
	outImage->descrip[0]	= '\0';					// No description
	outImage->datatype		= DT_DOUBLE;			// Using doubles...
	outImage->nbyper		= 8;					// ...so 8 byte per value...
	outImage->swapsize		= 8;					// ...and the swap size is also 8.
	outImage->iname_offset	= 1024;					// Offset for the image name
	outImage->xyz_units		= NIFTI_UNITS_MM;		// Spatial units (not important)
	outImage->time_units	= NIFTI_UNITS_UNKNOWN;	// Temporal units (not used)
	outImage->byteorder		= nifti_short_order();	// Use byte order of this CPU

	// Store the data pointer
	outImage->data = (void *) kernelData;

	// Copy the file name to both file name fields
	QByteArray fileNameBA = this->fileName.toLocal8Bit();
	char * fileNameChar	= fileNameBA.data();
	outImage->fname = (char *) malloc((this->fileName.length() + 1) * sizeof(char));
	outImage->iname = (char *) malloc((this->fileName.length() + 1) * sizeof(char));
#ifdef _WIN32
	strcpy_s(outImage->fname, this->fileName.length() + 1, fileNameChar);
	strcpy_s(outImage->iname, this->fileName.length() + 1, fileNameChar);
#else
	strcpy(outImage->fname, fileNameChar);
	strcpy(outImage->iname, fileNameChar);
#endif
	// Using the "Vector" intent code
	outImage->intent_code = NIFTI_INTENT_VECTOR;
	outImage->intent_p1   = 0.0f;
	outImage->intent_p2   = 0.0f;
	outImage->intent_p3   = 0.0f;
	
	// Set the intent name to "MiND"
	char * intentName = (char*) "MiND";
#ifdef _WIN32
	strcpy_s(&(outImage->intent_name[0]), 5, intentName);
#else
	strcpy(&(outImage->intent_name[0]), intentName);
#endif

	// Set the dimensions
	outImage->dim[0] = 5;
	outImage->dim[1] = this->dim[0];
	outImage->dim[2] = this->dim[1];
	outImage->dim[3] = this->dim[2];
	outImage->dim[4] = 1;
	outImage->dim[5] = this->directions->size();
	outImage->dim[6] = 0;
	outImage->dim[7] = 0;

	outImage->ndim = 5;
	outImage->nx = this->dim[0];
	outImage->ny = this->dim[1];
	outImage->nz = this->dim[2];
	outImage->nt = 1;
	outImage->nu = this->directions->size();
	outImage->nv = 0;
	outImage->nw = 0;

	// Compute the total number of values
	outImage->nvox = this->dim[0] * this->dim[1] * this->dim[2] * this->directions->size();

	// Set the spacing
	outImage->dx = 1.0f;
	outImage->dy = 1.0f;
	outImage->dz = 1.0f;
	outImage->dt = 1.0f;
	outImage->du = 1.0f;
	outImage->dv = 0.0f;
	outImage->dw = 0.0f;

	outImage->pixdim[0] = 0.0f;
	outImage->pixdim[1] = 1.0f;
	outImage->pixdim[2] = 1.0f;
	outImage->pixdim[3] = 1.0f;
	outImage->pixdim[4] = 1.0f;
	outImage->pixdim[5] = 1.0f;
	outImage->pixdim[6] = 0.0f;
	outImage->pixdim[7] = 0.0f;

	// Set the NIfTI type (single ".nii" file)
	outImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;

	// Initialize some stuff that we do not use
	outImage->cal_min			= 0.0f;
	outImage->cal_max			= 0.0f;
	outImage->scl_slope			= 0.0f;
	outImage->scl_inter			= 0.0f;
	outImage->toffset			= 0.0f;
	outImage->freq_dim			= 0;
	outImage->phase_dim			= 0;
	outImage->slice_dim			= 0;
	outImage->slice_code		= 0;
	outImage->slice_start		= 0;
	outImage->slice_end			= 0;
	outImage->slice_duration	= 0.0f;
	outImage->analyze75_orient	= a75_orient_unknown;

	// The kernel NIfTI image does not have a transformation matrix
	outImage->qform_code		= 0;
	outImage->quatern_b			= 0.0f;
	outImage->quatern_c			= 0.0f;
	outImage->quatern_d			= 0.0f;
	outImage->qoffset_x			= 0.0f;
	outImage->qoffset_y			= 0.0f;
	outImage->qoffset_z			= 0.0f;
	outImage->qfac				= 0.0f;

	outImage->sform_code = 0;
	outImage->sto_xyz.m[0][0] = outImage->sto_xyz.m[1][1] = outImage->sto_xyz.m[2][2] = outImage->sto_xyz.m[3][3] = 1.0f;
	outImage->sto_xyz.m[0][1] = outImage->sto_xyz.m[0][2] = outImage->sto_xyz.m[0][3] = 0.0f;
	outImage->sto_xyz.m[1][0] = outImage->sto_xyz.m[1][2] = outImage->sto_xyz.m[1][3] = 0.0f;
	outImage->sto_xyz.m[2][0] = outImage->sto_xyz.m[2][1] = outImage->sto_xyz.m[2][3] = 0.0f;
	outImage->sto_xyz.m[3][0] = outImage->sto_xyz.m[3][1] = outImage->sto_xyz.m[3][2] = 0.0f;

	// Initialize the extension list
	outImage->num_ext  = 0;
	outImage->ext_list = NULL;
	
	// Add the main MiND extension
	nifti_add_extension(outImage, "DISCSPHFUNC", 24, NIFTI_ECODE_MIND_IDENT);

	// Loop through all directions
	for (std::vector<double *>::iterator i = this->directions->begin(); i != this->directions->end(); ++i)
	{
		float angles[2];

		double * dir = (*i);

		// Convert the unit vector to spherical coordinates
		angles[0] = atan2((sqrt(pow(dir[0], 2) + pow(dir[1], 2))), dir[2]);
		angles[1] = atan2(dir[1], dir[0]);

		// Add the two angles to the NIfTI file as an extension
		nifti_add_extension(outImage, (char *) &(angles[0]), 2 * sizeof(float), NIFTI_ECODE_SPHERICAL_DIRECTION);
	}

	// Write the NIfTI image!
	nifti_image_write(outImage);

	// Free the allocated file names
	if (outImage->fname)
		free (outImage->fname);

	if (outImage->iname)
		free (outImage->iname);

	// Delete the NIfTI image struct
	delete outImage;

	// Done!
	return true;
}


} // namespace bmia
