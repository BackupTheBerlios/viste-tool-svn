/*
 * vtkImageMask2.cxx
 *
 * 2011-07-08	Evert van Aart
 * - First version. 
 * 
 */


/** Includes */

#include "vtkImageMask2.h"


namespace bmia {


vtkImageMask2::vtkImageMask2()
{
	// Set pointers to NULL
	this->inputImage0 = NULL;
	this->inputImage1 = NULL;
	this->outputImage = NULL;

	// Initialize matrices to identity matrices
	this->MSource = vtkMatrix4x4::New();
	this->MSource->Identity();

	this->MMask = vtkMatrix4x4::New();
	this->MMask->Identity();

	// Inversion is off by default
	this->invert = false;
}


vtkImageMask2::~vtkImageMask2()
{
	// Delete matrices
	if (this->MSource)
	{
		this->MSource->Delete();
		this->MSource = NULL;
	}

	if (this->MMask)
	{
		this->MMask->Delete();
		this->MMask = NULL;
	}

	this->MOut = NULL;
}


void vtkImageMask2::clearMatrices()
{
	// Delete existing matrices
	if (this->MSource)
	{
		this->MSource->Delete();
		this->MSource = NULL;
	}

	if (this->MMask)
	{
		this->MMask->Delete();
		this->MMask = NULL;
	}

	// Create identity matrices
	this->MSource = vtkMatrix4x4::New();
	this->MSource->Identity();

	this->MMask = vtkMatrix4x4::New();
	this->MMask->Identity();

	this->MOut = NULL;
}


void vtkImageMask2::setSourceMatrix(vtkMatrix4x4 * m)
{
	// Delete existing matrix
	if (this->MSource)
	{
		this->MSource->Delete();
	}

	// Create a new matrix
	this->MSource = vtkMatrix4x4::New();

	// Either copy the input matrix...
	if (m)
	{
		this->MSource->DeepCopy(m);
	}
	// ...or use the identity matrix
	else
	{
		this->MSource->Identity();
	}
}


void vtkImageMask2::setMaskMatrix(vtkMatrix4x4 * m)
{
	if (this->MMask)
	{
		this->MMask->Delete();
	}

	this->MMask = vtkMatrix4x4::New();

	if (m)
	{
		this->MMask->DeepCopy(m);
	}
	else
	{
		this->MMask->Identity();
	}
}


void vtkImageMask2::Update()
{
	// Set output image and output matrix to NULL
	this->MOut			= NULL;
	this->outputImage	= NULL;

	// Get the source image, and its point data and scalars
	vtkImageData * sourceImage = this->inputImage0;

	if (!sourceImage)
	{
		return;
	}

	vtkPointData * sourcePD = sourceImage->GetPointData();

	if (!sourcePD)
	{
		return;
	}

	vtkDataArray * sourceScalars = sourcePD->GetScalars();

	if (!sourceScalars)
	{
		return;
	}

	// Get the masking image, and its point data and scalars
	vtkImageData * maskImage = this->inputImage1;

	if (!maskImage)
	{
		return;
	}

	vtkPointData * maskPD = maskImage->GetPointData();

	if (!maskPD)
	{
		return;
	}

	vtkDataArray * maskScalars = maskPD->GetScalars();

	if (!maskScalars)
	{
		return;
	}

	// Create the output image, and get its point data
	this->outputImage = vtkImageData::New();
	vtkPointData * outPD = this->outputImage->GetPointData();

	// Output scalar array
	vtkDataArray * outScalars;

	// Get the dimensions of both images
	int sourceDims[3];
	sourceImage->GetDimensions(sourceDims);

	int maskDims[3];
	maskImage->GetDimensions(maskDims);

	// If both images have the same size, masking is easy!
	if (sourceDims[0] == maskDims[0] && sourceDims[1] == maskDims[1] && sourceDims[2] == maskDims[2])
	{
		// Use the source matrix for the output (both images probably have the same 
		// transformation matrix, but since they're already the same size, we do
		// not check for it).

		this->MOut = this->MSource;

		// Output image has the same size as the input image
		this->outputImage->SetDimensions(sourceDims[0], sourceDims[1], sourceDims[2]);

		// Set spacing and origin of the output
		double spacing[3];
		sourceImage->GetSpacing(spacing);
		this->outputImage->SetSpacing(spacing[0], spacing[1], spacing[2]);

		double origin[3];
		sourceImage->GetOrigin(origin);
		this->outputImage->SetOrigin(origin[0], origin[1], origin[2]);

		// Get the number of scalars of the input
		int numberOfScalars = sourceScalars->GetNumberOfTuples();

		// Create the output scalar array
		outScalars = vtkDataArray::CreateDataArray(VTK_DOUBLE);
		outScalars->SetNumberOfComponents(1);
		outScalars->SetNumberOfTuples(numberOfScalars);

		// Loop through all scalars
		for (int i = 0; i < numberOfScalars; ++i)
		{
			// Get the value of the mask
			double maskValue = maskScalars->GetTuple1(i);

			// Determine whether or not the source value should be copied to the output
			bool copySource = false;

			if (this->invert)
				copySource = (maskValue == 0.0);
			else
				copySource = (maskValue != 0.0);

			// Copy the source value...
			if (copySource)
			{
				double sourceValue = sourceScalars->GetTuple1(i);
				outScalars->SetTuple1(i, sourceValue);
			}
			// ...or set the output to zero
			else
			{
				outScalars->SetTuple1(i, 0.0);
			}
		}
	}

	// If the images have different sizes, masking is less easy. In this case,
	// we map the point coordinates of the masking image (including origin and
	// spacing) to the same space as the source image, using the transformation
	// matrices of both images. After that, we interpolate the source value at
	// the transformed position, and add this value to the output.

	else
	{
		// Use the masking matrix for the output
		this->MOut = this->MMask;

		// The output has the same size, spacing, and origin as the masking image
		this->outputImage->SetDimensions(maskDims[0], maskDims[1], maskDims[2]);

		double spacing[3];
		maskImage->GetSpacing(spacing);
		this->outputImage->SetSpacing(spacing[0], spacing[1], spacing[2]);

		double origin[3];
		maskImage->GetOrigin(origin);
		this->outputImage->SetOrigin(origin[0], origin[1], origin[2]);

		// Get the number of scalars in the masking image
		int numberOfScalars = maskScalars->GetNumberOfTuples();

		// Create the output scalar array
		outScalars = vtkDataArray::CreateDataArray(VTK_DOUBLE);
		outScalars->SetNumberOfComponents(1);
		outScalars->SetNumberOfTuples(numberOfScalars);

		// Invert the matrix of the source image
		vtkMatrix4x4 * invMSource = vtkMatrix4x4::New();
		vtkMatrix4x4::Invert(this->MSource, invMSource);

		// Loop through all scalars
		for (int i = 0; i < numberOfScalars; ++i)
		{
			// Get the mask value
			double maskValue = maskScalars->GetTuple1(i);

			// Determine whether or not we need to add the source value for this point
			bool copySource = false;

			if (this->invert)
				copySource = (maskValue == 0.0);
			else
				copySource = (maskValue != 0.0);

			// If this point is masked off, set the output value to zero
			if (!copySource)
			{
				outScalars->SetTuple1(i, 0.0);
				continue;
			}

			// Get the coordinates of the current point
			double p3[3];
			maskImage->GetPoint(i, p3);

			// Convert 3D coordinates to 4-element vector
			double p4[4] = {p3[0], p3[1], p3[2], 1.0};

			// Apply the transformation matrix of the mask image. This maps the point
			// to the general world space shown in the 3D view.

			this->MMask->MultiplyPoint(p4, p4);

			// Next, apply the inverse matrix of the source image. This maps the
			// transformed point to the image space of the source image.

			invMSource->MultiplyPoint(p4, p4);

			// Convert the coordinates back to a 3D vector
			p3[0] = p4[0];
			p3[1] = p4[1];
			p3[2] = p4[2];

			// Variables needed for "FindCell"
			int subId = 0;
			double pCoords[3];
			double weights[8];

			// Find the cell containing the transformed point (and the interpolation weights)
			int cellId = sourceImage->FindCell(p3, NULL, 0, 0.001, subId, pCoords, weights);

			// If the transformed point is not located in the source image,
			// set the output value to zero.

			if (cellId == -1)
			{
				outScalars->SetTuple1(i, 0.0);
				continue;
			}

			// Get the cell containing the transformed point
			vtkCell * currentCell = sourceImage->GetCell(cellId);

			// Double-check if the cell exists...
			if (currentCell == NULL)
			{
				outScalars->SetTuple1(i, 0.0);
				continue;
			}

			// ...and if it contains eight points
			if (currentCell->GetNumberOfPoints() != 8)
			{
				outScalars->SetTuple1(i, 0.0);
				continue;
			}

			double iScalar = 0.0;

			// Interpolate the cell values at the current position
			for (int j = 0; j < 8; ++j)
			{
				double sourceValue = sourceScalars->GetTuple1(currentCell->GetPointId(j));
				iScalar += weights[j] * sourceValue;
			}

			// Add the interpolated scalar to the output
			outScalars->SetTuple1(i, iScalar);
		}

		// Delete the inverted matrix
		invMSource->Delete();
	}

	// Add the scalar array to the output
	outPD->SetScalars(outScalars);
	outScalars->Delete();
}


} // namespace bmia
