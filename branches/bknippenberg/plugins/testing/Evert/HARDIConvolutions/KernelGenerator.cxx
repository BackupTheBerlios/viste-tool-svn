/*
 * KernelGenerator.cxx
 *
 * 2011-08-02	Evert van Aart
 * - First version. 
 * 
 */


/** Includes */

#include "KernelGenerator.h"


namespace bmia {


double KernelGenerator::CLOSE_TO_ZERO = 0.001;


vtkStandardNewMacro(KernelGenerator);


//-----------------------------[ Constructor ]-----------------------------\\

KernelGenerator::KernelGenerator()
{
	// Set default parameter values
	this->Type	= KT_Duits;
	this->D33	= 1;
	this->D44	= 0.04;
	this->T		= 1.4;
	this->Sigma = 1.0;
	this->Kappa = 10.0;
	this->NormalizeKernels = true;
	this->spacingHasBeenTransformed = false;

	// Default kernel size is 3x3x3
	this->Extent[0] = -1;
	this->Extent[1] =  1;
	this->Extent[2] = -1;
	this->Extent[3] =  1;
	this->Extent[4] = -1;
	this->Extent[5] =  1;
	this->Dim[0]	=  3;
	this->Dim[1]	=  3;
	this->Dim[2]	=  3;

	// Default spacing is (1, 1, 1)
	this->Spacing[0] = 1.0;
	this->Spacing[1] = 1.0;
	this->Spacing[2] = 1.0;

	// Set pointers to NULL
	this->directions			= NULL;
	this->fileNameList			= NULL;
	this->tM	= NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

KernelGenerator::~KernelGenerator()
{
	// Delete all directions, and clear and delete the list of directions
	for (std::vector<double *>::iterator i = this->directions->begin(); i != this->directions->end(); ++i)
	{
		delete (*i);
	}

	directions->clear();
	delete directions;
}


//---------------------------[ UpdateDimensions ]--------------------------\\

void KernelGenerator::UpdateDimensions()
{
	// Compute the dimensions from the extent
	this->Dim[0] = (this->Extent[1] - this->Extent[0]) + 1;
	this->Dim[1] = (this->Extent[3] - this->Extent[2]) + 1;
	this->Dim[2] = (this->Extent[5] - this->Extent[4]) + 1;
}


//----------------------------[ UpdateSpacing ]----------------------------\\

void KernelGenerator::UpdateSpacing()
{
	// Do nothing if we don't have a transformation matrix, or if we've already transformed the spacing
	if (this->tM == NULL || this->spacingHasBeenTransformed)
		return;

	// Copy the first, second and third column to 3D vectors
	double tx[3] = {this->tM->GetElement(0, 0), this->tM->GetElement(1, 0), this->tM->GetElement(2, 0)};
	double ty[3] = {this->tM->GetElement(0, 1), this->tM->GetElement(1, 1), this->tM->GetElement(2, 1)};
	double tz[3] = {this->tM->GetElement(0, 2), this->tM->GetElement(1, 2), this->tM->GetElement(2, 2)};

	// Multiply spacing by the length of the column vectors
	this->Spacing[0] *= vtkMath::Norm(tx);
	this->Spacing[1] *= vtkMath::Norm(ty);
	this->Spacing[2] *= vtkMath::Norm(tz);

	this->spacingHasBeenTransformed = true;
}


//--------------------------[ BuildKernelFamily ]--------------------------\\

bool KernelGenerator::BuildKernelFamily()
{
	double ijk[3] = {0.0, 0.0, 0.0};	// Position of the current voxel, seen from the kernel center
	double * firstDirection  = NULL;	// Pivot direction, one per kernel image
	double * secondDirection = NULL;	// Secondary direction
	double kernelValue = 0.0;			// Current kernel value (radius)
	double L1Norm = 0.0;				// L1 norm, used for normalization
	bool success = true;				// Whether or not everything went okay

	// Update the dimensions of the kernel, based on its extents
	this->UpdateDimensions();

	// Update the spacing if possible
	this->UpdateSpacing();

	// Clear existing file name list
	if (this->fileNameList)
	{
		this->fileNameList->clear();
		delete this->fileNameList;
	}

	// Create a new file name list
	this->fileNameList = new QStringList;

	// Get the kernel size and the number of directions
	unsigned int numberOfVoxels = this->Dim[0] * this->Dim[1] * this->Dim[2];
	unsigned int numberOfDirections = this->directions->size();

	// Create output array for the full kernel image
	double * outArray = new double[numberOfVoxels * numberOfDirections];

	// Create a progress dialog for the generator
	QProgressDialog progress("Generating Kernels", QString(), 0, numberOfDirections);
	progress.setWindowTitle("Kernel Generator");
	progress.setValue(0);
	progress.show();

	// We create one kernel image per direction ("firstDirection", also called the
	// pivot direction). One kernel image consists of a number of voxels (determined 
	// by "dims"), each with one radius value for each direction. This radius is 
	// computed based on the pivot direction, the direction of the radius itself,
	// and the position of the voxel in the kernel image.

	for(unsigned int firstDirectionID = 0; firstDirectionID < numberOfDirections; ++firstDirectionID)
	{
		// Update the progress bar
		progress.setValue(firstDirectionID);

		// Get the current direction
		firstDirection = this->directions->at(firstDirectionID);

		// Reset the L1 norm
		L1Norm = 0.0;

		// Loop through all voxels in the kernel
		for(int k = 0; k < this->Dim[2]; ++k)	{
			for(int j = 0; j < this->Dim[1]; ++j)	{
				for(int i = 0; i < this->Dim[0]; ++i)	{

					// Compute the position of the current voxel, relative to the kernel center
					ijk[0] = (double) (this->Extent[0] + i) * this->Spacing[0]; 
					ijk[1] = (double) (this->Extent[2] + j) * this->Spacing[1]; 
					ijk[2] = (double) (this->Extent[4] + k) * this->Spacing[2];

					// Base index for this voxel
					int baseIndex = i + j * this->Dim[1] + k * (this->Dim[0] * this->Dim[2]);

					// Loop through all directions again
					for(unsigned int secondDirectionID = 0; secondDirectionID < numberOfDirections; ++secondDirectionID)
					{
						// Get the current direction
						secondDirection = this->directions->at(secondDirectionID);

						// Compute the value of the kernel, based on the position and the two directions
						switch (this->Type)
						{
							case KT_Duits:

								kernelValue = this->ComputeKernelValueDuits(ijk, firstDirection, secondDirection);
								break;

							case KT_Barmpoutis:

								kernelValue = this->ComputeKernelValueBarmpoutis(ijk, firstDirection, secondDirection);
								break;

							default:

								kernelValue = 0.0;
								break;
						}

						// If we want to normalize the kernels, first compute the sum of all radii for one voxel
						if (this->NormalizeKernels)
							L1Norm += kernelValue;

						// Store the kernel value
						outArray[secondDirectionID * numberOfVoxels + baseIndex] = kernelValue;
					}

				}	}	}

		// Normalize the kernel values if desired
		if (this->NormalizeKernels && fabs(L1Norm) > 0.0)
		{
			// Compute the final value for the L1 norm
			L1Norm *= ((4.0 / vtkMath::DoublePi()) / this->directions->size());

			// Divide all kernel radii by the L1 norm
			for (unsigned int valueID = 0; valueID < numberOfVoxels * numberOfDirections; ++valueID)
			{
				outArray[valueID] /= L1Norm;
			}
		}

		// Create a NIfTI writer
		KernelNIfTIWriter * writer = new KernelNIfTIWriter;

		// Write the kernel to the output
		writer->setDimensions(this->Dim);
		writer->setDirections(this->directions);
		writer->setFileName(absolutePath + baseName + QString::number(firstDirectionID) + QString(".nii"));
		writer->writeKernel(outArray);

		// Delete the writer
		delete writer;

		// Add the file name (without the path) to the list
		this->fileNameList->append(baseName + QString::number(firstDirectionID) + QString(".nii"));
	}

	// Finalize the progress bar
	progress.setValue(numberOfDirections);
	progress.hide();

	return success;
}


//--------------------------[ BuildSingleKernel ]--------------------------\\

void KernelGenerator::BuildSingleKernel(int firstDirectionID, double * outArray)
{
	double ijk[3] = {0.0, 0.0, 0.0};	// Position of the current voxel, seen from the kernel center
	double * firstDirection  = NULL;	// Pivot direction, one per kernel image
	double * secondDirection = NULL;	// Secondary direction
	double kernelValue = 0.0;			// Current kernel value (radius)
	double L1Norm = 0.0;				// L1 norm, used for normalization

	// Update the dimensions of the kernel, based on its extents
	this->UpdateDimensions();

	// Update the spacing if possible
	this->UpdateSpacing();

	// Get the kernel size and the number of directions
	unsigned int numberOfVoxels = this->Dim[0] * this->Dim[1] * this->Dim[2];
	unsigned int numberOfDirections = this->directions->size();

	// Get the current direction
	firstDirection = this->directions->at(firstDirectionID);

	// Reset the L1 norm
	L1Norm = 0.0;

	// Loop through all voxels in the kernel
	for(int k = 0; k < this->Dim[2]; ++k)	{
		for(int j = 0; j < this->Dim[1]; ++j)	{
			for(int i = 0; i < this->Dim[0]; ++i)	{

				// Compute the position of the current voxel, relative to the kernel center
				ijk[0] = (double) (this->Extent[0] + i) * this->Spacing[0]; 
				ijk[1] = (double) (this->Extent[2] + j) * this->Spacing[1]; 
				ijk[2] = (double) (this->Extent[4] + k) * this->Spacing[2];

				// Base index for this voxel
				int baseIndex = i + j * this->Dim[1] + k * (this->Dim[0] * this->Dim[2]);

				// Loop through all directions again
				for(unsigned int secondDirectionID = 0; secondDirectionID < numberOfDirections; ++secondDirectionID)
				{
					// Get the current direction
					secondDirection = this->directions->at(secondDirectionID);

					// Compute the value of the kernel, based on the position and the two directions
					switch (this->Type)
					{
						case KT_Duits:

							kernelValue = this->ComputeKernelValueDuits(ijk, firstDirection, secondDirection);
							break;

						case KT_Barmpoutis:

							kernelValue = this->ComputeKernelValueBarmpoutis(ijk, firstDirection, secondDirection);
							break;

						default:

							kernelValue = 0.0;
							break;
					}

					// If we want to normalize the kernels, first compute the sum of all radii for one voxel
					if (this->NormalizeKernels)
						L1Norm += kernelValue;

					// Store the kernel value
					outArray[secondDirectionID * numberOfVoxels + baseIndex] = kernelValue;
				}

			}	}	}

	// Normalize the kernel values if desired
	if (this->NormalizeKernels && fabs(L1Norm) > CLOSE_TO_ZERO)
	{
		// Compute the final value for the L1 norm
		L1Norm *= ((4.0 / vtkMath::DoublePi()) / this->directions->size());

		// Divide all kernel radii by the L1 norm
		for (unsigned int valueID = 0; valueID < numberOfVoxels * numberOfDirections; ++valueID)
		{
			outArray[valueID] /= L1Norm;
		}
	}
}


//--------------------------[ calculateKDistance ]-------------------------\\

double KernelGenerator::calculateKDistance(double * y, double sigma)
{
	double distance = sqrt(pow(y[0], 2) + pow(y[1], 2) + pow(y[2], 2));

	return (1.0 / pow(2.0 * vtkMath::DoublePi() * sigma, 1.5)) * exp(-pow(distance, 2) / (2.0 * pow(sigma, 3)));
}


//---------------------------[ calculateKFiber ]---------------------------\\

double KernelGenerator::calculateKFiber(double * y, double * r, double kappa)
{
	double distance = sqrt(pow(y[0], 2) + pow(y[1], 2) + pow(y[2], 2));

	double cosPhi;

	if(distance > 0.0)
		cosPhi = -(r[0] * y[0] + r[1] * y[1] + r[2] * y[2]) / distance;
	else
		cosPhi = 1.0;

	return (kappa * exp(kappa * cosPhi)) / (4.0 * vtkMath::DoublePi() * sinh(kappa));
}


//------------------------[ calculateKOrientation ]------------------------\\

double KernelGenerator::calculateKOrientation(double * r, double * v, double kappa)
{
	double cosPhi = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];

	return (kappa * exp(kappa * cosPhi)) / (4.0 * vtkMath::Pi() * sinh(kappa));
}


//-----------------------[ ComputeKernelValueDuits ]-----------------------\\

double KernelGenerator::ComputeKernelValueDuits(double * ijk, double * d1, double * d2)
{
	double result = 0.0;

	double * a  = this->RTnyforCPP(d2, ijk);
	double * vr = this->RTnyforCPP(d1, d1);
	double * b  = this->ConvertToEulerN(vr);

	double factor = (sqrt(this->D33 * this->D44) * sqrt(vtkMath::DoublePi() / 2.0) * 
		this->T * sqrt(this->D33 * this->T)) / 8.0;

	result = (double) factor * this->KernelSE3(a, b);

	delete[] a;
	delete[] vr;
	delete[] b;

	return result;
}


//---------------------[ ComputeKernelValueBarmpoutis ]--------------------\\

double KernelGenerator::ComputeKernelValueBarmpoutis(double * ijk, double * d1, double * d2)
{
	double KDistanceAndFiber = this->calculateKDistance(ijk, this->Sigma) * this->calculateKFiber(ijk, d1, this->Kappa);

	double result = KDistanceAndFiber * this->calculateKOrientation(d1, d2, this->Kappa);

	return result;
}


//------------------------------[ RTnyforCPP ]-----------------------------\\

double * KernelGenerator::RTnyforCPP(double * n, double * y)
{
	double * outV = new double[3];

	if (fabs(n[2] - 1.0) < CLOSE_TO_ZERO)
	{
		outV[0] = y[0];
		outV[1] = y[1];
		outV[2] = y[2];
	}
	else if (fabs(n[2] + 1.0) < CLOSE_TO_ZERO)
	{
		outV[0] =  y[0];
		outV[1] = -y[1];
		outV[2] = -y[2];
	}
	else
	{
		double a = (n[0] * n[2] * y[0] - sqrt(pow(n[0], 2) + pow(n[1], 2)) * 
			sqrt(1.0 - pow(n[2], 2)) * y[2]);
		double b =  pow(n[1], 2) * y[0] + n[0] * n[1] *
			(-1.0 + n[2]) * y[1] + n[0] * a;

		outV[0] = (1.0 / (pow(n[0], 2) + pow(n[1], 2))) * b;

		a = (n[1] * n[2] * y[1] - sqrt(pow(n[0], 2) + pow(n[1], 2)) *
			sqrt(1.0 - pow(n[2], 2)) * y[2]);
		b =  n[0] * n[1] * (-1.0 + n[2]) * y[0] + 
			pow(n[0], 2) * y[1] + n[1] * a;

		outV[1] = (1.0 / (pow(n[0], 2) + pow(n[1], 2))) * b;
		outV[2] = (sqrt(1.0 - pow(n[2], 2)) * (n[0] * y[0] + n[1] * y[1])) / 
			sqrt(pow(n[0], 2) + pow(n[1], 2)) + n[2] * y[2];
	}

	return outV;
}


//---------------------------[ ConvertToEulerN ]---------------------------\\

double * KernelGenerator::ConvertToEulerN(double * vx)
{
	double t = sqrt(vx[0] * vx[0] + vx[1] * vx[1] + vx[2] * vx[2]);

	double x[3];
	x[0] = (fabs(t) > CLOSE_TO_ZERO) ? (vx[0] / t) : vx[0];
	x[0] = (fabs(t) > CLOSE_TO_ZERO) ? (vx[1] / t) : vx[1];
	x[0] = (fabs(t) > CLOSE_TO_ZERO) ? (vx[2] / t) : vx[2];

	return this->ConvertToEuler(x);
}


//----------------------------[ ConvertToEuler ]---------------------------\\

double * KernelGenerator::ConvertToEuler(double * x)
{
	double * euler = new double[2];

	if(fabs(x[0] - 1.0) < CLOSE_TO_ZERO)
	{
		euler[0] = vtkMath::DoublePi();
		euler[1] = 0.0;
		return euler;
	}

	if(fabs(x[0] + 1.0) < CLOSE_TO_ZERO)
	{
		euler[0] = -vtkMath::DoublePi();
		euler[1] = 0.0;
		return euler;
	}

	double v = ((x[2] >= 0.0) ? 1.0 : -1.0) * sqrt(x[1] * x[1] + x[2] * x[2]);

	if (v >= 1.0) 
		euler[0] = 0.0;
	else if (v <= -1.0)
		euler[0] = ((x[1] > 0.0) ? 1.0 : -1.0) * vtkMath::DoublePi();
	else
		euler[0] = ((x[0] > 0.0) ? 1.0 : -1.0) * acos(v);

	euler[1] = ((x[2] >= 0.0) ? -1.0 : 1.0) * asin(x[1] / sqrt(x[1] * x[1] + x[2] * x[2]));

	return euler;
}


//--------------------------------[ exact ]--------------------------------\\

double KernelGenerator::exact(double ang)
{
	return (0.5 * ang) / tan(ang / 2.0);
}


//--------------------------------[ approx ]-------------------------------\\

double KernelGenerator::approx(double ang)
{
	return cos(ang / 2.0) / (1.0 - pow(ang, 2) / 24.0);
}


//------------------------------[ KernelSE2 ]------------------------------\\

double KernelGenerator::KernelSE2(double D11, double D22, double x, double y, double theta)
{
	double a;
	double b;

	if(fabs(theta) < (vtkMath::DoublePi() / 12.0))
	{
		a = (pow(theta, 2) / D11) + pow((((theta * y) / 2.0) + approx(theta) * x), 2) / D22;
		b =  pow(((-x * theta) / 2.0) + approx(theta) * y, 2) / (D11 * D22);
	}
	else
	{
		a = (pow(theta, 2) / D11) + pow((((theta * y) / 2.0) + exact(theta) * x), 2) / D22;
		b =  pow(((-x * theta) / 2.0) + exact(theta) * y, 2) / (D11 * D22);
	}	

	double d = 1.0 / (4.0 * vtkMath::DoublePi() * this->T * this->T * D22 * D11);
	double v = exp((-1.0 / (4.0 * this->T)) * sqrt(a * a + b));

	return (v * d);
}


//------------------------------[ KernelSE3 ]------------------------------\\

double KernelGenerator::KernelSE3(double x, double y, double z, double beta, double gamma)
{
	double a = this->KernelSE2(this->D44, this->D33, z / 2.0,  x,  beta);
	double b = this->KernelSE2(this->D44, this->D33, z / 2.0, -y, gamma);

	return a * b;
}


//------------------------------[ KernelSE3 ]------------------------------\\

double KernelGenerator::KernelSE3(double * x, double * r)
{
	return this->KernelSE3(x[0], x[1], x[2], r[0], r[1]);		
}


} // namespace bmia
