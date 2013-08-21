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
* vtkSphericalHarmonicsToODFMaxVolumeFilter.cxx
*
* 2011-04-29	Evert van Aart
* - First version.
*
* 2011-05-04	Evert van Aart
* - Added the volume measure.
*
* 2011-08-05	Evert van Aart
* - Fixed an error in the computation of the unit vectors.
*
*/


/** Includes */

#include "vtkSphericalHarmonicsToODFMaxVolumeFilter.h"


namespace bmia {


	vtkStandardNewMacro(vtkSphericalHarmonicsToODFMaxVolumeFilter);


	//-----------------------------[ Constructor ]-----------------------------\\

	vtkSphericalHarmonicsToODFMaxVolumeFilter::vtkSphericalHarmonicsToODFMaxVolumeFilter()
	{
		// Initialize variables
		this->measure	= HARDIMeasures::GA;


		// Initialize the outputs
		//this->SetNumberOfOutputs(2);
		//this->SetNthOutput(0, NULL);
		// this->SetNthOutput(1, NULL);


		// Set pointers to NULL
		this->trianglesArray	= NULL;
		this->anglesArray		= NULL;
		this->radiiArray		= NULL;
		this->unitVectors		= NULL;
	//this->regionList = NULL;

		// Set default parameter values
		this->currentMeasure	= DSPHM_SurfaceArea;
		this->progressStepSize	= 1;
	}


	//------------------------------[ Destructor ]-----------------------------\\

	vtkSphericalHarmonicsToODFMaxVolumeFilter::~vtkSphericalHarmonicsToODFMaxVolumeFilter()
	{
		if (!(this->anglesArray)) 
			return;

		// Delete the unit vector array
		if (this->unitVectors)
		{
			for (int i = 0; i < this->anglesArray->GetNumberOfTuples(); ++i)
			{
				delete[] (this->unitVectors[i]);
			}

			delete[] this->unitVectors;
			this->unitVectors = NULL;
		}
	}


	//-------------------------[ getShortMeasureName ]-------------------------\\

	QString vtkSphericalHarmonicsToODFMaxVolumeFilter::getShortMeasureName(int index)
	{
		//if (index < 0 || index >= HARDIMeasures::SHARM_NumberOfMeasures)
		//	return "ERROR";

		// Return the short name of the selected measure
		switch(index)
		{
			//case DSPHM_SurfaceArea:		return "Area";
			//case DSPHM_Volume:			return "Volume";
			//case DSPHM_Average:			return "Average";
		//case HARDIMeasures::GA  : return "GA"; 		// General Anisotropy
		//case	HARDIMeasures::V : return "Variance";				// Variance
		//case	HARDIMeasures::GFA : return "GFA";			// General Fractional Anisotropy
		//case	HARDIMeasures::FMI : return "FMI";			// Fractional Multi-Fiber Index
		//case	HARDIMeasures::R0 : return "R0";			// Rank 0
		//case	HARDIMeasures::R2 : return "R2";
		//case	HARDIMeasures::Ri : return "Ri";
		//case	HARDIMeasures::Iso : return "Iso";			// Isotropic component
		//case	HARDIMeasures::SE : return "SE";				// ShannonEntropy
		//case	HARDIMeasures::CRE : return "CRE";				// Cumulative Residual Entropy
		//case	HARDIMeasures::NM : return "NM";				// Number of Maxima


		default:					return "ERROR";



		}
	}


	//--------------------------[ getLongMeasureName ]-------------------------\\

	QString vtkSphericalHarmonicsToODFMaxVolumeFilter::getLongMeasureName(int index)
	{
		//if (index < 0 || index >= HARDIMeasures::SHARM_NumberOfMeasures)
		//	return "ERROR";

		//// Return the long name of the selected measure
		switch(index)
		{
		//case HARDIMeasures::GA: return "General Anisotropy"; 		// General Anisotropy
		//case	HARDIMeasures::V: return "Variance";				// Variance
		//case	HARDIMeasures::GFA: return "General Fractional Anisotropy";			// General Fractional Anisotropy
		//case	HARDIMeasures::FMI:	return "Fractional Multi-Fiber Index";
		//case	HARDIMeasures::R0:	return "Rank 0";
		//case	HARDIMeasures::R2:	return "Rank 2";
		//case	HARDIMeasures::Ri:	return "Rank i";
		//case	HARDIMeasures::Iso:	return "Isotropic component";
		//case	HARDIMeasures::SE:	return "ShannonEntropy";
		//case	HARDIMeasures::CRE:	return "Cumulative Residual Entropy";
		//case	HARDIMeasures::NM:	return "Number of Maxima";

		default:					return "ERROR";
		}
	}


	//----------------------------[ SimpleExecute ]----------------------------\\

	void vtkSphericalHarmonicsToODFMaxVolumeFilter::SimpleExecute(vtkImageData * input, vtkImageData * output)
	{
		//vector to store the Id's if the found maxima on the ODF
			std::vector<int> maxima;
			//vector to store the unit vectors of the found maxima
			std::vector<double *> outputlistwithunitvectors;
			//neede for search space reduction
			bool searchRegion;
			std::vector<int> regionList;
			std::vector<double*> anglesArray1;
			for(int i=0; i< this->anglesArray->GetNumberOfTuples(); i++)
			{
				anglesArray1.push_back(this->anglesArray->GetTuple(i)); // carefull about the size of each pointer 
			}

			std::vector<double> ODFlist;

			//create a maximumfinder
			MaximumFinder MaxFinder(trianglesArray); // what does this arr do
		 
		// Start reporting the progress of this filter
		this->UpdateProgress(0.0);

		if (!input)
		{
			vtkErrorMacro(<<"No input has been set!");
			return;
		}

		if (!output)
		{
			vtkErrorMacro(<<"No output has been set!");
			return;
		}


		// Get the point data of the input
		vtkPointData * inputPD = input->GetPointData();

		// Check if the input has been set
		if (!inputPD)
		{
			vtkErrorMacro(<< "Input SH image does not contain point data!");
			return;
		}

		// Get the spherical harmonics coefficients
		vtkDataArray * SHCoefficientsArray = inputPD->GetScalars();

		// Check if the input has been set
		if (!SHCoefficientsArray)
		{
			vtkErrorMacro(<< "Input SH image does not contain an array with SH coefficients!");
			return;
		}


		// Get the point data of the input
		vtkPointData * inPD = input->GetPointData();

		if (!inPD)
		{
			vtkErrorMacro(<<"Input does not contain point data!");
			return;
		}

		// Get the point data of the output
		vtkPointData * outPD = output->GetPointData();

		if (!outPD)
		{
			vtkErrorMacro(<<"Output does not contain point data!");
			return;
		}

		// Get the number of points in the input image
		int numberOfPoints = input->GetNumberOfPoints();

		if (numberOfPoints < 1)
		{
			vtkWarningMacro(<<"Number of points in the input is not positive!");
			return;
		}


		// Set the dimensions of the output
		int dims[3];
		input->GetDimensions(dims);
		output->SetDimensions(dims);



		// Compute the step size for the progress bar
		this->progressStepSize = numberOfPoints / 25;
		this->progressStepSize += (this->progressStepSize == 0) ? 1 : 0;

		// Set the progress bar text
		this->SetProgressText("Computing scalar measure for spherical harmonics...");




		// Define output scalar array
		vtkIntArray * outArray = vtkIntArray::New(); // can keep indexes of maxes !!! What about there are angles in between then it can keep angles???
		outArray->SetNumberOfComponents(1);
		outArray->SetNumberOfTuples(numberOfPoints);

		// ID of the current point
		vtkIdType ptId;
		int size=SHCoefficientsArray->GetNumberOfComponents();
		// shOrder
		int l = 4 ;
			switch( SHCoefficientsArray->GetNumberOfComponents())
			{
			case 1:		l = 0;	break; // shOrder
			case 6:		l = 2;	break;
			case 15:	l = 4;	break;
			case 28:	l = 6;	break;
			case 45:	l = 8;	break;

			default:
				vtkErrorMacro(<< "Number of SH coefficients is not supported!");
				return;
			}
		// Current coefficients
		double *tempSH = new double[size] ; // size of l
		if(regionList.size()==0)
			for(int i=0;i< anglesArray1.size(); i++)
					regionList.push_back(i);
		// Loop through all points of the image
		for(ptId = 0; ptId < numberOfPoints; ++ptId)
		{
			// Get tensor value at current point
			SHCoefficientsArray->GetTuple(ptId, tempSH);

				//get maxima // correct angles array


			if(this->nMaximaForEachPoint == 1)
			{
				int indexOfMax;
				MaxFinder.getUniqueOutput(tempSH, l,this->treshold,  anglesArray1,  regionList,indexOfMax);
						double unitVector[3];
			 
			outArray->SetTuple1(ptId, indexOfMax);
			}
			else if (this->nMaximaForEachPoint > 1)
			{
			MaxFinder.getOutput(tempSH, l,this->treshold,  anglesArray1,  maxima, regionList);// SHAux is empty now we will give 8 differen , radiusun buyuk oldugu yerdeki angellari dizer donen 

			//Below 3 necessary?
			outputlistwithunitvectors.clear();
			//remove repeated maxima
			MaxFinder.cleanOutput(maxima, outputlistwithunitvectors,tempSH, ODFlist, this->unitVectors, anglesArray1);

			}
			else 
				cout << "this->nMaximaForEachPoint is not in the range" << endl; 

			
			//HARDIMeasures * HMeasures = new HARDIMeasures; // THIS will be used 

			// Check if tensor is NULL. This check is not necessary but can
			// save time on sparse datasets
			//if (vtkTensorMath::IsNullTensor(tensor))
			//{
			// Set output value to zero
			//     outArray->SetTuple1(ptId, 0.0);
			//}

			// Compute the output scalar value
			//else
			//{
			double unitVector[3];
			outArray->SetTuple3(ptId, unitVector[0], unitVector[1], unitVector[2]);
			// DO FOR ALL MEASURES ACCORDING TO GIVEN MEASURE
			//switch (this->currentMeasure)
		//	{
			//case HARDIMeasures::GA:		outScalar = HMeasures->GeneralAnisotropy(tempSH,l);	break; //  for all points do GA!!!
			//case HARDIMeasures::V:		outScalar = HMeasures->Variance(tempSH,l);			break;
			//case HARDIMeasures::GFA:    outScalar = HMeasures->GeneralFractionalAnisotropy(tempSH,l);	break;
			//case	HARDIMeasures::FMI:	outScalar = HMeasures->FractionalMultifiberIndex(tempSH,l); break; // "Fractional Multi-Fiber Index";
			//case	HARDIMeasures::R0:	outScalar = HMeasures->Rank0(tempSH,l); break; // 
			//case	HARDIMeasures::R2:	outScalar = HMeasures->Rank2(tempSH,l); break; // 
			//case	HARDIMeasures::Ri:	outScalar = HMeasures->RankI(tempSH,l); break; // 
			//case	HARDIMeasures::Iso:	outScalar = HMeasures->IsotropicComponent(tempSH,l); break; // 
			//case	HARDIMeasures::SE:	outScalar = HMeasures->ShannonEntropy(tempSH,l); break; // 
			//case	HARDIMeasures::CRE:	outScalar = HMeasures->CummulativeResidualEntropy(tempSH,l); break; // 
			//case	HARDIMeasures::NM:	outScalar = HMeasures->NumberMaxima(tempSH,l); break; // 
			//default:
			//	vtkErrorMacro(<<"Unknown scalar measure!");
			//	return;
			//}
			// Add scalar value to output array
			
			//}
			//
			// Update progress value
			if(ptId % 50000 == 0)
			{
				this->UpdateProgress(((float) ptId) / ((float) numberOfPoints));
			}
		}

		// Add scalars to the output
		outPD->SetScalars(outArray);

		outArray->Delete();
		outArray = NULL;

		// We're done!
		this->UpdateProgress(1.0);


		//	this->computeSHARMMeasureScalarVolume();


		// Add the scalar array to the output image
		//outPD->SetScalars(outArray);
		//outArray->Delete();
	}

	// Not used
	void vtkSphericalHarmonicsToODFMaxVolumeFilter::computeSHARMMeasureScalarVolume()
	{

		// Compute the desired measure
		switch (this->currentMeasure)
		{
			//	case HARDIMeasures::GA:		this->computeSurfaceArea(outArray);		break; //  for all points do GA!!!
			//	case HARDIMeasures::V:			this->computeVolume(outArray);			break;
			//	case HARDIMeasures::GFA:			this->computeAverageRadius(outArray);	break;

		default:
			vtkErrorMacro(<<"Unknown scalar measure!");
			return;
		}
	}

	//--------------------------[ computeUnitVectors ]-------------------------\\

	bool vtkSphericalHarmonicsToODFMaxVolumeFilter::computeUnitVectors() // for DS not SHARM
	{
		if (this->trianglesArray == NULL || this->anglesArray == NULL)
			return false;

		int numberOfTriangles = this->trianglesArray->GetNumberOfTuples();
		int numberOfAngles = this->anglesArray->GetNumberOfTuples();

		// Delete the unit vector array
		if (this->unitVectors)
		{
			for (int i = 0; i < numberOfAngles; ++i)
			{
				delete[] (this->unitVectors[i]);
			}

			delete[] this->unitVectors;
			this->unitVectors = NULL;
		}

		// Allocate new array for the unit vectors
		this->unitVectors = new double*[numberOfAngles];

		// Loop through all angles
		for (int i = 0; i < numberOfAngles; ++i)
		{
			this->unitVectors[i] = new double[3];

			// Get the two angles (azimuth and zenith)
			double * angles = anglesArray->GetTuple2(i);

			// Compute the 3D coordinates for these angles on the unit sphere
			this->unitVectors[i][0] = sinf(angles[0]) * cosf(angles[1]);
			this->unitVectors[i][1] = sinf(angles[0]) * sinf(angles[1]);
			this->unitVectors[i][2] = cosf(angles[0]);
		}

		return true;
	}

	 

	//--------------------------[ computeSurfaceArea ]-------------------------\\

	bool vtkSphericalHarmonicsToODFMaxVolumeFilter::computeSurfaceArea(vtkDoubleArray * outArray)
	{
		// First, compute the unit vectors
		if (!(this->computeUnitVectors()))
		{
			vtkErrorMacro(<<"Error computing unit vectors!");
			return false;
		}

		// Get the properties of the input image
		int numberOfTriangles = this->trianglesArray->GetNumberOfTuples();
		int numberOfAngles = this->anglesArray->GetNumberOfTuples();
		int numberOfPoints = this->radiiArray->GetNumberOfTuples();

		double d1[3];
		double d2[3];
		int T[3];

		// Loop through all voxels
		for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
		{
			// Get the vector of radii for the current voxel
			double * R = this->radiiArray->GetTuple(ptId);

			double area = 0.0;

			// Loop through all triangles
			for (int triangleId = 0; triangleId < numberOfTriangles; ++triangleId)
			{
				// Get the current triangle
				this->trianglesArray->GetTupleValue(triangleId, T);

				// Compute the area of the triangle as "A = 0.5 * |(V1 - V0) x (V2 - V0)|"
				d1[0] = R[T[1]] * this->unitVectors[T[1]][0] - R[T[0]] * this->unitVectors[T[0]][0];
				d1[1] = R[T[1]] * this->unitVectors[T[1]][1] - R[T[0]] * this->unitVectors[T[0]][1];
				d1[2] = R[T[1]] * this->unitVectors[T[1]][2] - R[T[0]] * this->unitVectors[T[0]][2];

				d2[0] = R[T[2]] * this->unitVectors[T[2]][0] - R[T[0]] * this->unitVectors[T[0]][0];
				d2[1] = R[T[2]] * this->unitVectors[T[2]][1] - R[T[0]] * this->unitVectors[T[0]][1];
				d2[2] = R[T[2]] * this->unitVectors[T[2]][2] - R[T[0]] * this->unitVectors[T[0]][2];

				double cross[3];

				vtkMath::Cross(d1, d2, cross);

				area += 0.5 * vtkMath::Norm(cross);
			}

			// Add the scalar measure value to the output array
			outArray->SetTuple1(ptId, area);

			// Update the progress bar
			if ((ptId % this->progressStepSize) == 0)
			{
				this->UpdateProgress((double) ptId / (double) numberOfPoints);
			}
		}

		return true;
	}


	//----------------------------[ computeVolume ]----------------------------\\

	bool vtkSphericalHarmonicsToODFMaxVolumeFilter::computeVolume(vtkDoubleArray * outArray)
	{
		// First, compute the unit vectors
		if (!(this->computeUnitVectors()))
		{
			vtkErrorMacro(<<"Error computing unit vectors!");
			return false;
		}

		// Get the properties of the input image
		int numberOfTriangles = this->trianglesArray->GetNumberOfTuples();
		int numberOfAngles = this->anglesArray->GetNumberOfTuples();
		int numberOfPoints = this->radiiArray->GetNumberOfTuples();

		double a[3];
		double b[3];
		double c[3];
		double bxc[3];
		int T[3];

		// Loop through all voxels
		for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
		{
			// Get the vector of radii for the current voxel
			double * R = this->radiiArray->GetTuple(ptId);

			double volume = 0.0;

			// Loop through all triangles
			for (int triangleId = 0; triangleId < numberOfTriangles; ++triangleId)
			{
				// Get the current triangle
				this->trianglesArray->GetTupleValue(triangleId, T);

				a[0] = R[T[0]] * this->unitVectors[T[0]][0];
				a[1] = R[T[0]] * this->unitVectors[T[0]][1];
				a[2] = R[T[0]] * this->unitVectors[T[0]][2];

				b[0] = R[T[1]] * this->unitVectors[T[1]][0];
				b[1] = R[T[1]] * this->unitVectors[T[1]][1];
				b[2] = R[T[1]] * this->unitVectors[T[1]][2];

				c[0] = R[T[2]] * this->unitVectors[T[2]][0];
				c[1] = R[T[2]] * this->unitVectors[T[2]][1];
				c[2] = R[T[2]] * this->unitVectors[T[2]][2];

				// Compute the volume of the tetrahedron formed by the three triangle
				// points and the glyph center. Since the glyph center is defined as
				// {0, 0, 0}, we can use the formula "V = |a . (b x c)| / 6".

				vtkMath::Cross(b, c, bxc);
				volume += abs(vtkMath::Dot(a, bxc)) / 6.0;
			}

			// Add the scalar measure value to the output array
			outArray->SetTuple1(ptId, volume); 

			// Update the progress bar
			if ((ptId % this->progressStepSize) == 0)
			{
				this->UpdateProgress((double) ptId / (double) numberOfPoints);
			}
		}

		return true;
	}


	//-------------------------[ computeAverageRadius ]------------------------\\

	bool vtkSphericalHarmonicsToODFMaxVolumeFilter::computeAverageRadius(vtkDoubleArray * outArray)
	{
		// Get the properties of the input image
		int numberOfAngles = this->anglesArray->GetNumberOfTuples();
		int numberOfPoints = this->radiiArray->GetNumberOfTuples();

		// Loop through all voxels
		for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
		{
			// Get the vector of radii for the current voxel
			double * R = this->radiiArray->GetTuple(ptId);

			double avg = 0.0;

			// Compute the average radius
			for (int angleId = 0; angleId < numberOfAngles; ++angleId)
			{
				avg += R[angleId] / (double) numberOfAngles;
			}

			// Add the scalar measure value to the output array
			outArray->SetTuple1(ptId, avg);

			// Update the progress bar
			if ((ptId % this->progressStepSize) == 0)
			{
				this->UpdateProgress((double) ptId / (double) numberOfPoints);
			}
		}

		return true;
	}


} // namespace bmia
