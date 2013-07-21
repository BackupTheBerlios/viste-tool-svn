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
 * HARDIMeasures.cxx
 *
 * 2008-10-22	Vesna Prckovska
 * - First version
 *
 * 2009-01-13	Tim Peeters
 * - Second version
 *
 * 2010-12-07	Evert van Aart
 * - Made compatible with DTITool3. 
 * - Removed dependence on GSL. Visualization of HARDI glyphs depends on this
 *   class, and we want basic visualization to be independent of GSL.
 *
 * 2010-12-16	Evert van Aart
 * - Removed the "Classify Voxel" measure, since it wasn't actually used.
 *
 */


/** Includes */

#include "HARDIMeasures.h"


namespace bmia {


const char * HARDIMeasures::longNames[] = 
{
	"General Anisotropy",
	"Variance",
	"General Fractional Anisotropy",
	"Fractional Multi-Fiber Index",
	"Rank 0",
	"Rank 2",
	"Rank i",
	"Isotropic component",
	"ShannonEntropy",
	"Cumulative Residual Entropy",
	"Number of Maxima"
}; 

const char * HARDIMeasures::shortNames[] = 
{
	"GA",
	"V",
	"GFA",
	"FMI",
	"R0",
	"R2",
	"Ri",
	"Iso",
	"SE",
	"CRE",
	"NM"
}; 


HARDIMeasures::HARDIMeasures()
{

}

HARDIMeasures::~HARDIMeasures()
{
	this->clearTessellation();
}

void HARDIMeasures::clearTessellation()
{
	// Iterator for the tessellation point list
	std::vector<double *>::iterator tessIter;

	// Loop through all tessellation points
	for (	tessIter  = this->tessellationPointsInSC.begin();
			tessIter != this->tessellationPointsInSC.end();
			++tessIter)
	{
		// Delete the coordinate array
		delete [] (*tessIter);
	}

	// Clear the list
	this->tessellationPointsInSC.clear();
}

double HARDIMeasures::HARDIMeasure(int measure, double * coeff, int l)
{
	// Measure should be between zero and the number of measures
	assert(measure >= 0 && measure < HARDIMeasures::numberOfMeasures);

	// Computed HARDI Measure
	double result;

	// Switch calls specific measure function, depending on "measure"
	switch (measure)
	{
		case V:
			result = HARDIMeasures::Variance(coeff, l);
			break;
		case GA:
			result = HARDIMeasures::GeneralAnisotropy(coeff, l);
			break;
		case GFA:
			result = HARDIMeasures::GeneralFractionalAnisotropy(coeff, l);
			break;
		case FMI:
			result = HARDIMeasures::FractionalMultifiberIndex(coeff, l);
			break;
		case R0:
			result = HARDIMeasures::Rank0(coeff, l);
			break;
		case R2:
			result = HARDIMeasures::Rank2(coeff, l); 
			break;
		case Ri:
			result = HARDIMeasures::RankI(coeff, l); 
			break;
		case Iso:
			result = HARDIMeasures::IsotropicComponent(coeff, l);
			break;
		case NM:
			result = HARDIMeasures::NumberMaxima(coeff, l);
			break;
		case SE:
			result = HARDIMeasures::ShannonEntropy(coeff, l);
			break;
		case CRE:
			result = HARDIMeasures::CummulativeResidualEntropy(coeff, l);
			break;
		default:
			// This should never happen. 
			result = 0.0;
			assert(false);
	} // switch [measure]
			
	return result;
}

const char * HARDIMeasures::GetLongName(int measure)
{
	// Measure should be between zero and the number of measures
	assert(measure >= 0 && measure < HARDIMeasures::numberOfMeasures);

	return longNames[measure];
}

const char * HARDIMeasures::GetShortName(int measure)
{
	// Measure should be between zero and the number of measures
	assert(measure >= 0 && measure < HARDIMeasures::numberOfMeasures);

	return shortNames[measure];
}


double HARDIMeasures::Variance(double * coeff, int l)
{
	// Do nothing if the first coefficient is zero
	if (coeff[0] == 0.0)
	{
		return 0.0;
	}

	// Resulting variance
	double result = 0.0;

	// Coefficient index
	int r = 1;

	// Compute the squared sum of all coefficients except for the first one
	for(int i = 2; i <= l; i += 2)
	{
		for(int j = -i; j <= i; j++, r++)
		{
			result += pow(coeff[r], 2);
		} 
		
	}

	// Divide by square of first coefficient times nine
	result /= 9 * pow(coeff[0], 2);

	return result;
}


double HARDIMeasures::GeneralFractionalAnisotropy(double * coeff, int l)
{
	// Squared sum of all coefficients
	double sum = 0;

	// Coefficient index
	int r = 0;

	// Compute the squared sum of all coefficients
	for (int i = 0; i <= l; i += 2)
	{
		for(int j = -i; j <= i; j++, r++)
		{
			sum += pow(coeff[r], 2);
		}
	}

	if (sum == 0)
	{
		return 0.0;
	} 

	// Compute output
	double result = sqrt(1.0 - pow(coeff[0], 2) / sum);

	return result;
}


double HARDIMeasures::GeneralAnisotropy(double * coeff, int l)
{
	// First, compute the variance
	double v = HARDIMeasures::Variance(coeff, l);

	double EV = 1 + 1 / (1 + 5000 * v);

	if ((1 + pow(250 * v, EV)) == 0)
	{
		return 0.0;
	}
	else
	{
		return (1 - 1 / (1 + pow(250 * v, EV)));
	}
}

double HARDIMeasures::FractionalMultifiberIndex(double * coeff, int l)
{
	// Numerator
	double num = 0.0;

	// Denominator
	double denom = 0.0;

	// Coefficient index
	int r = 1;

	// Loop through all coefficients, except for the first one
	for(int i = 2; i <= l; i += 2)
	{
		for(int j = -i; j <= i; j++, r++)
		{
			// Add second-order coefficients to the denominator
			if (i == 2)
			{
				denom += pow(coeff[r], 2);
			}
			// Add coefficients of other orders to the numerator
			else
			{
				num += pow(coeff[r], 2);
			}
		}
	}

	// Check if the denominator is zero	
	if (denom == 0)
	{
		return 0.0;
	} 
	// If not, return the fraction
	else
	{
		return num / denom;
	}
}

double HARDIMeasures::Rank0(double * coeff, int l)
{
	// Do nothing if the first coefficient is zero
	if (coeff[0] == 0.0)
	{
		return 0.0;
	}

	// Output
	double result = 0.0;
	
	// Coefficient index
	int r = 0;

	// Loop through all coefficients
	for(int i = 0; i <= l; i += 2)
	{
		for(int j = -i; j <= i; j++, r++)
		{
			// Compute the sum of absolute coefficient values
			result += fabs(coeff[r]);
		}
	}

	// Divide absolute first coefficient by coefficient sum
	result = fabs(coeff[0]) / result;

	return result;
}

double HARDIMeasures::Rank2(double * coeff, int l)
{
	// Numerator
	double num = 0.0;

	// Denominator
	double denom = 0.0;

	// Output
	double res = 0;

	// Coefficient index
	int r = 0;

	// Loop through all coefficients
	for(int i = 0; i <= l; i += 2)
	{
		for(int j = -i; j <= i; j++, r++)
		{
			// Add absolute coefficients of second order to numerator
			if (i == 2)
			{
				num += fabs(coeff[r]);
			}
			
			// Add absolute coefficients of all orders to denominator
			denom += fabs(coeff[r]);
		}
	}

	// Check if the denominator is zero
	if (denom == 0.0)
	{
		return 0.0;
	} 
	// Otherwise, return the fraction
	else
	{
		return num/denom;
	}
}


double HARDIMeasures::RankI(double * coeff, int l)
{
	// Numerator
	double num = 0.0;

	// Denominator
	double denom = 0.0;

	// Output
	double res = 0;

	// Coefficient index
	int r = 0;

	// Loop through all coefficients
	for(int i = 0; i <= l; i += 2)
	{
		for(int j = -i; j <= i; j++, r++)
		{
			// Add absolute coefficient of orders higher than four to numerator
			if (i >= 4)
			{
				num += fabs(coeff[r]);
			}

			// Add absolute coefficients of all orders to denominator
			denom += fabs(coeff[r]);
		}
	}

	// Check if the denominator is zero
	if (denom == 0.0)
	{
		return 0.0;
	} 
	// Otherwise, return the fraction
	else
	{
		return num/denom;
	}
}


double HARDIMeasures::IsotropicComponent(double * coeff, int l)
{
	// Simply return the first coefficient
	return coeff[0];
}

double HARDIMeasures::NumberMaxima(double * coeff, int l)
{
	double maxThreshhold = 0.5;
	int numMaxima = 4;

	// Compute tessellation if necessary
	if (this->tessellationPointsInSC.size() == 0)
	{
		this->getTessellation(4);
	}

	// Compute the deformator based on the SH coefficients, the tessellation, and the order of the SH coefficients
	std::vector<double> QBallDeformator = HARDITransformationManager::CalculateDeformator(coeff, &(this->tessellationPointsInSC), l);

	// Minimum and maximum for deformator
	double Minimum =  1000000.0;
	double Maximum = -1000000.0;

	// Get minimum and maximum of the deformator
	for (unsigned int i = 0; i < QBallDeformator.size(); ++i)
	{
		if (QBallDeformator[i] < Minimum)	Minimum = QBallDeformator[i];
		if (QBallDeformator[i] > Maximum)	Maximum = QBallDeformator[i];
	}

	assert (Maximum > Minimum);

	for (unsigned int i = 0; i < QBallDeformator.size(); ++i)
	{
		std::vector<double>::pointer deformatorValue = &(QBallDeformator[i]);
		(*deformatorValue) = (QBallDeformator[i] - Minimum) / (Maximum - Minimum);
	}

	Maximum = 1.0;

/* TODO: Re-enable once HARDISources has been added. 
			vtkPolyData * poly = vtkHARDISources::CreateDeformedSurface(deformator, sphere, Maximum, 1 );
			int * maximaList = vtkHARDISources::CalculateMaxima(poly, deformator, Maximum, MaxThreshhold, numMaxima, sphere);

			int actualMaxima = 0;
			for (int i=0; i<numMaxima;i++)
			{
				if (maximaList[i]!= -1)
					actualMaxima++;
			}

			double foundAngle = vtkHARDISources::MinimalAngle(maximaList, poly->GetPoints(),  numMaxima);

			return foundAngle;
*/

	return 0.0;
}


double HARDIMeasures::ShannonEntropy(double * coeff, int l)
{
	// Compute tessellation if necessary
	if (this->tessellationPointsInSC.size() == 0)
	{
		this->getTessellation();
	}

	// Compute the deformator based on the SH coefficients, the tessellation, and the order of the SH coefficients
	std::vector<double> QBallDeformator = HARDITransformationManager::CalculateDeformator(coeff, &(this->tessellationPointsInSC), l);

	// Check if the deformator contains elements
	assert(QBallDeformator.size() > 0);

	// Total sum of the deformator
	double deformatorSum = 0.0;

	// Loop through all values in the deformator
	for (unsigned int i = 0; i < QBallDeformator.size(); ++i)
	{
		std::vector<double>::pointer deformatorValue = &(QBallDeformator[i]);

		// Set negative values to zero to avoid negative logarithms
		if ((*deformatorValue) < 0)
		{
			(*deformatorValue) = 0;
		}

		// Increment sum of deformator
		deformatorSum += (*deformatorValue);
	}
	
	// Sum of the normalized deformator values multiplied by their logarithms
	double PLogPSum = 0.0;

	// Loop through all values in the deformator
	for (unsigned int i = 0; i < QBallDeformator.size(); ++i)
	{
		double deformatorValue = QBallDeformator[i];

		// Normalize the deformator value
		double deformatorValueNorm = deformatorValue / deformatorSum;

		// Compute the logarithm of the normalized value
		double deformatorValueNormLog = log(deformatorValueNorm);

		// Multiply normalized value by its logarithm, and add to the sum
		PLogPSum += deformatorValueNorm * deformatorValueNormLog;
	}

	// Compute entropy
	double entropy = -PLogPSum / (double) QBallDeformator.size();

	return entropy;
}


double HARDIMeasures::CummulativeResidualEntropy(double * coeff, int l)
{
	// Compute tessellation if necessary
	if (this->tessellationPointsInSC.size() == 0)
	{
		this->getTessellation();
	}

	// Compute the deformator based on the SH coefficients, the tessellation, and the order of the SH coefficients
	std::vector<double> QBallDeformator = HARDITransformationManager::CalculateDeformator(coeff, &(this->tessellationPointsInSC), l);

	// Check if the deformator contains elements
	assert(QBallDeformator.size() > 0);

	// Total sum of the deformator
	double deformatorSum = 0.0;

	// Loop through all values in the deformator
	for (unsigned int i = 0; i < QBallDeformator.size(); ++i)
	{
		std::vector<double>::pointer deformatorValue = &(QBallDeformator[i]);

		// Set negative values to zero to avoid negative logarithms
		if ((*deformatorValue) < 0)
		{
			(*deformatorValue) = 0;
		}

		// Increment sum of deformator
		deformatorSum += (*deformatorValue);
	}

	// Normalized deformator
	std::vector<double> QBallDeformatorNorm;

	// Minimum and maximum for deformator
	double Minimum =  1000000.0;
	double Maximum = -1000000.0;

	// Get minimum and maximum of the deformator
	for (unsigned int i = 0; i < QBallDeformator.size(); ++i)
	{
		// Get deformator value, normalize it
		double deformatorValue = QBallDeformator[i];
		double deformatorValueNorm = deformatorValue / deformatorSum;

		// Keep track of maximum and minimum
		if (deformatorValueNorm < Minimum)	Minimum = deformatorValueNorm;
		if (deformatorValueNorm > Maximum)	Maximum = deformatorValueNorm;

		// Add normalized value to the list
		QBallDeformatorNorm.push_back(deformatorValueNorm);
	}

	// Maximum should be higher than minimum
	assert (Maximum > Minimum);

	// Number of steps in range
	double M = 100.0;

	// Step size
	double delta = (Maximum - Minimum) / M;

	// Output value
	double cre = 0.0;

	// Increment "delta_i" from "Minimum" to "Maximum" in "M" steps
	for(double delta_i = Minimum; delta_i <= Maximum; delta_i += delta) 
	{
		// Get the probability
		double p = getProbability(&QBallDeformatorNorm, delta_i);

		// Increment entropy sum
		if(p > 0.0)
		{
			cre -= p * log(p) * delta;
		}
	}

	return cre;
}


double  HARDIMeasures::getProbability(std::vector<double> * deformator, double delta_i) 
{
	// Number of deformator elements higher than "delta_i"
	int p = 0;

	// Loop through all deformator elements
	for(unsigned int j = 0; j < deformator->size(); j++) 
	{
		// Increment "p" if the element is higher than "delta_i"
		if(deformator->at(j) > delta_i)
		{
			p++;
		}
	}

	// Divide by the number of elements to get the probability
	return (double) p / deformator->size();
}


/** TODO: We do not currently need this function. When we do, we should evaluate whether
	the HARDIMeasures class is the right place for it, and what the input arguments should be. 
double HARDIMeasures::(int ngrad, double ** gradDirrInSphericalCoordinates, double * ADCtrue, int lOld, int lCurrent, double * SHcoeffOld, double * SHcoeffCurrent)
{
	size_t ngrad = gradDirrInSphericalCoordinates.getNRow();

	int p1 = (lOld+1)*(lOld+2)/2;
	int p2 = (lCurr+1)*(lCurr+2)/2;

	VectorDouble ADCold = VectorDouble(ngrad);
	VectorDouble ADCcurrent = VectorDouble(ngrad);
	VectorDouble errOld = VectorDouble(ngrad);
	VectorDouble errCurrent = VectorDouble(ngrad);

	double meanErrOld =0;
	double meanErrCurrent=0;

	//variances
	double V1 = 0;
	double V2 = 0;
	double E = 0;

	double MinMax[2];

	ADCcurrent = HARDITransformationManager::CalculateDeformator(shcurr, gradDirrInSphericalCoordinates, lCurr);



	double Etemp = 0;
	for (int i=0;i<ngrad;i++)
	{
		Etemp+=pow(ADCcurrent(i)-ADCtrue(i),2);

	}
	
	E = Etemp/ngrad;

	for ( int i=1;i<p1;i++)
	{
		V1+=pow(shold(i),2);
		V2+=pow(shcurr(i),2);

	}
	for ( int i=p1;i<p2;i++)
	{
		V2+=pow(shcurr(i),2);


	}

	V1 = V1/(4*vtkMath::DoublePi());
	V2 = V2/(4*vtkMath::DoublePi());

	return ((ngrad - p2-1)*(V2-V1))/((p2-p1)*E);
} 
*/


void HARDIMeasures::getTessellation(int tessOrder)
{
	// Delete existing tessellation
	if (this->tessellationPointsInSC.size() != 0)
	{
		this->clearTessellation();
	}

	// Temporary tessellation point
	double p[3];

	// Set of tessellation points
	vtkPoints * points = NULL;

	// Create the sphere tessellator
	Visualization::sphereTesselator<double> * st = new Visualization::sphereTesselator<double>();

	// Create the output of the tessellator
	vtkPolyData * sphere = vtkPolyData::New();

	// Set the tessellation order
	st->tesselate(tessOrder);

	// Get the output
	st->getvtkTesselation(true, sphere);

	// Get the number of tessellation points
	int numberOfPoints = sphere->GetPoints()->GetNumberOfPoints();

	// Get the tessellation points
	points = sphere->GetPoints();

	// Loop through all points
	for (int i = 0; i < numberOfPoints; i++)
	{
		// Get the current point
		points->GetPoint(i, p);

		// Create a double array with two components
		double * sc = new double[2];

		// Compute the spherical coordinates of the point
		sc[0] = atan2((sqrt(pow(p[0], 2) + pow(p[1], 2))), p[2]);
		sc[1] = atan2(p[1], p[0]);

		// Add the spherical coordinates to the list
		this->tessellationPointsInSC.push_back(sc);
	}

	// Get rid of the tessellator
	delete st;
}


} // namespace bmia
