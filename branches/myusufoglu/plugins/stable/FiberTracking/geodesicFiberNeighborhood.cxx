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
 * geodesicFiberNeighborhood.cxx
 *
 * 2011-05-25	Evert van Aart
 * - First version. 
 *
 */


/** Includes */

#include "geodesicFiberNeighborhood.h"


namespace bmia {


int geodesicFiberNeighborhood::MOBILITY_NUMBER_OF_STEPS = 1000;


//-----------------------------[ Constructor ]-----------------------------\\

geodesicFiberNeighborhood::geodesicFiberNeighborhood()
{
	// Set pointers to NULL
	this->dtiImage		= NULL;
	this->scalarArray	= NULL;
	this->PreProcessor  = NULL;

	// Reset image data to zero
	for (int i = 0; i < 3; ++i)
	{
		this->offset[i]			= 0;
		this->ijk[i]			= 0;
		this->spacing[i]		= 0.0;
		this->extents[2*i]		= 0.0;
		this->extents[2*i+1]	= 0.0;
	}

	// Clear data of the full neighborhood
	for (int i = 0; i < 64; ++i)
	{
		this->scalars[i]  = 0.0;
		this->pointIds[i] = -1;

		for (int j = 0; j < 6; ++j)
		{
			this->ppTensors[i][j]  = 0.0;
			this->invTensors[i][j] = 0.0;
		}
	}

	// Clear the derivatives
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 6; ++j)
		{
			this->du[i][j] = 0.0;
			this->dv[i][j] = 0.0;
			this->dw[i][j] = 0.0;
		}
	}
}


//------------------------------[ Destructor ]-----------------------------\\

geodesicFiberNeighborhood::~geodesicFiberNeighborhood()
{
	// Clear the list of movement vectors
	this->moveList.clear();
}


//-----------------------------[ setDTIImage ]-----------------------------\\

void geodesicFiberNeighborhood::setDTIImage(vtkImageData * rImage)
{
	// Set the image pointer
	this->dtiImage = rImage;

	// Get the spacing and extent of the image
	this->dtiImage->GetSpacing(this->spacing);
	this->dtiImage->GetExtent(this->extents);
}


//----------------------------[ setScalarArray ]---------------------------\\

void geodesicFiberNeighborhood::setScalarArray(vtkImageData * scalarImage)
{
	// Reset the array pointer to NULL
	this->scalarArray = NULL;

	// Check the image pointer
	if (scalarImage == NULL)
		return;
	
	// Check if the image has point data
	if (!(scalarImage->GetPointData()))
		return;

	// Check if the image has a scalar array
	if (!(scalarImage->GetPointData()->GetScalars()))
		return;

	// Check if the scalar array has one component
	if (scalarImage->GetPointData()->GetScalars()->GetNumberOfComponents() != 1)
		return;

	// Success, store the scalar array pointer
	this->scalarArray = scalarImage->GetPointData()->GetScalars();
}


//------------------------[ initializeNeighborhood ]-----------------------\\

void geodesicFiberNeighborhood::initializeNeighborhood(double p[3])
{
	// Invalidate all neighborhood data
	this->invalidateData(0, 0);
	this->invalidateData(0, 1);
	this->invalidateData(0, 2);
	this->invalidateData(0, 3);

	// Reset the offsets
	this->offset[0] = 0;
	this->offset[1] = 0;
	this->offset[2] = 0;

	// Compute the indices of the lowest corner of the neighborhood
	this->ijk[0] = (int) floor(p[0] / this->spacing[0]) - 1;
	this->ijk[1] = (int) floor(p[1] / this->spacing[1]) - 1;
	this->ijk[2] = (int) floor(p[2] / this->spacing[2]) - 1;

	// Get the data for the neighborhood
	this->getNeighborhoodData();

	// Compute the 1D array indices corresponding to the three central points
	this->updateCenterIndices();

	// Compute the derivatives
	this->computeDerivatives();
}


//---------------------------------[ move ]--------------------------------\\

bool geodesicFiberNeighborhood::move(int dir[3])
{
	// Step size should not be larger than one in any direction
	if (dir[0] < -1 || dir[0] > 1 || dir[1] < -1 || dir[1] > 1 || dir[2] < -1 || dir[2] > 1)
		return false;

	// Add the step (movement vector) to the list
	MovementVector v;
	v.dir[0] = dir[0];
	v.dir[1] = dir[1];
	v.dir[2] = dir[2];
	this->moveList.append(v);

	// Loop through all three dimensions
	for (int i = 0; i < 3; ++i)
	{
		// If we do not have to move in this dimension, skip it
		if (dir[i] == 0)
			continue;

		// If we're moving in the positive direction, we invalidate the CURRENT slice.
		// In other words, slices 1, 2 and 3 keep their data and become slices 0,
		// 1 and 2, respectively, while slice 0 becomes invalid and turns into slice 3.

		if (dir[i] == 1)
			this->invalidateData(i, this->offset[i]);

		// Increment or decrement the offset
		this->offset[i] += dir[i];

		// Limit the offset to the range 0-3
		if (this->offset[i] ==  4)		this->offset[i] = 0;
		if (this->offset[i] == -1)		this->offset[i] = 3;

		// If we're moving in the negative direction, we invalid the NEW slice. 
		// In other words, slices 0, 1 and 2 keep their data and become slices 1,
		// 2 and 3, respectively, while slice 3 becomes invalid and turn into slice 0.

		if (dir[i] == -1)
			this->invalidateData(i, this->offset[i]);
	}

	// Increment or decrement the base indices
	this->ijk[0] += dir[0];
	this->ijk[1] += dir[1];
	this->ijk[2] += dir[2];

	// Fetch the new data for the invalidated points
	this->getNeighborhoodData();

	// Update the array indices for the eight central points
	this->updateCenterIndices();

	// Compute the derivatives for the eight central points
	this->computeDerivatives();

	return true;
}


//---------------------------[ computeMobility ]---------------------------\\

double geodesicFiberNeighborhood::computeMobility()
{
	// If the list is empty, we haven't changed cells in the past "MOBILITY_NUMBER_OF_STEPS"
	// steps, so we're stuck. Return 0.0, so that the tracker will stop execution.

	if (this->moveList.isEmpty())
	{
		return 0.0;
	}

	// Total movement vector
	double totalDir[3] = {0.0, 0.0, 0.0};
	
	// Compute the sum of all movement vectors
	for (QList<MovementVector>::iterator i = this->moveList.begin(); i != this->moveList.end(); ++i)
	{
		totalDir[0] += (double) (*i).dir[0];
		totalDir[1] += (double) (*i).dir[1];
		totalDir[2] += (double) (*i).dir[2];
	}

	// Clear the list of movement vectors
	this->moveList.clear();

	// Return the length of the sum vector
	return (vtkMath::Norm(totalDir));
}

//----------------------------[ invalidateData ]---------------------------\\

void geodesicFiberNeighborhood::invalidateData(int axis, int pos)
{
	// X-Axis: Invalidate every fourth point, starting at "pos"
	if (axis == 0)
	{
		for (int i = pos; i < 64; i += 4)
		{
			this->pointIds[i] = -1;
		}

		return;
	}

	// Y-Axis: Invalidate one block of four points per block of 16 points. The
	// "pos" index determines the position of the 4-point block.

	if (axis == 1)
	{
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				this->pointIds[i * 16 + pos * 4 + j] = -1;
			}
		}

		return;
	}

	// Z-axis: Invalidate one block of 16 points
	if (axis == 2)
	{
		for (int i = 0; i < 16; ++i)
		{
			this->pointIds[pos * 16 + i] = -1;
		}

		return;
	}
}


//-------------------------[ getNeighborhoodData ]-------------------------\\

void geodesicFiberNeighborhood::getNeighborhoodData()
{
	// Loop through all 64 points of the full neighborhood
	for (int di = 0; di < 4; ++di)	{
	for (int dj = 0; dj < 4; ++dj)	{
	for (int dk = 0; dk < 4; ++dk)	{

		// Compute the 1D index from the 3D indices (including offsets)
		int index = this->computeIndex(di, dj, dk);

		// Only do something if the current point was invalidated
		if (this->pointIds[index] == -1)
		{
			// Compute the index of the corresponding point in the image
			int ijkC[3] = {(this->ijk[0] + di), (this->ijk[1] + dj), (this->ijk[2] + dk)};

			// Limit the index to the extents
			if (ijkC[0] < this->extents[0])		ijkC[0] = this->extents[0];
			if (ijkC[0] > this->extents[1])		ijkC[0] = this->extents[1];
			if (ijkC[1] < this->extents[2])		ijkC[1] = this->extents[2];
			if (ijkC[1] > this->extents[3])		ijkC[1] = this->extents[3];
			if (ijkC[2] < this->extents[4])		ijkC[2] = this->extents[4];
			if (ijkC[2] > this->extents[5])		ijkC[2] = this->extents[5];

			// Compute the point coordinates, using the indices and the voxel spacing
			double p[3] = {ijkC[0] * this->spacing[0], ijkC[1] * this->spacing[1], ijkC[2] * this->spacing[2]};

			// Get the ID of the corresponding point
			this->pointIds[index] = this->dtiImage->FindPoint(p);

			// Pre-processed and inverted tensor
			double ppTensor[6]  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
			double invTensor[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

			// Fetch the pre-processed and inverted tensor for this voxel
			if (this->PreProcessor)
			{
				this->PreProcessor->preProcessSingleTensor(this->pointIds[index], ppTensor);
				this->PreProcessor->invertSingleTensor(this->pointIds[index], ppTensor, invTensor);
			}

			// Copy the tensors to the neighborhood arrays
			for (int i = 0; i < 6; ++i)
			{
				this->ppTensors[index][i]  = ppTensor[i];
				this->invTensors[index][i] = invTensor[i];
			}


			// If we've got a scalar array, get the scalar value for this point
			if (this->scalarArray)
				this->scalars[index] = this->scalarArray->GetTuple1(this->pointIds[index]);
			// Otherwise, just use zero
			else
				this->scalars[index] = 0.0;
		}

	} } } // for [3D indices]
}


//-------------------------[ updateCenterIndices ]-------------------------\\

void geodesicFiberNeighborhood::updateCenterIndices()
{
	// Compute the 1D indices for the central eight points of the 64 neighborhood points
	this->ci[0] = this->computeIndex(1, 1, 1);
	this->ci[1] = this->computeIndex(2, 1, 1);
	this->ci[2] = this->computeIndex(1, 2, 1);
	this->ci[3] = this->computeIndex(2, 2, 1);
	this->ci[4] = this->computeIndex(1, 1, 2);
	this->ci[5] = this->computeIndex(2, 1, 2);
	this->ci[6] = this->computeIndex(1, 2, 2);
	this->ci[7] = this->computeIndex(2, 2, 2);
}


//--------------------------[ computeDerivatives ]-------------------------\\

void geodesicFiberNeighborhood::computeDerivatives()
{
	// Avoid divide-by-zero errors
	if (this->spacing[0] == 0.0)	this->spacing[0] = 1.0;
	if (this->spacing[1] == 0.0)	this->spacing[1] = 1.0;
	if (this->spacing[2] == 0.0)	this->spacing[2] = 1.0;

	// For all six tensor elements
	for (int i = 0; i < 6; ++i)
	{
		this->du[0][i] = (this->invTensors[this->computeIndex(2, 1, 1)][i] - this->invTensors[this->computeIndex(0, 1, 1)][i]) / (2.0 * this->spacing[0]);
		this->du[1][i] = (this->invTensors[this->computeIndex(3, 1, 1)][i] - this->invTensors[this->computeIndex(1, 1, 1)][i]) / (2.0 * this->spacing[0]);
		this->du[2][i] = (this->invTensors[this->computeIndex(2, 2, 1)][i] - this->invTensors[this->computeIndex(0, 2, 1)][i]) / (2.0 * this->spacing[0]);
		this->du[3][i] = (this->invTensors[this->computeIndex(3, 2, 1)][i] - this->invTensors[this->computeIndex(1, 2, 1)][i]) / (2.0 * this->spacing[0]);
		this->du[4][i] = (this->invTensors[this->computeIndex(2, 1, 2)][i] - this->invTensors[this->computeIndex(0, 1, 2)][i]) / (2.0 * this->spacing[0]);
		this->du[5][i] = (this->invTensors[this->computeIndex(3, 1, 2)][i] - this->invTensors[this->computeIndex(1, 1, 2)][i]) / (2.0 * this->spacing[0]);
		this->du[6][i] = (this->invTensors[this->computeIndex(2, 2, 2)][i] - this->invTensors[this->computeIndex(0, 2, 2)][i]) / (2.0 * this->spacing[0]);
		this->du[7][i] = (this->invTensors[this->computeIndex(3, 2, 2)][i] - this->invTensors[this->computeIndex(1, 2, 2)][i]) / (2.0 * this->spacing[0]);

		this->dv[0][i] = (this->invTensors[this->computeIndex(1, 2, 1)][i] - this->invTensors[this->computeIndex(1, 0, 1)][i]) / (2.0 * this->spacing[1]);
		this->dv[1][i] = (this->invTensors[this->computeIndex(2, 2, 1)][i] - this->invTensors[this->computeIndex(2, 0, 1)][i]) / (2.0 * this->spacing[1]);
		this->dv[2][i] = (this->invTensors[this->computeIndex(1, 3, 1)][i] - this->invTensors[this->computeIndex(1, 1, 1)][i]) / (2.0 * this->spacing[1]);
		this->dv[3][i] = (this->invTensors[this->computeIndex(2, 3, 1)][i] - this->invTensors[this->computeIndex(2, 1, 1)][i]) / (2.0 * this->spacing[1]);
		this->dv[4][i] = (this->invTensors[this->computeIndex(1, 2, 2)][i] - this->invTensors[this->computeIndex(1, 0, 2)][i]) / (2.0 * this->spacing[1]);
		this->dv[5][i] = (this->invTensors[this->computeIndex(2, 2, 2)][i] - this->invTensors[this->computeIndex(2, 0, 2)][i]) / (2.0 * this->spacing[1]);
		this->dv[6][i] = (this->invTensors[this->computeIndex(1, 3, 2)][i] - this->invTensors[this->computeIndex(1, 1, 2)][i]) / (2.0 * this->spacing[1]);
		this->dv[7][i] = (this->invTensors[this->computeIndex(2, 3, 2)][i] - this->invTensors[this->computeIndex(2, 1, 2)][i]) / (2.0 * this->spacing[1]);

		this->dw[0][i] = (this->invTensors[this->computeIndex(1, 1, 2)][i] - this->invTensors[this->computeIndex(1, 1, 0)][i]) / (2.0 * this->spacing[2]);
		this->dw[1][i] = (this->invTensors[this->computeIndex(2, 1, 2)][i] - this->invTensors[this->computeIndex(2, 1, 0)][i]) / (2.0 * this->spacing[2]);
		this->dw[2][i] = (this->invTensors[this->computeIndex(1, 2, 2)][i] - this->invTensors[this->computeIndex(1, 2, 0)][i]) / (2.0 * this->spacing[2]);
		this->dw[3][i] = (this->invTensors[this->computeIndex(2, 2, 2)][i] - this->invTensors[this->computeIndex(2, 2, 0)][i]) / (2.0 * this->spacing[2]);
		this->dw[4][i] = (this->invTensors[this->computeIndex(1, 1, 3)][i] - this->invTensors[this->computeIndex(1, 1, 1)][i]) / (2.0 * this->spacing[2]);
		this->dw[5][i] = (this->invTensors[this->computeIndex(2, 1, 3)][i] - this->invTensors[this->computeIndex(2, 1, 1)][i]) / (2.0 * this->spacing[2]);
		this->dw[6][i] = (this->invTensors[this->computeIndex(1, 2, 3)][i] - this->invTensors[this->computeIndex(1, 2, 1)][i]) / (2.0 * this->spacing[2]);
		this->dw[7][i] = (this->invTensors[this->computeIndex(2, 2, 3)][i] - this->invTensors[this->computeIndex(2, 2, 1)][i]) / (2.0 * this->spacing[2]);
	}
}


//-----------------------------[ computeIndex ]----------------------------\\

int geodesicFiberNeighborhood::computeIndex(int i, int j, int k)
{
	// Add the index to the array offset, and limit it to the range 0-3
	int ri = (this->offset[0] + i) % 4;
	int rj = (this->offset[1] + j) % 4;
	int rk = (this->offset[2] + k) % 4;

	// Compute the 1D index from the wrapped 3D indices
	return ri + rj * 4 + rk * 16;
}


//--------------------------[ interpolateScalar ]--------------------------\\

double geodesicFiberNeighborhood::interpolateScalar(double * weights)
{
	// Output value
	double o = 0.0;

	// Add weighted scalar value of each of the eight central points
	for (int i = 0; i < 8; ++i)
	{
		o += weights[i] * this->scalars[this->ci[i]];
	}

	return o;
}


//----------------------[ computeChristoffelSymbols ]----------------------\\

void geodesicFiberNeighborhood::computeChristoffelSymbols(double * weights, double * symbols)
{
	// Interpolated tensors
	double Di[6]    = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double dGdui[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double dGdvi[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double dGdwi[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	// Interpolate the pre-processed tensors and the derivate tensors
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 6; ++j)
		{
			Di[j]	 += weights[i] * this->ppTensors[this->ci[i]][j];
			dGdui[j] += weights[i] * this->du[i][j];
			dGdvi[j] += weights[i] * this->dv[i][j];
			dGdwi[j] += weights[i] * this->dw[i][j];
		}
	}

	// Compute symbols
	symbols[G111] = 0.5 * Di[0] * dGdui[0] + 0.5 * Di[1] * (2 * dGdui[1] - dGdvi[0]) + 0.5 * Di[2] * (2 * dGdui[2] - dGdwi[0]);
	symbols[G112] = 0.5 * Di[0] * dGdvi[0] + 0.5 * Di[1] * dGdui[3] + 0.5 * Di[2] * (dGdui[4] + dGdvi[2] - dGdwi[1]);
	symbols[G122] = 0.5 * Di[0] * (2 * dGdvi[1] - dGdui[3]) + 0.5 * Di[1] * dGdvi[3] + 0.5 * Di[2] * (2 * dGdvi[4] - dGdwi[3]);
	symbols[G113] = 0.5 * Di[0] * dGdwi[0] + 0.5 * Di[1] * (dGdui[4] + dGdwi[1] - dGdvi[2]) + 0.5 * Di[2] * dGdui[5];
	symbols[G123] = 0.5 * Di[0] * (dGdvi[2] + dGdwi[1] - dGdui[4]) + 0.5 * Di[1] * dGdwi[3] + 0.5 * Di[2] * dGdvi[5];
	symbols[G133] = 0.5 * Di[0] * (2 * dGdwi[2] - dGdui[5]) + 0.5 * Di[1] * (2 * dGdwi[4] - dGdvi[5]) + 0.5 * Di[2] * dGdwi[5];

	symbols[G211] = 0.5 * Di[1] * dGdui[0] + 0.5 * Di[3] * (2 * dGdui[1] - dGdvi[0]) + 0.5 * Di[4] * (2 * dGdui[2] - dGdwi[0]);
	symbols[G212] = 0.5 * Di[1] * dGdvi[0] + 0.5 * Di[3] * dGdui[3] + 0.5 * Di[4] * (dGdui[4] + dGdvi[2] - dGdwi[1]);
	symbols[G222] = 0.5 * Di[1] * (2 * dGdvi[1] - dGdui[3]) + 0.5 * Di[3] * dGdvi[3] + 0.5 * Di[4] * (2 * dGdvi[4] - dGdwi[3]);
	symbols[G213] = 0.5 * Di[1] * dGdwi[0] + 0.5 * Di[3] * (dGdui[4] + dGdwi[1] - dGdvi[2]) + 0.5 * Di[4] * dGdui[5];
	symbols[G223] = 0.5 * Di[1] * (dGdvi[2] + dGdwi[1] - dGdui[4]) + 0.5 * Di[3] * dGdwi[3] + 0.5 * Di[4] * dGdvi[5];
	symbols[G233] = 0.5 * Di[1] * (2 * dGdwi[2] - dGdui[5]) + 0.5 * Di[3] * (2 * dGdwi[4] - dGdvi[5]) + 0.5 * Di[4] * dGdwi[5];

	symbols[G311] = 0.5 * Di[2] * dGdui[0] + 0.5 * Di[4] * (2 * dGdui[4] - dGdvi[0]) + 0.5 * Di[5] * (2 * dGdui[2] - dGdwi[0]);
	symbols[G312] = 0.5 * Di[2] * dGdvi[0] + 0.5 * Di[4] * dGdui[3] + 0.5 * Di[5] * (dGdui[4] + dGdvi[2] - dGdwi[1]);
	symbols[G322] = 0.5 * Di[2] * (2 * dGdvi[1] - dGdui[3]) + 0.5 * Di[4] * dGdvi[3] + 0.5 * Di[5] * (2 * dGdvi[4] - dGdwi[3]);
	symbols[G313] = 0.5 * Di[2] * dGdwi[0] + 0.5 * Di[4] * (dGdui[4] + dGdwi[1] - dGdvi[2]) + 0.5 * Di[5] * dGdui[5];
	symbols[G323] = 0.5 * Di[2] * (dGdvi[2] + dGdwi[1] - dGdui[4]) + 0.5 * Di[4] * dGdwi[3] + 0.5 * Di[5] * dGdvi[5];
	symbols[G333] = 0.5 * Di[2] * (2 * dGdwi[2] - dGdui[5]) + 0.5 * Di[4] * (2 * dGdwi[4] - dGdvi[5]) + 0.5 * Di[5] * dGdwi[5];
}


} // namespace bmia
