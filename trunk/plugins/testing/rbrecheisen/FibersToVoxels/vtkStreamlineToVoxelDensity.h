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

#ifndef __vtkStreamlineToVoxelDensity_h
#define __vtkStreamlineToVoxelDensity_h

#include "vtkObject.h"
#include "vtkPolyData.h"
#include "vtkImageData.h"

class vtkStreamlineToVoxelDensity : public vtkObject
{
public:

	vtkStreamlineToVoxelDensity();
	virtual ~vtkStreamlineToVoxelDensity();
	
	void SetInput(vtkPolyData *data);
	
	void SetBinary( int binary )
		{ m_Binary = binary; }
	void SetDimensions(int dimX, int dimY, int dimZ);
	void SetSpacing(double dX, double dY, double dZ);
    void SetScores( double * scores, int count );
	
	vtkImageData *GetOutput();
    vtkImageData *GetOutput1();
	
	double GetMeanVoxelDensity()
		{ return m_MeanVoxelDensity; }
	double GetStandardDeviationVoxelDensity()
		{ return m_StandardDeviationVoxelDensity; }
	double GetTractVolume()
		{ return m_TractVolume; }
	
private:

    int NextPowerOfTwo(int N);

	vtkPolyData *m_Streamlines;
	
	int m_Binary;
	int m_Dimensions[3];
	double m_Spacing[3];

    double * m_Scores;
	
	double m_TractVolume;
	double m_MeanVoxelDensity;
	double m_StandardDeviationVoxelDensity;
};

#endif
