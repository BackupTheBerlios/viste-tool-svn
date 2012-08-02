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
