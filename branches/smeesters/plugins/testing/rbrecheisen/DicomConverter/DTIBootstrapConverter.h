#ifndef __DTIBootstrapConverter_h
#define __DTIBootstrapConverter_h

#include "DTIData2TensorConverter2.h"
#include "DTISliceGroup.h"
#include "DTITensorSlice.h"
#include "DTIVectorSlice.h"

#include "gsl/gsl_linalg.h"
#include <vector>

class DTIBootstrapConverter : public DTIData2TensorConverter2
{
public:

	DTIBootstrapConverter ();
	virtual ~DTIBootstrapConverter ();

	void SetNumberOfBootstrapVolumes ( int nrvolumes );
	void SetStartIndex( int index )
	{
		StartIndex = (index < 0) ? 0 : index;
	};
	void SetPixelSpacing ( double x, double y );
	void SetSliceThickness ( double slicethickness );
	bool Execute ();
	void Write ( int index, std::vector<DTITensorSlice *> * tensorslices );

protected:

	void ComputeADCAndResidualVectors ( int i, int j, DTISliceGroup * slicegroup, unsigned short * b0pixels,
		gsl_matrix * gradients, gsl_matrix * gradientsinverse, gsl_matrix *& adcvector, gsl_matrix *& residualvector );
	gsl_matrix * ComputeRandomTensor ( gsl_matrix * adcvector, gsl_matrix * residualvector, gsl_matrix * gradientsinverse );

	std::vector<DTIVectorSlice *> * ADCVectorSlices;
	std::vector<DTIVectorSlice *> * ResidualVectorSlices;
	int NumberOfBootstrapVolumes;
	double PixelSpacing[2];
	double SliceThickness;
	int StartIndex;
};

#endif
