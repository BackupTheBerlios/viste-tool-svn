#ifndef __DTIData2HardiConverter_h
#define __DTIData2HardiConverter_h

#include "DTIUtils.h"
#include "DTISlice.h"
#include "DTISliceGroup.h"
#include "gsl/gsl_linalg.h"
#include <vector>
using namespace std;

//---------------------------------------------------------------------------
//! \file   DTIData2HardiConverter.h
//! \class  DTIData2HardiConverter
//! \author Ralph Brecheisen
//! \brief  Converts DTI slice data to Hardi direction volumes.
//---------------------------------------------------------------------------
class DTIData2HardiConverter
{
public:
	virtual void SetInput(vector<DTISliceGroup *> *input);
	virtual void SetGradients(gsl_matrix * gradients);
	virtual void SetDataType(char *type);
	virtual void SetVersion(int version);
	virtual void SetBValue(double value);
	virtual bool Execute();

	DTIData2HardiConverter();
	~DTIData2HardiConverter();

protected:
	vector<DTISliceGroup *> *Input;
	gsl_matrix *Gradients;

	int NumberOfGradients;
	int Rows;
	int Columns;
	int NumberOfSlices;
	int DataType;
	int Version;
	double BValue;
	double SliceThickness;
	double *PixelSpacing;
};

#endif
