#ifndef __DTIVectorSlice_h
#define __DTIVectorSlice_h

#include "DTISlice.h"
#include "gsl/gsl_linalg.h"

class DTIVectorSlice : public DTISlice
{
public:

	DTIVectorSlice ();
	virtual ~DTIVectorSlice ();
	virtual void SetVectorAt ( gsl_matrix * vector, int row, int column );
	virtual gsl_matrix * GetVectorAt ( int row, int column );

protected:

	gsl_matrix ** Vectors;
};

#endif
