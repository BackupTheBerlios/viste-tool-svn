#include "DTIVectorSlice.h"

///////////////////////////////////////////////////////////
DTIVectorSlice::DTIVectorSlice ()
{
	this->Vectors = NULL;
	this->Rows = 0;
	this->Columns = 0;
}

///////////////////////////////////////////////////////////
DTIVectorSlice::~DTIVectorSlice ()
{
	if ( this->Vectors != NULL )
	{
		for ( int i = 0; i < (this->Rows * this->Columns); i++)
		{
			gsl_matrix_free ( this->Vectors[i] );
		}
	}
}

///////////////////////////////////////////////////////////
void DTIVectorSlice::SetVectorAt ( gsl_matrix * vector, int row, int column )
{
	if ( this->Vectors == NULL )
	{
		this->Vectors = new gsl_matrix * [this->Rows * this->Columns];
	}

	this->Vectors[row * this->Columns + column] = vector;
}

///////////////////////////////////////////////////////////
gsl_matrix * DTIVectorSlice::GetVectorAt ( int row, int column )
{
	return this->Vectors[row * this->Columns + column];
}
