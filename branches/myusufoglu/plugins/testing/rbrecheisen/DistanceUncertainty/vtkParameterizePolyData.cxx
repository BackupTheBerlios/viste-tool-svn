#include <vtkParameterizePolyData.h>

#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkDataArray.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>

vtkCxxRevisionMacro( vtkParameterizePolyData, "$Revision: 1.0 $" );
vtkStandardNewMacro( vtkParameterizePolyData );

//////////////////////////////////////////////////////////////////////
vtkParameterizePolyData::vtkParameterizePolyData()
{
	this->Input = NULL;
	this->Centroid[0] = 0.0;
	this->Centroid[1] = 0.0;
	this->Centroid[2] = 0.0;
}

//////////////////////////////////////////////////////////////////////
vtkParameterizePolyData::~vtkParameterizePolyData()
{
}

//////////////////////////////////////////////////////////////////////
void vtkParameterizePolyData::Execute()
{
	if( this->Input == NULL ||
			this->Input->GetPoints() == NULL ||
			this->Input->GetLines() == NULL )
		return;

	vtkIdType nrPtIds = this->Input->GetNumberOfPoints();
	vtkPoints * pts = this->Input->GetPoints();

	vtkDoubleArray * texCoords = vtkDoubleArray::New();
	texCoords->SetNumberOfComponents( 2 );
	texCoords->SetNumberOfTuples( nrPtIds );
	texCoords->SetName( "Texture Coordinates" );

	this->Input->GetPointData()->SetTCoords( texCoords );

	for( int ptId = 0; ptId < nrPtIds; ++ptId )
	{
		double P[3];
		pts->GetPoint( ptId, P );

		double V[3];
		V[0] = P[0] - this->Centroid[0];
		V[1] = P[1] - this->Centroid[1];
		V[2] = P[2] - this->Centroid[2];
		this->NormalizeVector( V );

		double PI = 3.14159265358979323;

		double theta = atan2( V[1], V[0] );
		double phi = acos( V[2] );

		texCoords->SetTuple2( ptId, 0.5 * theta / PI, phi / PI );
	}
}

//////////////////////////////////////////////////////////////////////
void vtkParameterizePolyData::SetCentroid( double centroid[3] )
{
	this->Centroid[0] = centroid[0];
	this->Centroid[1] = centroid[1];
	this->Centroid[2] = centroid[2];
}

//////////////////////////////////////////////////////////////////////
double vtkParameterizePolyData::GetCentroidX()
{
	return this->Centroid[0];
}

//////////////////////////////////////////////////////////////////////
double vtkParameterizePolyData::GetCentroidY()
{
	return this->Centroid[1];
}

//////////////////////////////////////////////////////////////////////
double vtkParameterizePolyData::GetCentroidZ()
{
	return this->Centroid[2];
}

//////////////////////////////////////////////////////////////////////
void vtkParameterizePolyData::SetInput( vtkPolyData * input )
{
	this->Input = input;
}

//////////////////////////////////////////////////////////////////////
vtkPolyData * vtkParameterizePolyData::GetInput()
{
	return this->Input;
}

//////////////////////////////////////////////////////////////////////
vtkPolyData * vtkParameterizePolyData::GetOutput()
{
	return this->Input;
}

///////////////////////////////////////////////////////////////////////////
void vtkParameterizePolyData::NormalizeVector( double V[3] )
{
	double length = sqrt( V[0]*V[0] + V[1]*V[1] + V[2]*V[2] );
	V[0] /= length;
	V[1] /= length;
	V[2] /= length;
}
