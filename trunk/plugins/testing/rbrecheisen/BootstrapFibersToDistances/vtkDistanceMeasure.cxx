/**
 * vtkDistanceMeasure.cxx
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#include "vtkDistanceMeasure.h"
#include "vtkMath.h"

//namespace bmia {

	////////////////////////////////////////////////////////////////////////
	vtkDistanceMeasure::vtkDistanceMeasure()
	{
		this->Points = NULL;
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceMeasure::~vtkDistanceMeasure()
	{
		if( this->Points )
			this->Points->UnRegister( this );
	}

	////////////////////////////////////////////////////////////////////////
	void vtkDistanceMeasure::SetPoints( vtkPoints * _points )
	{
		if( this->Points )
			this->Points->UnRegister( this );
		this->Points = _points;
		if( this->Points )
			this->Points->Register( this );
	}

	////////////////////////////////////////////////////////////////////////
	vtkPoints * vtkDistanceMeasure::GetPoints()
	{
		return this->Points;
	}

    ////////////////////////////////////////////////////////////////////////
	double vtkDistanceMeasure::Compute(
		vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 )
	{
		// Check if we have points to compute our distances on. 
		// If not, crash mercilessly
		assert( this->GetPoints() );

		double distance = this->ComputeSpecific( _pointIds1, _nrPointIds1, _pointIds2, _nrPointIds2 );
		return distance;
	}

    ////////////////////////////////////////////////////////////////////////
	double vtkDistanceMeasure::GetClosestPointDistance( double * _point, vtkIdType * _pointIds, int _nrPointIds )
	{
		double distance = VTK_DOUBLE_MAX;

		for( int i = 0; i < _nrPointIds; i++ )
		{
			// WARNING: Do not use the return pointer of GetPoint() because it somehow gives
			// corrupt data resulting in distances that are always zero.
			double pt[3];
			this->GetPoints()->GetPoint( _pointIds[i], pt );

			double d = vtkMath::Distance2BetweenPoints( _point, pt );

			if( d < distance )
				distance = d;
		}
		
		return distance;
	}

//} // namespace bmia
