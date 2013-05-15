/**
 * vtkDistanceMeasureClosestPointDistance.cxx
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#include "vtkDistanceMeasureClosestPointDistance.h"
#include "vtkObjectFactory.h"

#include <algorithm>

namespace bmia {

	vtkStandardNewMacro( vtkDistanceMeasureClosestPointDistance );

	////////////////////////////////////////////////////////////////////////
	vtkDistanceMeasureClosestPointDistance::vtkDistanceMeasureClosestPointDistance() : 
		vtkDistanceMeasure()
	{
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceMeasureClosestPointDistance::~vtkDistanceMeasureClosestPointDistance()
	{
	}

	////////////////////////////////////////////////////////////////////////
	double vtkDistanceMeasureClosestPointDistance::ComputeSpecific( 
		vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 )
	{
		double distance = VTK_DOUBLE_MAX;
		
		for( int i = 0; i < _nrPointIds1; i++ )
		{
			// WARNING: Do not use the return pointer of GetPoint() because it somehow gives
			// corrupt data resulting in distances that are always zero.
			double point[3];
			this->GetPoints()->GetPoint( _pointIds1[i], point );

			double d = this->GetClosestPointDistance( point, _pointIds2, _nrPointIds2 );

			if( d < distance )
				distance = d;
		}

		return distance;
	}
} // namespace bmia
