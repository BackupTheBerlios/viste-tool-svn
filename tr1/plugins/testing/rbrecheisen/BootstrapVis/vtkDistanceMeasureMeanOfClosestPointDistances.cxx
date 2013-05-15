/**
 * vtkDistanceMeasureMeanOfClosestPointDistances.cxx
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#include "vtkDistanceMeasureMeanOfClosestPointDistances.h"
#include "vtkObjectFactory.h"

namespace bmia {

	vtkStandardNewMacro( vtkDistanceMeasureMeanOfClosestPointDistances );

	////////////////////////////////////////////////////////////////////////
	vtkDistanceMeasureMeanOfClosestPointDistances::vtkDistanceMeasureMeanOfClosestPointDistances() :
		vtkDistanceMeasure()
	{
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceMeasureMeanOfClosestPointDistances::~vtkDistanceMeasureMeanOfClosestPointDistances()
	{
	}

	////////////////////////////////////////////////////////////////////////
	double vtkDistanceMeasureMeanOfClosestPointDistances::ComputeSpecific( 
		vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 )
	{
		double distance = 0.0;

		// Calculate total distance from fiber1 -> fiber2
		for( int i = 0; i < _nrPointIds1; i++ )
		{
			distance += GetClosestPointDistance( 
				this->GetPoints()->GetPoint( _pointIds1[i] ), _pointIds2, _nrPointIds2 );
		}

		// Calculate total distance from fiber2 -> fiber1
		for( int i = 0; i < _nrPointIds2; i++ )
		{
			distance += GetClosestPointDistance(
				this->GetPoints()->GetPoint( _pointIds2[i] ), _pointIds1, _nrPointIds1 );
		}

		// Take average of both and return result
		distance = distance / (_nrPointIds1 + _nrPointIds2);
		return distance;
	}

} // namespace bmia
