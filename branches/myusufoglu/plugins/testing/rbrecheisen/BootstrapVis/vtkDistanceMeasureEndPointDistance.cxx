/**
 * vtkDistanceMeasureEndPointDistance.cxx
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#include "vtkDistanceMeasureEndPointDistance.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"

#include <algorithm>

namespace bmia {

    vtkStandardNewMacro( vtkDistanceMeasureEndPointDistance );

	////////////////////////////////////////////////////////////////////////
    vtkDistanceMeasureEndPointDistance::vtkDistanceMeasureEndPointDistance() :
		vtkDistanceMeasure()
	{
	}

	////////////////////////////////////////////////////////////////////////
    vtkDistanceMeasureEndPointDistance::~vtkDistanceMeasureEndPointDistance()
	{
	}

	////////////////////////////////////////////////////////////////////////
    double vtkDistanceMeasureEndPointDistance::ComputeSpecific(
		vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 )
	{
        // WARNING: do not use the return pointer of this->GetPoints() because it gives
        // corrupt data resulting in a distance that is always zero
        double pt1[3];
        double pt2[3];
        double pt3[3];
        double pt4[3];

        this->GetPoints()->GetPoint( _pointIds1[0], pt1 );
        this->GetPoints()->GetPoint( _pointIds1[_nrPointIds1 - 1], pt2 );
        this->GetPoints()->GetPoint( _pointIds2[0], pt3 );
        this->GetPoints()->GetPoint( _pointIds2[_nrPointIds2 - 1], pt4 );

        double dist1 = vtkMath::Distance2BetweenPoints( pt1, pt3 ) + vtkMath::Distance2BetweenPoints( pt2, pt4 );
        double dist2 = vtkMath::Distance2BetweenPoints( pt1, pt4 ) + vtkMath::Distance2BetweenPoints( pt2, pt3 );

        return std::min( dist1, dist2 );
	}
} // namespace bmia
