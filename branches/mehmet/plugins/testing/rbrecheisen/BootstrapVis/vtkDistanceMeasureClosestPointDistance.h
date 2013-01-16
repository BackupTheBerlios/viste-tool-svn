/**
 * vtkDistanceMeasureClosestPointDistance.h
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkDistanceMeasureClosestPointDistance_h
#define bmia_vtkDistanceMeasureClosestPointDistance_h

#include "vtkDistanceMeasure.h"

namespace bmia
{
	/**
	 * This class computes a distance measure based on the closest point
	 * distance method.
	 */
	class vtkDistanceMeasureClosestPointDistance : public vtkDistanceMeasure
	{
	public:

		/** Creats an instance of the object */
		static vtkDistanceMeasureClosestPointDistance * New();

	protected:

		/** Constructor and destructor */
		vtkDistanceMeasureClosestPointDistance();
		virtual ~vtkDistanceMeasureClosestPointDistance();

	private:

		/** NOT IMPLEMENTED 
			copy constructor and assignment operator */
		vtkDistanceMeasureClosestPointDistance( const vtkDistanceMeasureClosestPointDistance & );
		void operator = ( const vtkDistanceMeasureClosestPointDistance & );

		/** Computes the closest-point distance */
		virtual double ComputeSpecific( 
			 vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 );
    };

} // namespace bmia

#endif // bmia_vtkDistanceMeasureClosestPointDistance_h
