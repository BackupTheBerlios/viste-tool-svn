/**
 * vtkDistanceMeasure.h
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkDistanceMeasure_h
#define bmia_vtkDistanceMeasure_h

#include "vtkObject.h"
#include "vtkPoints.h"

namespace bmia
{
	/**
	 * This class is an abstract class for computing distance measures.
	 * Children of this class should implement the Compute() method that
	 * returns a scalar.
	 */
	class vtkDistanceMeasure : public vtkObject
	{
	public:

		/** Computes the distance. Checks whether we have points and then
			passes the computation to child class */
		virtual double Compute( 
            vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 );

        /** Sets/gets the point data on which the distances are computed */
		void SetPoints( vtkPoints * _points );
		vtkPoints * GetPoints();

	protected:

		/** Constructor/destructor */
		vtkDistanceMeasure();
		virtual ~vtkDistanceMeasure();

		/** Computes closest-point distance for given point */
		double GetClosestPointDistance( double * _point, vtkIdType * _pointIds, int _nrPointIds );

	private:

		/** NOT IMPLEMENTED 
			copy constructor and assignment operator */
		vtkDistanceMeasure( const vtkDistanceMeasure & );
		void operator = ( const vtkDistanceMeasure & );

		/** Computes specific distance. Must be implemented by children */
		virtual double ComputeSpecific(
			vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 ) = 0;

		vtkPoints * Points;
	};

} // namespace bmia

#endif // bmia_vtkDistanceMeasure_h
