/**
 * vtkDistanceMeasureMeanOfClosestPointDistances.h
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkDistanceMeasureMeanOfClosestPointDistances_h
#define bmia_vtkDistanceMeasureMeanOfClosestPointDistances_h

#include "vtkDistanceMeasure.h"

//namespace bmia
//{
	/**
	 * This class computes a distance measure based on the closest point
	 * distance method.
	 */
	class vtkDistanceMeasureMeanOfClosestPointDistances : public vtkDistanceMeasure
	{
	public:

		/** Creats an instance of the object */
		static vtkDistanceMeasureMeanOfClosestPointDistances * New();

	protected:

		/** Constructor and destructor */
		vtkDistanceMeasureMeanOfClosestPointDistances();
		virtual ~vtkDistanceMeasureMeanOfClosestPointDistances();

	private:

		/** NOT IMPLEMENTED 
			copy constructor and assignment operator */
		vtkDistanceMeasureMeanOfClosestPointDistances( const vtkDistanceMeasureMeanOfClosestPointDistances & );
		void operator = ( const vtkDistanceMeasureMeanOfClosestPointDistances & );

		/** Computes the mean of closest-point distances */
		virtual double ComputeSpecific( 
			 vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 );
    };

//} // namespace bmia

#endif // bmia_vtkDistanceMeasureMeanOfClosestPointDistances_h
