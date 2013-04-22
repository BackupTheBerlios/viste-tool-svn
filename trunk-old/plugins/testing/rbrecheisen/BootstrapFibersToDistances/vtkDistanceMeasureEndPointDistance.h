/**
 * vtkDistanceMeasureEndPointDistance.h
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkDistanceMeasureEndPointDistance_h
#define bmia_vtkDistanceMeasureEndPointDistance_h

#include "vtkDistanceMeasure.h"

//namespace bmia
//{
	/**
     * This class computes a distance measure based on the end-point
	 * distance method.
	 */
    class vtkDistanceMeasureEndPointDistance : public vtkDistanceMeasure
	{
	public:

		/** Creats an instance of the object */
        static vtkDistanceMeasureEndPointDistance * New();

	protected:

		/** Constructor and destructor */
        vtkDistanceMeasureEndPointDistance();
        virtual ~vtkDistanceMeasureEndPointDistance();

	private:

		/** NOT IMPLEMENTED 
			copy constructor and assignment operator */
        vtkDistanceMeasureEndPointDistance( const vtkDistanceMeasureEndPointDistance & );
        void operator = ( const vtkDistanceMeasureEndPointDistance & );

        /** Computes the end-point distance */
		virtual double ComputeSpecific( 
			 vtkIdType * _pointIds1, int _nrPointIds1, vtkIdType * _pointIds2, int _nrPointIds2 );
    };

//} // namespace bmia

#endif // bmia_vtkDistanceMeasureClosestPointDistance_h
