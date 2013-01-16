/**
 * vtkDistanceTable.h
 * by Ralph Brecheisen
 *
 * 2010-01-20	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkDistanceTable_h
#define bmia_vtkDistanceTable_h

#include "vtkObject.h"
#include "vtkPolyData.h"

#include "vtkDistanceMeasure.h"
#include "vtkDistanceTable.h"

#include <vector>

//namespace bmia
//{
	/**
	 * This class represents a special-purpose table. Each element is a value pair
	 * containing a distance value and a integer offset. The integer offset points
	 * into a vtkCellArray object to identify a single streamline. The distance value
	 * represents the distance of this streamline to a certain center streamline.
	 */
	class vtkDistanceTable : public vtkObject
	{
	public:

		/** Creates new instance of the table */
		static vtkDistanceTable * New();

		/** Adds a new <distance, cell index> pair. It also updates the min/max
		    range for this table */
		void Add( double _distance, int _index );

		/** Returns the number of elements in the table */
		const int GetNumberOfElements() const;

		/** Returns distance and cell index at given index. These functions are
		    called during rendering so they need to be fast */
		double GetDistance( int _index ) const;
		int GetCellIndex( int _index ) const;

		/** Returns min/max distance range */
		double GetMinDistance() const;
		double GetMaxDistance() const;
		void GetMinMaxDistance( double & _min, double & _max ) const;
		double * GetMinMaxDistance() const;

        /** Sorts the distance table by distance from small to large */
		void Sort();
		
		/** Normalizes the distances to a value between [0,1]. This way we can
		    use the distances for computing confidence intervals */
		void Normalize();
		bool IsNormalized();

		/** Prints distance table. If filename is empty, the output will be
		    redirected to the standard output */
		void Print( const std::string & _fileName = "" );

		/** Writes contents of distance table to given filename
			@param fileName The filename */
		void Write( const std::string & fileName );

		/** Reads contents of distance table from given filename
			@param fileName The filename */
		void Read( const std::string & fileName );

	protected:

		/** Constructor and destructor */
		vtkDistanceTable();
		virtual ~vtkDistanceTable();

	private:

		/** NOT IMPLEMENTED copy constructor and assignment operator */
		vtkDistanceTable( const vtkDistanceTable & );
		void operator = ( const vtkDistanceTable & );

		double Min, Max;
		bool Normalized;

        std::vector<std::pair<double,int> > * Table;
	};

//} // namespace bmia

#endif // bmia_vtkDistanceTable_h

