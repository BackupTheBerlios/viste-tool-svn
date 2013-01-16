/**
 * vtkDistanceTable.cxx
 * by Ralph Brecheisen
 *
 * 2010-01-20	Ralph Brecheisen
 * - First version
 */
#include "vtkDistanceTable.h"

#include "vtkObjectFactory.h"

#include <cassert>
#include <algorithm>

namespace bmia {

	vtkStandardNewMacro( vtkDistanceTable );

	////////////////////////////////////////////////////////////////////////
	bool Compare( const std::pair<double, int> & A, const std::pair<double, int> & B )
	{
		bool result = A.first < B.first;
		return result;
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceTable::vtkDistanceTable()
	{
		this->Min =  VTK_DOUBLE_MAX;
		this->Max = -VTK_DOUBLE_MAX;

		// Instantiate the table right away so we don't have to check for
		// its non-nullity in every method
		this->Table = new std::vector<std::pair<double, int> >;

		this->Normalized = false;
}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceTable::~vtkDistanceTable()
	{
		delete this->Table;
	}

	////////////////////////////////////////////////////////////////////////
	void vtkDistanceTable::Add( double _distance, int _index )
	{
		this->Table->push_back( std::pair<double, int>(_distance, _index) );

		// Update min/max range
		this->Min = std::min( this->Min, _distance );
		this->Max = std::max( this->Max, _distance );
	}

	////////////////////////////////////////////////////////////////////////
	const int vtkDistanceTable::GetNumberOfElements() const
	{
		return this->Table->size();
	}

	////////////////////////////////////////////////////////////////////////
	double vtkDistanceTable::GetDistance( int _index ) const
	{
		std::pair<double, int> elem = this->Table->at( _index );
		return elem.first;
	}

	////////////////////////////////////////////////////////////////////////
	int vtkDistanceTable::GetCellIndex( int _index ) const
	{
		std::pair<double, int> elem = this->Table->at( _index );
		return elem.second;
	}

	////////////////////////////////////////////////////////////////////////
	double vtkDistanceTable::GetMinDistance() const
	{
		return this->Min;
	}

	////////////////////////////////////////////////////////////////////////
	double vtkDistanceTable::GetMaxDistance() const
	{
		return this->Max;
	}

	////////////////////////////////////////////////////////////////////////
	void vtkDistanceTable::GetMinMaxDistance( double & _min, double & _max ) const
	{
		_min = this->Min;
		_max = this->Max;
	}

	////////////////////////////////////////////////////////////////////////
	double * vtkDistanceTable::GetMinMaxDistance() const
	{
		double * range = new double[2];
		range[0] = this->Min;
		range[1] = this->Max;
		return range;
	}

	////////////////////////////////////////////////////////////////////////
	void vtkDistanceTable::Sort()
	{
		std::sort( this->Table->begin(), this->Table->end(), Compare );
	}

	////////////////////////////////////////////////////////////////////////
	void vtkDistanceTable::Normalize()
	{
		double range  = (this->Max - this->Min);
		assert( range != 0.0 );
		
		std::vector<std::pair<double, int> >::iterator iter = this->Table->begin();
		for( ; iter != this->Table->end(); iter++ )
		{
			double normDist = (iter->first - this->Min) / range;
			iter->first = normDist;
		}
		
		this->Normalized = true;
	}

	////////////////////////////////////////////////////////////////////////
	bool vtkDistanceTable::IsNormalized()
	{
		return this->Normalized;
	}
	
	////////////////////////////////////////////////////////////////////////
	void vtkDistanceTable::Print( const std::string & _fileName )
	{
		std::vector<std::pair<double, int> >::iterator iter = this->Table->begin();

		for( ; iter != this->Table->end(); iter++ )
			std::cout << "vtkDistanceTable::Print() " << (*iter).first << " " << (*iter).second << std::endl;

		std::cout << "vtkDistanceTable::Print() minimum distance: " << this->Min << std::endl;
		std::cout << "vtkDistanceTable::Print() maximum distance: " << this->Max << std::endl;
	}
}
