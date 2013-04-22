#ifndef __vtkDistanceTableCollection_h
#define __vtkDistanceTableCollection_h

#include "vtkObject.h"
#include <vector>

namespace bmia
{
	class vtkDistanceTable;
    class vtkDistanceTableCollection : public vtkObject
	{
	public:

		static vtkDistanceTableCollection * New();

		void AddItem( vtkDistanceTable * _item );
		vtkDistanceTable * GetNextItem();

	protected:

		vtkDistanceTableCollection();
		virtual ~vtkDistanceTableCollection();

        std::vector<std::pair<std::string, vtkDistanceTable *> > * Collection;

	private:

		vtkDistanceTableCollection( const vtkDistanceTableCollection & _other );
		void operator = ( const vtkDistanceTableCollection & other );
	};
}

#endif
