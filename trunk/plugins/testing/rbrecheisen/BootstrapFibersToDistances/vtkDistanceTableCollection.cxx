#include "vtkDistanceTableCollection.h"
#include "vtkDistanceTable.h"

#include "vtkObjectFactory.h"

namespace bmia
{
    vtkStandardNewMacro( vtkDistanceTableCollection );

    ///////////////////////////////////////////////////////////////////
    vtkDistanceTableCollection::vtkDistanceTableCollection()
    {
        this->Collection = NULL;
    }

    ///////////////////////////////////////////////////////////////////
    vtkDistanceTableCollection::~vtkDistanceTableCollection()
    {
        if( this->Collection )
        {
            std::vector<std::pair<std::string, vtkDistanceTable *> >::iterator i = this->Collection->begin();
            for( ; i != this->Collection->end(); i++ )
                (*i)->Delete();
            this->Collection->clear();
            delete this->Collection;
        }
    }

    ///////////////////////////////////////////////////////////////////
    void vtkDistanceTableCollection::AddItem( vtkDistanceTable * _item )
    {
        if( _item == NULL )
            return;
        if( this->Collection == NULL )
            this->Collection = new std::vector<std::pair<std::string, vtkDistanceTable *> >;
        this->Collection->push_back( _item );
    }

    ///////////////////////////////////////////////////////////////////
    vtkDistanceTable * vtkDistanceTableCollection::GetNextItem()
    {
        return NULL;
    }
}
