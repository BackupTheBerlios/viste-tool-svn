#include <vtkConfidenceTable.h>
#include <vtkObjectFactory.h>
#include <algorithm>

vtkCxxRevisionMacro( vtkConfidenceTable, "$Revision: 1.0 $");
vtkStandardNewMacro( vtkConfidenceTable );

bool vtkConfidenceTable_SortKeys( vtkConfidenceTable::T A, vtkConfidenceTable::T B )
{
	return (A.key > B.key);
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceTable::vtkConfidenceTable()
{
	this->Normalized = false;
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceTable::~vtkConfidenceTable()
{
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceTable::Add( float confidence, int id )
{
	T pair;
	pair.key = confidence;
	pair.value = id;
	this->Table.push_back( pair );
}

///////////////////////////////////////////////////////////////////////////
int vtkConfidenceTable::GetNumberOfValues()
{
	return this->Table.size();
}

///////////////////////////////////////////////////////////////////////////
float vtkConfidenceTable::GetConfidenceAt( int index )
{
	T & pair = this->Table.at( index );
	return pair.key;
}

///////////////////////////////////////////////////////////////////////////
float * vtkConfidenceTable::GetConfidenceValues()
{
	int k = 0;
	float * tmp = new float[this->Table.size()];
	std::vector< T >::iterator i = this->Table.begin();
	for( ; i != this->Table.end(); ++i )
		tmp[k++] = (*i).key;
	return tmp;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceTable::Print()
{
    std::vector< T >::iterator i = this->Table.begin();
    for( ; i != this->Table.end(); ++i )
        std::cout << (*i).key << " " << (*i).value << std::endl;
}

///////////////////////////////////////////////////////////////////////////
int vtkConfidenceTable::GetIdAt( int index )
{
	T & pair = this->Table.at( index );
	return pair.value;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceTable::Normalize()
{
	float minimum =  999999.0f;
	float maximum = -999999.0f;

	std::vector< T >::iterator i = this->Table.begin();
	for( ; i != this->Table.end(); ++i )
	{
		if( (*i).key < minimum )
			minimum = (*i).key;
		if( (*i).key > maximum )
			maximum = (*i).key;
	}

	for( i = this->Table.begin(); i != this->Table.end(); ++i )
		(*i).key = ((*i).key - minimum) / (maximum - minimum);

	this->Sort();
	this->Normalized = true;
}

///////////////////////////////////////////////////////////////////////////
bool vtkConfidenceTable::IsNormalized()
{
	return this->Normalized;
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceTable::Sort()
{
	std::sort( this->Table.begin(), this->Table.end(),
			   vtkConfidenceTable_SortKeys );
}

///////////////////////////////////////////////////////////////////////////
void vtkConfidenceTable::Invert()
{
	if( this->IsNormalized() == false )
		this->Normalize();

	std::vector< T >::iterator i = this->Table.begin();
	for( ; i != this->Table.end(); ++i )
		(*i).key = 1.0f - (*i).key;

	this->Sort();
}
