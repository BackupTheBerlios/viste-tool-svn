#ifndef __vtkConfidenceTable_h
#define __vtkConfidenceTable_h

#include <vtkObject.h>
#include <vector>

class vtkConfidenceTable : public vtkObject
{
public:

	typedef struct T
	{
		float key;
		int value;
	} T;

	static vtkConfidenceTable * New();
	vtkTypeRevisionMacro( vtkConfidenceTable, vtkObject );

	void Add( float confidence, int id );

	void Normalize();
	bool IsNormalized();
	void Sort();
	void Invert();

	float GetConfidenceAt( int index );
	float * GetConfidenceValues();
	int GetIdAt( int index );
	int GetNumberOfValues();

protected:

	vtkConfidenceTable();
	virtual ~vtkConfidenceTable();

private:

	std::vector< T > Table;
	bool Normalized;

	vtkConfidenceTable( const vtkConfidenceTable & );
	void operator = ( const vtkConfidenceTable & );
};

#endif
