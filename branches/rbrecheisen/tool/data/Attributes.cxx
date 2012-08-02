/*
 * Attributes.cxx
 *
 * 2009-10-29	Tim Peeters
 * - First version.
 *
 * 2009-10-30	Tim Peeters
 * - Removed "MatrixDouble" and "VectorDouble" attributes, added "vector<double>" 
 *   attributes. This is more standard, and can be returned by value without including 
 *   "VectorDouble.h" and "MatrixDouble.h". For easy use of "VectorDouble" and 
 *   "MatrixDouble" I can add constructors to those classes that take the 
 *   "vector<double>" as input.
 * - Added "print" functions.
 *
 * 2009-12-16	Tim Peeters
 * - Use "QList" and "QString" instead of "vector" and "string".
 *
 * 2010-02-26	Tim Peeters
 * - Added "vtkObject" as an attribute type.
 *
 * 2010-12-07	Tim Peeters
 * - Added "removeAttribute()" functions.
 *
 * 2011-05-13	Evert van Aart
 * - We now use "QHash" containers instead of "QList".
 * - The "addAttribute" functions now overwrite existing attributes with the same name.
 * - Because of this, the "changeAttribute" functions were no longer needed, and 
 *   have been removed. 
 * - Added "copyTransformationMatrix" and the "hasAttribute" functions. 
 *
 */


/** Includes */

#include "Attributes.h"


namespace bmia {


namespace data {


//-----------------------------[ Constructor ]-----------------------------\\

Attributes::Attributes()
{
	// Set hash pointers to NULL. The hash table won't be created until they are
	// actually needed, to save some memory.

	this->intAttributes				= NULL;
	this->doubleAttributes			= NULL;
	this->vectorDoubleAttributes	= NULL;
	this->vtkAttributes				= NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

Attributes::~Attributes()
{
	// Delete hash table for integer attributes
	if (this->intAttributes)
	{
		this->intAttributes->clear();
		delete this->intAttributes;
	}

	// Delete hash table for double attributes
	if (this->doubleAttributes)
	{
		this->doubleAttributes->clear();
		delete this->doubleAttributes;
	}

	if (this->vectorDoubleAttributes)
	{
		// Clear all existing double vectors
		for (QHash<QString, QList<double> >::iterator i = this->vectorDoubleAttributes->begin();
				i != this->vectorDoubleAttributes->end(); ++i)
		{
			(i.value()).clear();
		}

		// Clear and delete the vector list itself
		this->vectorDoubleAttributes->clear();
		delete this->vectorDoubleAttributes;
	}

	if (this->vtkAttributes)
	{
		// Unregister all VTK objects
		for (QHash<QString, vtkObject *>::iterator i = this->vtkAttributes->begin();
			i != this->vtkAttributes->end(); ++i)
		{
			(i.value())->UnRegister(NULL);
		}

		// Clear and delete the object list itself
		this->vtkAttributes->clear();
		delete this->vtkAttributes;
	}
}


//-----------------------[ copyTransformationMatrix ]----------------------\\

bool Attributes::copyTransformationMatrix(bmia::data::DataSet * ds)
{
	// check if the input data set contains a transformation matrix
	if (!(ds->getAttributes()->hasVTKObjectAttribute("transformation matrix")))
		return false;

	vtkObject * obj;

	// If so, try to get it
	if (!(ds->getAttributes()->getAttribute("transformation matrix", obj)))
		return false;

	if (!obj)
		return false;

	// Cast the object pointer to a matrix pointer
	vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(obj);

	if (!m)
		return false;

	// Make a copy of the matrix
	vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
	mCopy->DeepCopy(m);

	// Add the copied matrix to the VTK attributes
	this->addAttribute("transformation matrix", vtkObject::SafeDownCast(mCopy));

	return true;
}


//--------------------------[ printAllAttributes ]-------------------------\\

void Attributes::printAllAttributes()
{
	// Create a text stream for the standard output
    QTextStream out(stdout);

	// Print the integer attributes
    out << endl << this->intAttributes->size() << " integer attributes:" << endl;
    this->printAttributes<int>(this->intAttributes);

	// Print the double attributes
    out << endl << this->doubleAttributes->size() << " double attributes:" << endl;
    this->printAttributes<double>(this->doubleAttributes);
    
	// Print the double vector attributes
	out << endl << this->vectorDoubleAttributes->size() << " vector<double> attributes:" << endl;
    this->printAttributesVector<double>(this->vectorDoubleAttributes);
}


//---------------------------[ printAttributes ]---------------------------\\

template <class AttributeType>
void Attributes::printAttributes(QHash<QString, AttributeType> * hash)
{
	// Check if the hash table exists
	if (hash == NULL)
		return;

	// Create a text stream for the standard output
	QTextStream out(stdout);

	// Print the name and value for every entry
	for (typename QHash<QString, AttributeType>::const_iterator i = hash->constBegin(); i != hash->constEnd(); ++i)
	{
		out << " - " << i.key() << ": " << i.value() << endl;
	}
}


//------------------------[ printAttributesVector ]------------------------\\

template <class AttributeType>
void Attributes::printAttributesVector(QHash<QString, QList<AttributeType> > * hash)
{
	// Check if the hash table exists
	if (hash == NULL)
		return;

	// Create a text stream for the standard output
	QTextStream out(stdout);

	// Loop through all hash table entries
	for (typename QHash<QString, QList<AttributeType> >::const_iterator i = hash->constBegin(); i != hash->constEnd(); ++i)
	{
		// Get the current vector
		QList<AttributeType> v = i.value();

		// Print the vector name
		out << " - " << i.key() << ":";

		// Print all vector values
		for (typename QList<AttributeType>::const_iterator j = v.constBegin(); j != v.constEnd(); ++j)
		    out << " " << (*j);
	
		out << endl;
	}    
}


//----------------------------[ addAttribute ]-----------------------------\\

template <class AttributeType>
void Attributes::addAttribute(QString name, AttributeType value, QHash<QString, AttributeType> * & hash)
{
	// Create a new hash table if it does not yet exist
	if (hash == NULL)
	{
		hash = new QHash<QString, AttributeType>;
	}

	// Insert the name and value pair
	hash->insert(name, value);
}


void Attributes::addAttribute(QString name, int value)
{
	this->addAttribute<int>(name, value, this->intAttributes);
}


void Attributes::addAttribute(QString name, double value)
{
	this->addAttribute<double>(name, value, this->doubleAttributes);
}


void Attributes::addAttribute(QString name, QList<double> values)
{
	// Ignore empty lists
	if (values.isEmpty())
		return;

	this->addAttribute<QList<double> >(name, values, this->vectorDoubleAttributes);
}


void Attributes::addAttribute(QString name, vtkObject * object)
{
	// Ignore NULL pointers
	if (object == NULL)
		return;

	// Register the object (increments reference count)
	object->Register(NULL);

	// Try to find a VTK attribute with the target name
	if (this->vtkAttributes)
	{
		QHash<QString, vtkObject *>::const_iterator i = this->vtkAttributes->find(name);

		// If this attribute exists, unregister the old VTK object (since it will
		// be overwritten when the new object pointer is added to the attributes).

		if (i != this->vtkAttributes->constEnd())
			(i.value())->UnRegister(NULL);
	}

	this->addAttribute<vtkObject *>(name, object, this->vtkAttributes);
}


//-----------------------------[ getAttribute ]----------------------------\\

template <class AttributeType>
bool Attributes::getAttribute(QString name, AttributeType & value, QHash<QString, AttributeType> * & hash)
{
	// Check if the hash table exists
	if (hash == NULL)
		return false;

	// Check if the hash table contains data
	if (hash->isEmpty())
		return false;

	// Try to find the input name
	typename QHash<QString, AttributeType>::const_iterator i = hash->constFind(name);

	// Return false if the name was not found
	if (i == hash->constEnd())
		return false;

	// Otherwise, copy the value to the output
	value = i.value();

	return true;
}


bool Attributes::getAttribute(QString name, int & value)
{
    return this->getAttribute<int>(name, value, this->intAttributes);
}


bool Attributes::getAttribute(QString name, double & value)
{
    return this->getAttribute<double>(name, value, this->doubleAttributes);
}


bool Attributes::getAttribute(QString name, QList<double> & values)
{
    return this->getAttribute<QList<double> >(name, values, this->vectorDoubleAttributes);
}


bool Attributes::getAttribute(QString name, vtkObject * & object)
{
    return this->getAttribute<vtkObject *>(name, object, this->vtkAttributes);
}


//---------------------------[ removeAttribute ]---------------------------\\

template <class AttributeType>
bool Attributes::removeAttribute(QString name, QHash<QString, AttributeType> * & hash)
{
	// Check if the hash table exists
	if (hash == NULL)
		return false;

	// Check if the hash table contains data
	if (hash->isEmpty())
		return false;

	// Remove the entry with the target name
	int removeResult = hash->remove(name);

	// If the result is positive, one or more items were removed
	return (removeResult > 0);
}


bool Attributes::removeIntAttribute(QString name)
{
    return this->removeAttribute<int>(name, this->intAttributes);
}


bool Attributes::removeDoubleAttribute(QString name)
{
    return this->removeAttribute<double>(name, this->doubleAttributes);
}


bool Attributes::removeVectorDoubleAttribute(QString name)
{
	QList<double> v;

	// Try to get an existing attribute with the same name
	if (!(this->getAttribute<QList<double> >(name, v, this->vectorDoubleAttributes)))
		return false;

	// If it exists, clear it now
	v.clear();

    return this->removeAttribute<QList<double> >(name, this->vectorDoubleAttributes);
}


bool Attributes::removeVTKObjectAttribute(QString name)
{
	vtkObject * obj;

	// Try to get an existing attribute with the same name
	if (!(this->getAttribute<vtkObject *>(name, obj, this->vtkAttributes)))
		return false;

	// If it exists, unregister the VTK object
	obj->UnRegister(NULL);

    return this->removeAttribute<vtkObject *>(name, this->vtkAttributes);
}


bool Attributes::removeAttribute(QString name)
{
	bool wasRemoved = false;

	// Try to remove an attribute with the target name from each category
	wasRemoved |= this->removeIntAttribute(name);
	wasRemoved |= this->removeDoubleAttribute(name);
	wasRemoved |= this->removeVectorDoubleAttribute(name);
	wasRemoved |= this->removeVTKObjectAttribute(name);

	return wasRemoved;
}


//-----------------------------[ hasAttribute ]----------------------------\\

bool Attributes::hasIntAttribute(QString name)
{
	if (this->intAttributes == NULL)
		return false;

	return (this->intAttributes->contains(name));
}


bool Attributes::hasDoubleAttribute(QString name)
{
	if (this->doubleAttributes == NULL)
		return false;

	return (this->doubleAttributes->contains(name));
}


bool Attributes::hasVectorDoubleAttribute(QString name)
{
	if (this->vectorDoubleAttributes == NULL)
		return false;

	return (this->vectorDoubleAttributes->contains(name));
}


bool Attributes::hasVTKObjectAttribute(QString name)
{
	if (this->vtkAttributes == NULL)
		return false;

	return (this->vtkAttributes->contains(name));
}


bool Attributes::hasAttribute(QString name)
{
	// Check all categories for the target name
	if (this->hasIntAttribute(name))			return true;
	if (this->hasDoubleAttribute(name))			return true;
	if (this->hasVectorDoubleAttribute(name))	return true;
	if (this->hasVTKObjectAttribute(name))		return true;

	return false;
}


} // namespace data


} // namespace bmia
