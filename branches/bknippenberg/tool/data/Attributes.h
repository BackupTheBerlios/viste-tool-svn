/*
 * Attributes.h
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


#ifndef bmia_data_Attributes_h
#define bmia_data_Attributes_h


/** Includes - Custom Files */

#include "data/DataSet.h"

/** Includes - Qt */

#include <QHash>
#include <QtDebug>
#include <QTextStream>

/** Includes - VTK */

#include <vtkObject.h>
#include <vtkMatrix4x4.h>


namespace bmia {


namespace data {


/** This class contains attributes for a data set. Attributes can be of several 
	different data types; for each data type, a hash table is maintained, which
	uses the attribute names as keys and their values as data. The attributes
	currently support the following data types:
	- Integers.
	- Doubles.
	- Double vectors (implemented as "QList<double>"). 
	- VTK Objects.
*/


class Attributes
{
	public:
    
		/** Constructor. */

		Attributes();

		/** Destructor. */

		virtual ~Attributes();

		/** Copy the transformation matrix from the input data set to the current
			attribute set. Returns true if the transformation matrix was found
			and successfully copied, and false otherwise. Convenience function
			to make copying of transformation matrices - which is one of the 
			most common applications of attributes - much easier. 
			@param ds		Input data set. */

		bool copyTransformationMatrix(data::DataSet * ds);

		/** Add an integer attribute. If an integer attribute with the target 
			name already exists, it is replaced by the new value.
			@param name		Attribute name.
			@param value	Integer value. */
    
		void addAttribute(QString name, int value);

		/** Add a double attribute. If a double attribute with the target name 
			already	exists, it is replaced by the new value.
			@param name		Attribute name.
			@param value	Double value. */
    
		void addAttribute(QString name, double value);

		/** Add an attribute that is an array of vectors. If a double vector 
			attribute with the target name already exists, it is replaced by 
			the new double vector.
			@param name		Attribute name.
			@param values	Double vector. */

		void addAttribute(QString name, QList<double> values);

		/** Add an attribute that is a VTK object. If a VTK object attribute with
			the target name already exists, it is replaced by the new object pointer.
			The new object is registered (i.e., its reference count is incremented);
			if the attribute name already exists, the old VTK object is unregistered
			(i.e., its reference count is decremented, and it may get deleted). 
			@param name		Attribute name.
			@param values	Double vector. */

		void addAttribute(QString name, vtkObject * object);

		/** Get the value of the integer attribute with the given name. If the attribute
			does not exist, the function return false.
			@param name		Attribute name.
			@param value	Output attribute value. */

		bool getAttribute(QString name, int & value);

		/** Get the value of the double attribute with the given name. If the attribute
			does not exist, the function return false.
			@param name		Attribute name.
			@param value	Output attribute value. */

		bool getAttribute(QString name, double & value);

		/** Get the value of the double vector attribute with the given name. If 
			the attribute does not exist, the function return false.
			@param name		Attribute name.
			@param values	Output attribute vector. */

		bool getAttribute(QString name, QList<double> & values);

		/** Get the pointer of the VTK object attribute with the given name. If 
			the attribute does not exist, the function return false.
			@param name		Attribute name.
			@param value	Output attribute pointer. */

		bool getAttribute(QString name, vtkObject * & object);

		/** Removes the integer attribute with the given name. Returns true if the
			attribute existed and was successfully removed, and false otherwise.
			@param name		Attribute name. */

		bool removeIntAttribute(QString name);
 
		/** Removes the double attribute with the given name. Returns true if the
			attribute existed and was successfully removed, and false otherwise.
			@param name		Attribute name. */

		bool removeDoubleAttribute(QString name);

		/** Removes the double vector attribute with the given name. Returns true if the
			attribute existed and was successfully removed, and false otherwise.
			@param name		Attribute name. */

		bool removeVectorDoubleAttribute(QString name);

		/** Removes the VTK Object attribute with the given name. Returns true if the
			attribute existed and was successfully removed, and false otherwise. 
			Unregisters the VTK Object before removing it from the list. 
			@param name		Attribute name. */

		bool removeVTKObjectAttribute(QString name);	

		/** Removes attribute(s) with the given name from all categories. In other 
			words, it calls all "remove<DataType>Attribute".
			@param name		Attribute name. */
		
		bool removeAttribute(QString name);

		/** Check if the target integer attribute exists.
			@param name		Target attribute name. */

		bool hasIntAttribute(QString name);

		/** Check if the target double attribute exists.
			@param name		Target attribute name. */

		bool hasDoubleAttribute(QString name);

		/** Check if the target vector double attribute exists.
			@param name		Target attribute name. */

		bool hasVectorDoubleAttribute(QString name);

		/** Check if the target VTK object attribute exists.
			@param name		Target attribute name. */

		bool hasVTKObjectAttribute(QString name);

		/** Check if the target attribute exists. Checks all supported attribute types.
			@param name		Target attribute name. */

		bool hasAttribute(QString name);

		/** Print all attributes to the standard output. */

		void printAllAttributes();

		/** Return the hash table pointer for the integer attributes. */

		QHash<QString, int> * getIntAttributes()
		{
			return intAttributes;
		}

		/** Return the hash table pointer for the double attributes. */

		QHash<QString, double> * getDoubleAttributes()
		{
			return doubleAttributes;
		}

		/** Return the hash table pointer for the double vector attributes. */

		QHash<QString, QList<double> > * getVectorDoubleAttributes()
		{
			return vectorDoubleAttributes;
		}

		/** Return the hash table pointer for the VTK attributes. */

		QHash<QString, vtkObject *> * getVtkAttributes()
		{
			return vtkAttributes;
		}

	protected:

		/** Template for adding an attribute. Creates the hash table if it does not yet
			exist, and adds the input value to this hash.
			@param name		Attribute name.
			@param value	Attribute value.
			@param hash		Hash table for the specified attribute type. */

		template <class AttributeType>
		void addAttribute(QString name, AttributeType value, QHash<QString, AttributeType> * & hash);

		/** Template for fetching an attribute. Returns false if the desired attribute
			does not exist, or if the hash table does not exist. 
			@param name		Target attribute name.
			@param value	Output attribute value.
			@param hash		Hash table for the specified attribute type. */

		template <class AttributeType>
		bool getAttribute(QString name, AttributeType & value, QHash<QString, AttributeType> * & hash);

		/** Template for removing an attribute. Returns true if the target attribute 
			was found in and deleted from the specified hash table, and false otherwise.
			@param name		Target attribute name.
			@param hash		Hash table for the specified attribute type. */

		template <class AttributeType>
		bool removeAttribute(QString name, QHash<QString, AttributeType> * & hash);

		/** Template for printing scalar attributes (e.g., integers or doubles).
			For each attribute in the hash table, we print its name and its value.
			@param hash		Hash table for the specified attribute type. */

		template <class AttributeType>
		void printAttributes(QHash<QString, AttributeType> * hash);

		/** Template for printing vector attributes of scalar types. The vectors
			are stored as "QList" containers; the scalar types can for example
			be integers or doubles. For each vector, we print its name, followed
			by all its values.
			@param hash		Hash table for the specified attribute type. */

			template <class AttributeType>
			void printAttributesVector(QHash<QString, QList<AttributeType> > * hash);

		QHash<QString, int> *				intAttributes;				/**< Hash table for integer attributes. */
		QHash<QString, double> *			doubleAttributes;			/**< Hash table for double attributes. */
		QHash<QString, QList<double> > *	vectorDoubleAttributes;		/**< Hash table for double vector attributes. */
		QHash<QString, vtkObject *> *		vtkAttributes;				/**< Hash table for VTK object attributes. */

}; // class Attributes


} // namespace data


} // namespace bmia


#endif // bmia_data_Attributes_h
