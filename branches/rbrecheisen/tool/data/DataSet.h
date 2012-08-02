/*
 * DataSet.h
 * 
 * 2009-10-15	Tim Peeters
 * - First version
 *
 * 2010-09-29	Evert van Aart
 * - Added "updateData" function.
 *
 * 2010-11-10	Tim Peeters
 * - Add setName() function to rename data set.
 */


#ifndef bmia_data_DataSet_h
#define bmia_data_DataSet_h


class vtkDataSet;
class vtkImageData;
class vtkObject;
class vtkPolyData;
class vtkPointSet;


/** Includes - Qt */

#include <QString>


namespace bmia {


namespace data {


class Attributes;

/** This class represents a data set. It contains the name and kind of data, and 
	pointers to data attributes and the actual data. */

class DataSet
{
	public:

		/** Create a new object. An associated "Attributes" object is automatically 
			created and can be accessed by calling the "getAttributes()" function.
			@param name		The name of the data set.
			@param kind		The kind of data represented.
			@param obj		The VTK object containing the data of this data set. */

		DataSet(QString name, QString kind, vtkObject * obj = NULL);

		/** Destructor */
    
		~DataSet();

		/** Get the name of this data set. */
    
		QString getName();

		/** Rename this data set.
			@param newname	New data set name. */
    
		void setName(QString newname);

		/** Get the kind of this data set. */
    
		QString getKind();

		/** Get the attributes of this data set. */
    
		Attributes * getAttributes();

		/** Get the "vtkImageData" object associated with this data set. If there 
			is no associated "vtkImageData", NULL will be returned. This function 
			uses "getVTKDataSet()" and casts its returned value if it is of the 
			correct type. */

		vtkImageData * getVtkImageData();

		/** Get the "vtkPolyData" object associated with this data set. If there 
			is no associated "vtkPolyData", NULL will be returned. This function 
			uses "getVtkDataSet()" and casts its returned value if it is of the 
			correct type. */
		
		vtkPolyData * getVtkPolyData();

		/** Get the "vtkObject" associated with this data set. If there is no 
			associated "vtkObject", NULL will be returned. */

		vtkObject * getVtkObject();

		/** Update the pointer to the actual data. This can for example be used 
			for fibers: if you rerun a fiber tracking method with different 
			parameters, you'll end up with a new "vtkPolyData" object, and using 
			this function (in combination with "dataSetChanged") allows you to
			update the actual data without changing the data set pointer.  
			@param obj	New data object. */
	
		void updateData(vtkObject * obj);

	protected:
    
		/** Returns the "vtkDataSet" object associated with this data set. If 
			there is no associated "vtkDataSet", NULL will be returned. This 
			function casts the result of "getVtkObject()" to a "vtkDataSet"
			before returning it. */

		vtkDataSet * getVtkDataSet();

	private:

		QString name;				/**< Name of the data set. */
		QString kind;				/**< Data type. */
		Attributes * attributes;	/**< Data attributes. */
		vtkObject * vtkData;		/**< Data object. */

}; // class DataSet

} // namespace data


} // namespace bmia


#endif // bmia_data_DataSet_h
