/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
