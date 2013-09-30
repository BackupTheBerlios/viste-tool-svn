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
 * VtiReaderPlugin.cxx
 *
 * 2010-11-09	Tim Peeters
 * - First version
 *
 * 2011-01-27	Evert van Aart
 * - Added support for loading transformation matrices
 *
 */

#include "VtiReaderPlugin.h"

#include "Helpers/TransformationMatrixIO.h"

#include <vtkXMLImageDataReader.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkMatrix4x4.h>

#include <QtCore/QFileInfo>
#include <QtCore/QDir>
#include <QtDebug>

namespace bmia {

VtiReaderPlugin::VtiReaderPlugin() : plugin::Plugin("VTI reader")
{
    // nothing to do
}

VtiReaderPlugin::~VtiReaderPlugin()
{
    // bye bye!!
}

// Return the file extensions that this reader support as a QStringList.
QStringList VtiReaderPlugin::getSupportedFileExtensions()
{
    QStringList list;
    list.push_back("vti");
    return list;
}

QStringList VtiReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("VTK Image Data");
	return list;
}

void VtiReaderPlugin::loadDataFromFile(QString filename)
{
    // Write a message to the log so we know what's going on in case something
    // goes wrong
    this->core()->out()->logMessage("Trying to load data from file "+filename);

    // Create a new reader object and set the filename
    vtkXMLImageDataReader* reader = vtkXMLImageDataReader::New();
    reader->SetFileName(filename.toAscii().data());
    
    // Start a progress dialog that shows the progress of reading the data
    this->core()->out()->createProgressBarForAlgorithm(reader, "VTI Reader", "Reading VTK XML Image file...");

    // Read the data from file by updating the VTK reader object.
    reader->Update();

    // Stop reporting progress
    this->core()->out()->deleteProgressBarForAlgorithm(reader);

    // Define the name of the dataset as the filename.
    QDir dir(filename);
    QFileInfo fi(filename);
    QString name = fi.dir().dirName() +"/"+ fi.baseName();
    // Another possibility is to define the name of the dataset as
    // the next free Id (integer number):
    //QString name = this->core()->data()->getNextDataSetIdAsString();

    // Construct a new DataSet object with that has the output
    // of the VTK reader as VTK data set.
    QString kind = "scalar volume";

    vtkImageData* data = reader->GetOutput();
    Q_ASSERT(data);
  if (data->GetNumberOfScalarComponents() == 9 || data->GetNumberOfScalarComponents() == 6 ) // Roy's tensor data has 9
        {
            vtkImageData* newdata = vtkImageData::New();
            newdata->CopyStructure(data);
            newdata->SetScalarTypeToFloat();
            newdata->SetNumberOfScalarComponents(6);
            vtkFloatArray* out_array = vtkFloatArray::New();
            out_array->SetName("Tensors");

            Q_ASSERT(data->GetPointData());
            vtkDataArray* in_array = data->GetPointData()->GetScalars();
            Q_ASSERT(in_array);

            if(data->GetNumberOfScalarComponents() == 9) {
                out_array->SetNumberOfComponents(6);
                out_array->SetNumberOfTuples(in_array->GetNumberOfTuples());
                double* tuple; double newtuple[6];
                int i;

                for (vtkIdType n = 0; n < in_array->GetNumberOfTuples(); n++)
                {
                    tuple = in_array->GetTuple9(n);
                    for (i=0; i < 3; i++) newtuple[i] = tuple[i];
                    newtuple[3] = tuple[4];
                    newtuple[4] = tuple[5];
                    newtuple[5] = tuple[8];
                    out_array->SetTuple(n, newtuple);

                }  
            }
            else if(data->GetNumberOfScalarComponents() == 6) // this loop suitable for general but leave 6 for now
            {
                out_array->SetNumberOfComponents(data->GetNumberOfScalarComponents());
                out_array->SetNumberOfTuples(in_array->GetNumberOfTuples());
                const int comp = data->GetNumberOfScalarComponents();
                double* tuple; double *newtuple = new double[comp];
                int i;
                for (vtkIdType n = 0; n < in_array->GetNumberOfTuples(); n++)
                {
                    tuple = in_array->GetTuple(n);
                    for (i=0; i < data->GetNumberOfScalarComponents(); i++) newtuple[i] = tuple[i];

                    out_array->SetTuple(n, newtuple);

                } // for n
            }

            Q_ASSERT(newdata->GetPointData());
            newdata->GetPointData()->SetScalars(out_array);
            newdata->GetPointData()->SetNumberOfTuples(in_array->GetNumberOfTuples());
            newdata->SetScalarTypeToFloat();
            data = newdata;
            kind = "DTI";
        }//if

  else if(data->GetPointData()->GetNumberOfArrays() > 1) {

	  	int nArrays;
			nArrays = data->GetPointData()->GetNumberOfArrays() ;  // 1 for the original image N for the arrays added for unit vectors
			  
			bool hasVector=false;
			for(unsigned int nr = 0; nr <nArrays  ; nr++)
			{
				//	outUnitVectorList.push_back(vtkDoubleArray::New());
				//outUnitVectorList.at(nr)->SetNumberOfComponents(3);
				//outUnitVectorList.at(nr)->SetNumberOfTuples(maximaVolume->GetNumberOfPoints());

				//outUnitVectorList.at(nr)->SetName( arrName.toStdString().c_str() );  //fist vector array for each point (keeps only the first vector)
				QString name(data->GetPointData()->GetArrayName(nr));
				cout << name.toStdString() << endl;
				if(name=="") return;
				if ((data->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( data->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
				{
					hasVector=true;
				}

			}
			if (hasVector) kind="unit vector volume" ;
  }

    Q_ASSERT(!kind.isEmpty());

    data::DataSet* ds = new data::DataSet(name, kind, data);

	std::string err;

	// Try to read the transformation matrix from a ".tfm" file
	vtkMatrix4x4 * m = TransformationMatrixIO::readMatrix(filename.toStdString(), err);

	// If we succeeded, add the matrix to the data set
	if (m)
	{
		ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(m));
	}
	// If an error occurred while reading the matrix file, print it
	else if (!(err.empty()))
	{
		this->core()->out()->showMessage(QString(err.c_str()));
	}

	// Add the new DataSet to the data manager:
    this->core()->data()->addDataSet(ds);

    if (data->GetNumberOfScalarComponents() == 9)
	{
	data->Delete(); data = NULL;
	}

	// Load Vectos if Any: Left For Future
	// loadArraysAsDataSet(reader->GetOutput());
	// Delete the reader that was used to read the data.
    reader->Delete(); reader = NULL;
}

void VtiReaderPlugin::loadArraysAsDataSet(vtkImageData *img) 
{
	unsigned int nArrays=0;  

		 Q_ASSERT(img->GetPointData());
			nArrays =  img->GetPointData()->GetNumberOfArrays() ;  // 1 for the original image N for the arrays added for unit vectors
			std::vector<vtkDoubleArray *> outUnitVectorListFromFile;
			 
		QString kind("vectors");
		
		for(unsigned int nr = 0; nr <nArrays  ; nr++)
		{
			//	outUnitVectorList.push_back(vtkDoubleArray::New());
			//outUnitVectorList.at(nr)->SetNumberOfComponents(3);
			//outUnitVectorList.at(nr)->SetNumberOfTuples(maximaVolume->GetNumberOfPoints());
			 
			//outUnitVectorList.at(nr)->SetName( arrName.toStdString().c_str() );  //fist vector array for each point (keeps only the first vector)
			QString name(img->GetPointData()->GetArrayName(nr));
			if ((img->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( img->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
			{
				//outUnitVectorListFromFile.push_back( vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str()  )));
			  data::DataSet* ds = new data::DataSet(name, kind, vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str())));
	 this->core()->data()->addDataSet(ds);
			}
		}
}

} // namespace bmia
Q_EXPORT_PLUGIN2(libVtiReaderPlugin, bmia::VtiReaderPlugin)
