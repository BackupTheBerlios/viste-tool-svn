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
* bmiaNiftiWriter.cxx
*
* * 2013-03-16   Mehmet Yusufoglu
* - Create the class. Writes the scalar data in Nifti format.
* - 
*/


/** Includes */


#include "NiftiWriterPlugin.h"


namespace bmia {


	//-----------------------------[ Constructor ]-----------------------------\\

	NiftiWriterPlugin::NiftiWriterPlugin() : plugin::Plugin("NIfTI Writer")
	{

	}


	//------------------------------[ Destructor ]-----------------------------\\

	NiftiWriterPlugin::~NiftiWriterPlugin()
	{

	}


	//----------------------[ getSupportedFileExtensions ]---------------------\\

	QStringList NiftiWriterPlugin::getSupportedFileExtensions()
	{
		QStringList list;
		list.push_back("nii");
		//list.push_back("nii.gz");
		return list;
	}


	//---------------------[ getSupportedFileDescriptions ]--------------------\\

	QStringList NiftiWriterPlugin::getSupportedFileDescriptions()
	{
		QStringList list;
		list.push_back("NIfTI Files");
		list.push_back("GZipped NIfTI Files");
		return list;
	}



	void NiftiWriterPlugin::writeDataToFile(QString saveFileName,  data::DataSet *ds)
	{
		cout << "writeDataToFile"<< endl; 
		// Create a new reader object and set the filename
		bmiaNiftiWriter * writer = new bmiaNiftiWriter(this->core()->out());




		QString name(ds->getName());  
		QString kind(ds->getKind()); 

		if(saveFileName==NULL)
			return;
		QStringRef fileNameExtention(&saveFileName,saveFileName.lastIndexOf(QString(".")),4 );


		cout << "saveFileName and kind" << saveFileName.toStdString() << " " << kind.toStdString() << endl;
		// cout << fileNameExtention.toString().toStdString() << endl;
		//cin.get();


		vtkImageData * image   = ds->getVtkImageData();

		if(!image) cout << "Not image"<< endl;
		if( fileNameExtention.toString()==".nii" || fileNameExtention.toString()==".gz"  )
		{
			cout << "kind" << kind.toStdString() << endl;

			if(image && kind.contains("scalar volume"))// && (ds->getVtkImageData()->GetNumberOfScalarComponents() ==1 ))
			{
				qDebug() << "Writing the image data. No of scalar components is:" << image->GetNumberOfScalarComponents() << endl;

				cout << "saving nifti" << endl;
				vtkObject * attObject = vtkObject::New();
				cout << "Get attribute transf mat. "<< endl;
				ds->getAttributes()->getAttribute("transformation matrix", attObject);

				// Write the data to file 
				//QString err =

				writer->writeScalarVolume(image, saveFileName,attObject);
			}

			else if(image && ( kind.contains("DTI") || kind.contains("Eigen") ||  kind.contains("discrete sphere") || kind.contains("spherical harmonics") ) )
			{
				vtkObject * attObject = vtkObject::New();
				cout << "Get attribute transf mat. "<< endl;
				ds->getAttributes()->getAttribute("transformation matrix", attObject);

				// writer->writeDTIVolume(image, saveFileName, attObject); // works ok, not mind standart nifti
				writer->writeMindData(image,saveFileName, attObject, ds->getKind());
			}

			else 
			{
				qDebug() << "The data can not be saved due to data type2."<< endl;	
				return; 
			}

		}
		else 
		{
			qDebug() << "The data can not be saved due to extention mismatch."<< endl;	
			return; 
		}




	}


} // namespace bmia


Q_EXPORT_PLUGIN2(libNiftiWriterPlugin, bmia::NiftiWriterPlugin)
