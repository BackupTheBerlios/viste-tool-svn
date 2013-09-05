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
/*
* SaveDialog.cxx
*
* 2013-02-08 Mehmet Yusufoglu	
* Created for saving the data, a similar class of DataDialog. saveSelectedItem function 
* is the main function.  vti saving and .pol and .fbs saving are here may be moved to a 
* plugin. Nifti andNifti Mind saving is implemented in NiftiWriterPlugin.
*
* 2013-02-08 Mehmet Yusufoglu
* -vti saving is converted to binary. 
* -Derivatices of DTI like FA is listed as being 0 byte. When the scalar volume is 
* to be saved get image data and update it so that the data will be produced from 
* the DTI.
*
* 2013-05-16   Mehmet Yusufoglu
* - add saving eigen image as .vti binary binary.
*/


#include "SaveDialog.h"


namespace bmia {


	using namespace data;


	namespace gui {


		//-----------------------------[ Constructor ]-----------------------------\\

		SaveDialog::SaveDialog(Manager * dManager, QWidget * parent) :
			QDialog(parent),
			treeWidget(new QTreeWidget),
			closeButton(new QPushButton("Close")),
			saveButton(new QPushButton("Save"))
		{
			// Store the pointer to the data manager
			Q_ASSERT(dManager);
			this->manager = dManager;

			// Setup the tree widgets
			this->treeWidget->setColumnCount(3);
			this->treeWidget->setColumnWidth(0, 300);
			this->treeWidget->setColumnWidth(1, 10);
			this->treeWidget->setColumnWidth(2, 75);
			this->treeWidget->setAlternatingRowColors(false);
			this->treeWidget->setAnimated(true);
			this->treeWidget->header()->hide();
			this->treeWidget->header()->setStretchLastSection(false);

			this->setMinimumWidth(430);

			// Connect the close button to the "close" function
			connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));
			connect(saveButton, SIGNAL(clicked()), this, SLOT(saveSelectedItem()));
			connect(this->treeWidget,SIGNAL(itemClicked(QTreeWidgetItem*, int)), SLOT(setClickedItem(QTreeWidgetItem*,int)));


			// Create the main layout and the layout for the buttons
			QVBoxLayout * mainLayout   = new QVBoxLayout;
			QHBoxLayout * buttonLayout = new QHBoxLayout;

			// Add the tree widget to the main layout
			mainLayout->addWidget(this->treeWidget);

			// Setup the button layout
			buttonLayout->addStretch(0);
			buttonLayout->addWidget(saveButton);
			buttonLayout->addWidget(closeButton);
			buttonLayout->addStretch(0);

			// Add the button layout to the main layout
			mainLayout->addLayout(buttonLayout);

			// Set the layout
			this->setLayout(mainLayout);

			this->setWindowTitle(tr("List of available data sets"));

			// Add self to the data manager as a consumer
			this->manager->addConsumer(this);
		}


		//------------------------------[ Destructor ]-----------------------------\\

		SaveDialog::~SaveDialog()
		{
			// Clear the list of data sets
			this->dataSets.clear();

			// Clear the tree widget
			if (this->treeWidget)
				this->treeWidget->clear();

			// Delete the main layout of the dialog
			if (this->layout())
				delete this->layout();
		}


		//--------------------------------[ update ]-------------------------------\\

		void SaveDialog::update()
		{
			// Remove everything from the tree widget
			this->treeWidget->clear();

			// Get all data sets from the manager
			QList<DataSet *> dataSets = this->manager->listAllDataSets();

			// Loop through the data sets
			for (int i = 0; i < dataSets.size(); ++i)
			{
				if (!dataSets[i])
					continue;

				// Add the data set to the tree widget
				this->populateTreeWidget(dataSets[i]);
			}
		}


		//--------------------------[ populateTreeWidget ]-------------------------\\

		void SaveDialog::populateTreeWidget(DataSet * ds)
		{
			// Create a new item from the data set
			QTreeWidgetItem * newItem = this->createWidgetItem(ds);

			// Append it to the tree widget
			this->treeWidget->addTopLevelItem(newItem);

			// Recompute the size of the window
			this->treeWidget->resizeColumnToContents(0);
			this->treeWidget->resizeColumnToContents(1);

			if (this->treeWidget->columnWidth(0) < 300)
				this->treeWidget->setColumnWidth(0, 300);
			if (this->treeWidget->columnWidth(2) <  60)
				this->treeWidget->setColumnWidth(2, 60);

			this->setFixedWidth(this->treeWidget->columnWidth(0) + this->treeWidget->columnWidth(2) + 50);
		}


		//---------------------------[ createWidgetItem ]--------------------------\\

		QTreeWidgetItem * SaveDialog::createWidgetItem(bmia::data::DataSet *ds)
		{
			// Create a new item
			QTreeWidgetItem * dsItem = new QTreeWidgetItem;

			// Set the text of the data set
			dsItem->setText(0, ds->getName() + " : " + ds->getKind());

			// Get the VTK image data, polydata, and object. If the VTK data in the data
			// set is not of the specified type, the pointer will be NULL.

			vtkImageData * image   = ds->getVtkImageData();
			vtkPolyData * polyData = ds->getVtkPolyData();
			vtkObject * obj        = ds->getVtkObject();

			vtkPointSet * pointSet = NULL;

			// Cast the VTK object to a point set pointer
			if (obj) 
			{
				pointSet = vtkPointSet::SafeDownCast(obj);
			}
			

			// Estimated data size
			unsigned long dataSize = 0;

			// True if we've got a VTK data object for which the function "GetActualMemorySize"
			// is available, false otherwise.

			bool dataSizeAvailable = false;

			// VTK image data
			if (image)
			{

				// Why derivatices of DTI like FA is listed as being 0 byte. lets update. This may take time at the startup.
				//if (ds->getVtkImageData()->GetActualMemorySize() == 0)
				//{
				//	ds->getVtkImageData()->Update();
				//}
				// Print data set type and dimensionality
				this->addSubItem(dsItem, "Type: Image (" + QString::number(image->GetDataDimension()) + "D)");
				this->addSubItem(dsItem, "Components Per Voxel: " + QString::number(image->GetNumberOfScalarComponents()) + "");
				this->addSubItem(dsItem, "Scalar Type: " + QString::number(image->GetScalarType()) + "");
				if(image->GetDataDimension() == 3)
					this->addSubItem(dsItem, "Spacing: " + QString::number(image->GetSpacing()[0]) + " "+ QString::number(image->GetSpacing()[1]) + " " + QString::number(image->GetSpacing()[2]));
				if(image->GetDataDimension() == 3)
					this->addSubItem(dsItem, "Dimensions: " + QString::number(image->GetDimensions()[0]) + " "+ QString::number(image->GetDimensions()[1]) + " " + QString::number(image->GetDimensions()[2]));
				
				dataSize = image->GetActualMemorySize();
				dataSizeAvailable = true;

			}  // if [image data]

			// VTK polydata
			// Note: No "else" here on purpose. Might have both image- and polydata.

			if (polyData) 
			{
				// Get and print information about the polydata
				this->addSubItem(dsItem, "Type: PolyData");
				this->addSubItem(dsItem, "Number of Verts: "  + QString::number(polyData->GetNumberOfVerts()));
				this->addSubItem(dsItem, "Number of Lines: "  + QString::number(polyData->GetNumberOfLines()));
				this->addSubItem(dsItem, "Number of Polys: "  + QString::number(polyData->GetNumberOfPolys()));
				this->addSubItem(dsItem, "Number of Strips: " + QString::number(polyData->GetNumberOfStrips()));

				// Get the memory size
				dataSize += polyData->GetActualMemorySize();
				dataSizeAvailable = true;

			} // if [polyData]

			// VTK point set
			else if (pointSet)
			{
				// Print the number of points in the set
				this->addSubItem(dsItem, "Type: Point Set");
				this->addSubItem(dsItem, "Number of Points: " + QString::number(pointSet->GetNumberOfPoints()));

				// Get the memory size
				dataSize = pointSet->GetActualMemorySize();
				dataSizeAvailable = true;
			}

			// Print the data size in the second column
			if (dataSizeAvailable)
			{
				dsItem->setText(2, QString::number(dataSize) + "kB");
			}
			else
			{
				dsItem->setText(2, "N/A");
			}

			// Align the data size on the right
			dsItem->setTextAlignment(2, Qt::AlignRight);

			// Get the attributes of the data set
			Attributes * attr = ds->getAttributes();
			Q_ASSERT(attr);

			// Print all integer attributes
			QHash<QString, int> * intHash = attr->getIntAttributes();

			if (intHash)
			{
				for (QHash<QString, int>::const_iterator i = intHash->constBegin(); i != intHash->constEnd(); ++i)
				{
					this->addSubItem(dsItem, i.key() + " = " + QString::number(i.value()));
				}
			}

			// Print all double attributes
			QHash<QString, double> * doubleHash = attr->getDoubleAttributes();

			if (doubleHash)
			{
				for (QHash<QString, double>::const_iterator i = doubleHash->constBegin(); i != doubleHash->constEnd(); ++i)
				{
					this->addSubItem(dsItem, i.key() + " = " + QString::number(i.value()));
				}
			}

			// Print all vector attribute names
			QHash<QString, QList<double> > * vectorHash = attr->getVectorDoubleAttributes();

			if (vectorHash)
			{
				for (QHash<QString, QList<double> >::const_iterator i = vectorHash->constBegin(); i != vectorHash->constEnd(); ++i)
				{
					this->addSubItem(dsItem, "Vector: " + i.key());
				}
			}

			// Print all VTK attribute names
			QHash<QString, vtkObject *> * vtkHash = attr->getVtkAttributes();

			if (vtkHash)
			{
				for (QHash<QString, vtkObject *>::const_iterator i = vtkHash->constBegin(); i != vtkHash->constEnd(); ++i)
				{
					this->addSubItem(dsItem, "VTK Object: " + i.key());
				}
			}

			// Return the new widget item
			return dsItem;
		}


		//------------------------------[ addSubItem ]-----------------------------\\

		void SaveDialog::addSubItem(QTreeWidgetItem * parentItem, QString itemText)
		{
			// Create a new tree widget item
			QTreeWidgetItem * subItem = new QTreeWidgetItem(parentItem);

			// Set the required text
			subItem->setText(0, itemText);
		}


		//-----------------------------[ dataSetAdded ]----------------------------\\

		void SaveDialog::dataSetAdded(DataSet * ds)
		{
			if (!ds)
				return;

			// Check if the data set has already been added (should never happen)
			if (this->dataSets.contains(ds))
				this->dataSetChanged(ds);

			// Add the data set to the list
			this->dataSets.append(ds);

			// Add it to the tree widget
			this->populateTreeWidget(ds);
		}


		//----------------------------[ dataSetChanged ]---------------------------\\

		void SaveDialog::dataSetChanged(DataSet * ds)
		{
			if (!ds)
				return;

			// Check if the data set has been added before
			if (!(this->dataSets.contains(ds)))
				return;

			// Get the index of the data set
			int dsIndex = this->dataSets.indexOf(ds);

			// Remove the corresponding item from the tree widget, and delete it
			QTreeWidgetItem * currentItem = this->treeWidget->takeTopLevelItem(dsIndex);
			delete currentItem;

			// Create a new item for the data set
			QTreeWidgetItem * newItem = this->createWidgetItem(ds);

			// Insert the new item at the position of the old one
			this->treeWidget->insertTopLevelItem(dsIndex, newItem);

			// Recompute the size of the window
			this->treeWidget->resizeColumnToContents(0);
			this->treeWidget->resizeColumnToContents(1);

			if (this->treeWidget->columnWidth(0) < 300)
				this->treeWidget->setColumnWidth(0, 300);
			if (this->treeWidget->columnWidth(2) <  60)
				this->treeWidget->setColumnWidth(2, 60);

			this->setFixedWidth(this->treeWidget->columnWidth(0) + this->treeWidget->columnWidth(2) + 50);
		}


		//----------------------------[ dataSetRemoved ]---------------------------\\

		void SaveDialog::dataSetRemoved(DataSet * ds)
		{
			if (!ds)
				return;

			// Check if the data set has been added before
			if (!(this->dataSets.contains(ds)))
				return;

			// Get the index of the data set
			int dsIndex = this->dataSets.indexOf(ds);

			// Remove the corresponding item from the tree widget, and delete it
			QTreeWidgetItem * currentItem = this->treeWidget->takeTopLevelItem(dsIndex);
			delete currentItem;

			// Remove the data set from the list
			this->dataSets.removeAt(dsIndex);
		}
		//--------------------------------[ setClickedItem ]--------------------------------\\

		void SaveDialog::setClickedItem(QTreeWidgetItem* item, int index)
		{
			//Might be used in the future, leave.
			//qDebug() << "Clicked row and item child count:"<<   this->treeWidget->currentIndex().row()  << ","<< item->childCount() << endl;
			//QTreeWidgetItem *item = this->treeWidget->takeTopLevelItem(this->treeWidget->currentIndex().row());

		}


		//--------------------------------[ saveSelectedItem ]--------------------------------\\

		void SaveDialog::saveSelectedItem()
		{

			bool isFiber(false);
			//Get the selected item and save, if it is a child item return.
			//qDebug() << "Save selected item current row() index:"<< this->treeWidget->currentIndex().row() << endl;
			foreach( QTreeWidgetItem *item, this->treeWidget->selectedItems() ) {

				for( int col = 0; col < item->columnCount(); ++col ) {
					qDebug() << "Item Text [" << col << "]: " << item->text( col ) << " Type:" << item->type() ;
					if(item->childCount() ==0)   {	
						qDebug() << "Item is a child, please select the data itself."<< endl;	
						return; 
					} 
					//else
					//{
					//	for( int i = 0; i < item->childCount(); ++i )  {
					//		for( int col = 0; col < item->columnCount(); ++col )
					//			cout << item->child(i)->text(col).toStdString() ;  cout  << endl;
					//	}
					//}

				}

			}


			DataSet * ds= this->dataSets[this->treeWidget->currentIndex().row()];
			QString name(ds->getName());  
			QString kind(ds->getKind()); 
			QString fileName (name+"-"+kind);
			QString saveFileName;
			
			if(kind!="fibers" && kind!= "seed points" && kind != "regionOfInterest" && kind != "eigen")
			{

				// Why derivatives of DTI like FA is listed as being 0 byte. lets update the volumes so that they will be produced.  
				if(ds->getVtkImageData()) 
				if ((ds->getVtkImageData()->GetActualMemorySize() == 0) && (kind=="scalar volume"))
				 {
				 	ds->getVtkImageData()->Update();
				 }

				saveFileName = QFileDialog::getSaveFileName(this,
					"Save Data as...",
					fileName,	
					"Nifti (*.nii);; VTK Image (*.vti);;VTK Polydata (*.vtp)");
			}
			else if(kind=="eigen")
			{
				 
				saveFileName = QFileDialog::getSaveFileName(this,
					"Save Data as...",
					fileName,	
					"VTK Image (*.vti)");
			}
			else if(kind=="fibers")
			{
				isFiber = true;
				saveFileName = QFileDialog::getSaveFileName(this,
					"Save Data as...",
					fileName,	
					"Fibers (*.fbs);;VTK (*.vtk);;VTK Polydata (*.vtp)");
			}
				else if(kind=="regionOfInterest")
			{
				 
				saveFileName = QFileDialog::getSaveFileName(this,
					"Save Data as...",
					fileName,	
					"VTK Polydata (*.pol);;VTK (*.vtk);;VTK Polydata (*.vtp)");
			}
			else{
				saveFileName = QFileDialog::getSaveFileName(this,
					"Save Data as...",
					fileName,	
					"VTK (*.vtk);;VTK Polydata (*.vtp)");
			}
			if(saveFileName==NULL)
				return;
			QStringRef fileNameExtention(&saveFileName,saveFileName.lastIndexOf(QString(".")),4 );
			this->hide();

			vtkImageData * image   = ds->getVtkImageData();
			vtkPolyData * polyData =  ds->getVtkPolyData();
			vtkObject * obj        =  ds->getVtkObject();

			vtkDataSetWriter *writer = vtkDataSetWriter::New();

			vtkPointSet * pointSet = NULL;

			// Cast the VTK object to a point set pointer
			if (obj) 
			{
				pointSet = vtkPointSet::SafeDownCast(obj);
			}

			if(image && (kind.contains("scalar volume") || kind.contains("DTI") || kind.contains("discrete sphere") || kind.contains("spherical harmonics")  ))// && (ds->getVtkImageData()->GetNumberOfScalarComponents() ==1 ))
            {
                //qDebug() << "Writing the image data. No of scalar components is:" << image->GetNumberOfScalarComponents() << endl;

                if( fileNameExtention.toString()==".vti" )
                {
                    vtkXMLImageDataWriter *writerXML = vtkXMLImageDataWriter::New();                
                    writerXML->SetInput ( (vtkDataObject*)(image) );
                   // writerXML->SetFileTypeToBinary();
					writerXML->SetDataModeToBinary();
                    writerXML->SetFileName( saveFileName.toStdString().c_str() );

                    if(writerXML->Write()) 
						cout << "Writing finished. "<< endl;
					else
					    cout << "Writing error. "<< endl;
                    //save the transfer matrix along with the image
					this->saveTransferMatrix(saveFileName, ds ); 
                    writerXML->Delete();
                }

				else if( fileNameExtention.toString()==".nii" || fileNameExtention.toString()==".gz"  )
				{
	
					this->getManager()->writeDataToFile(saveFileName, ds); // who will decide the data type supported extention writer can decide. Niftiwriter can decide.
				}


			}
			else if(image && (kind.contains("eigen") ) ){	 		 
                    vtkXMLImageDataWriter *writerXML = vtkXMLImageDataWriter::New();                
                    writerXML->SetInput ( (vtkDataObject*)(image) );
                   // writerXML->SetFileTypeToBinary();
					writerXML->SetDataModeToBinary();
                    writerXML->SetFileName( saveFileName.toStdString().c_str() );

                    if(writerXML->Write()) 
						cout << "Writing vti finished. "<< endl;
					else
					    cout << "Writing vti error. "<< endl;
                    //save the transfer matrix along with the image
					this->saveTransferMatrix(saveFileName, ds ); 
                    writerXML->Delete();              

			}
			else if(polyData){	 

				qDebug() << "Writing the polydata data" << endl;
				writer->SetInput( (vtkDataObject*)(polyData) );
				writer->SetFileTypeToASCII();
				writer->SetFileName( saveFileName.toStdString().c_str() );
                this->saveTransferMatrix(saveFileName, ds);
				writer->Write();

			}
			else if(pointSet)
			{		 
				qDebug() << "Exporting the pointset data" << endl;
				writer->SetInput( (vtkDataObject*)(pointSet) );
				writer->SetFileTypeToASCII();
				writer->SetFileName( saveFileName.toStdString().c_str() );
				writer->Write();
			}
			else 
			{
				qDebug() << "The data can not be saved due to data type."<< endl;
				return; 
			}


		}

		//--------------------------------[ saveTransferMatrix ]--------------------------------\\

		void SaveDialog::saveTransferMatrix(const QString saveFileName, data::DataSet * ds )
		{



			vtkObject * attObject;

			//remove if the name includes an extention
			QStringRef fileNameRoot(&saveFileName, 0, saveFileName.lastIndexOf("."));

			// Check if the current treewidget item ie the corresponding data set contains a transformation matrix
			if (ds->getAttributes()->getAttribute("transformation matrix", attObject))
			{
				std::string err = "";
				QString fileNameRootQStr = fileNameRoot.toString()  +".tfm";

				// write the matrix to a ".tfm" file
				bool success = TransformationMatrixIO::writeMatrix(fileNameRootQStr.toStdString() , vtkMatrix4x4::SafeDownCast(attObject), err);

				// Display error messages if necessary
				if (!success)
				{
					qDebug() << err.c_str() ;
				}
			}

		}


		//--------------------------------[ close ]--------------------------------\\

		void SaveDialog::close()
		{
			// Simply hide the dialog window
			this->hide();
		}
		 
	} // namespace gui


} // namespace bmia
