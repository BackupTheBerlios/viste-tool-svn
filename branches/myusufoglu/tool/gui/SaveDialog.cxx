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
* is the main function. 
* 
*/


#include "SaveDialog.h"
#include  "D:\vISTe\subversion\myusufoglu\libs\NIfTI\vtkNiftiWriter.h"


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
				// Print data set type and dimensionality
				this->addSubItem(dsItem, "Type: Image (" + QString::number(image->GetDataDimension()) + "D)");

				// Get and print image dimensions
				int dims[3]; 
				image->GetDimensions(dims);
				this->addSubItem(dsItem, "Dimensions: " +	QString::number(dims[0]) + " x " + 
					QString::number(dims[1]) + " x " + 
					QString::number(dims[2]));

				// Get and print scalar range
				double range[2];
				image->GetScalarRange(range);
				this->addSubItem(dsItem, "Range: " + QString::number(range[0]) + ", " + QString::number(range[1]));

				// Print scalar type
				this->addSubItem(dsItem, "Scalar Type: " + QString(image->GetScalarTypeAsString()));

				// Print the number of scalar components
				this->addSubItem(dsItem, "Components: " + QString::number(image->GetNumberOfScalarComponents()));

				// Get and print the voxel spacing
				double spacing[3]; 
				image->GetSpacing(spacing);
				this->addSubItem(dsItem, "Spacing: " +	QString::number(spacing[0]) + ", " + 
					QString::number(spacing[1]) + ", " + 
					QString::number(spacing[2]));

				// Get the memory size
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
			//qDebug() << index << endl;
			qDebug() << "Clicked row and item child count:"<<   this->treeWidget->currentIndex().row()  << ","<< item->childCount() << endl;
			//QTreeWidgetItem *item = this->treeWidget->takeTopLevelItem(this->treeWidget->currentIndex().row());



		}


		//--------------------------------[ saveSelectedItem ]--------------------------------\\

		void SaveDialog::saveSelectedItem()
		{

			bool isFiber(false);
			//Get the selected item and save, if it is a child item return.
			qDebug() << "Save selected item current row() index:"<< this->treeWidget->currentIndex().row() << endl;
			foreach( QTreeWidgetItem *item, this->treeWidget->selectedItems() ) {

				for( int col = 0; col < item->columnCount(); ++col ) {
					//qDebug() << "Item Text [" << col << "]: " << item->text( col ) << " Type:" << item->type() ;
					if(item->childCount() ==0)   {	
						qDebug() << "Item is a child, please select the data itself."<< endl;	
						return; 
					} 

				}

			}

		

			DataSet * ds= this->dataSets[this->treeWidget->currentIndex().row()];
			QString name(ds->getName()); // << endl;
			QString kind(ds->getKind());// << endl;
			QString fileName (name+"-"+kind);
			QString saveFileName;
			// fiber selection must be automaticly fbs.
			if(kind!="fibers")
			{
			
				saveFileName = QFileDialog::getSaveFileName(this,
				"Save Data as...",
				fileName,	
				"VTK (*.vtk);;Nifti (*.nii);; Nifti (*.nii.gz);; VTK Image (*.vti);;VTK Polydata (*.vtp)");
			}
			else
			{
	              	isFiber = true;
				saveFileName = QFileDialog::getSaveFileName(this,
				"Save Data as...",
				fileName,	
				"Fibers (*.fbs);;VTK (*.vtk);;VTK Polydata (*.vtp)");
			}
				if(saveFileName==NULL)
				return;
				QStringRef fileNameExtention(&saveFileName, saveFileName.lastIndexOf("."), saveFileName.length());
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

			if(image){
				qDebug() << "Writing the image data" << endl;
				
				if(fileNameExtention.toString().compare(QString("vtk")) )
				{
					writer->SetInput ( (vtkDataObject*)(image) );
				    //save the transfer matrix along with the image
				    this->saveTransferMatrix(saveFileName, ds );
				}
//				else(fileNameExtention.toString().compare(QString("nii")) )
//				{
//					        Reader* reader = NULL;
//                          int i = 0;
//                          QStringList ext1;
//				        	while ((reader == NULL) )//&& i < this->readers. )
//							
//							{
////								ext1 = this->manager->readers.at(i)->getSupportedFileExtensions();
//								for (int j=0; j < ext1.size(); j++)
//								{
//									if ( saveFileName.fileName().endsWith( QString(ext1.at(j)) ) )
//									{
////										reader = this->manager->readers.at(i);
//									}
//								} // for j
//								i++;
//							} //  
//								
//							//reader->sa
//				} 

				

			}
			else if(polyData){	 
			 
				qDebug() << "Writing the polydata data" << endl;
				writer->SetInput( (vtkDataObject*)(polyData) );
				if(isFiber) this->saveTransferMatrix(saveFileName, ds);

				
			}
			else if(pointSet)
			{		 
				qDebug() << "Writing the pointset data" << endl;
				writer->SetInput( (vtkDataObject*)(pointSet) );
			}
			else 
			{
				qDebug() << "The data can not be saved due to data type."<< endl;	
				return; 
			}
			if(fileNameExtention.toString().compare(QString("nii")) || fileNameExtention.toString().compare(QString("nii.gz")) && image )
			{
				this->setNiftiFields(image,saveFileName.toStdString().c_str());
			}
			else{
//				{
			writer->SetFileTypeToASCII();
			writer->SetFileName( saveFileName.toStdString().c_str() );
			writer->Write();
			//writer->Delete();
			}

			
			//image->Delete();
			//polyData->Delete();
			//obj->Delete();


			//writer->Update();

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
				cout << fileNameRootQStr.toStdString() << endl;
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
		//--------------------------------[ setNiftiFields ]--------------------------------\\

		void SaveDialog::setNiftiFields(vtkImageData * image, const QString saveFileName )
		{
			
			cout << "write file nifti " << saveFileName.toStdString() <<  endl;
			vtkNIfTIWriter *writer = vtkNIfTIWriter::New();
			std::ofstream *file = new std::ofstream();
			writer->SetFileType(1);
			writer->SetInputConnection(image->GetProducerPort());
			writer->SetFileName(saveFileName.toStdString().c_str());
			writer->SetFileDimensionality(3);
			writer->WriteFile(file,image,image->GetExtent(),image->GetWholeExtent());
			//writer->Update();
			//writer->Update();
			writer->Delete();
			/*
			nifti_image * outImage = new nifti_image;

			outImage->dim[3] =     outImage->nz			= image->GetDimensions()[2];
			outImage->pixdim[3] = outImage->dz = static_cast<float>( image->GetSpacing()[2] );
			outImage->nvox *=     outImage->dim[3];

			outImage->dim[2] =     outImage->ny = image->GetDimensions()[1];
			outImage->pixdim[2] =  outImage->dy = static_cast<float>( image->GetSpacing()[1] );
			outImage->nvox *=  outImage->dim[2];

			outImage->dim[1] =     outImage->nx = image->GetDimensions()[0];
			outImage->pixdim[1] = outImage->dx = static_cast<float>( image->GetSpacing()[0] );
			outImage->nvox *=     outImage->dim[1];
			QString HeaderFileName= "C:/Users/MYusufog/Desktop/test";
			outImage->iname= (char *) saveFileName.toStdString().c_str() ;
			outImage->fname = (char *) HeaderFileName.toStdString().c_str()  ;
			outImage->data =  image->GetScalarPointer();
			outImage->nbyper =  image->GetScalarSize();
			outImage->datatype =   image->GetScalarType();
			if (image->GetNumberOfScalarComponents() > 4    || image->GetNumberOfScalarComponents() == 2)
				return;
		

			*/

		}

	} // namespace gui


} // namespace bmia
