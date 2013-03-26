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
// temporary
#include "vtkTransform.h"

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



				// Why derivatices of DTI like FA is listed as being 0 byte. lets update
				if (ds->getVtkImageData()->GetActualMemorySize() == 0)
				{
					ds->getVtkImageData()->Update();
					//this->manager->dataSetChanged(ds);
				}
				// Print data set type and dimensionality
				this->addSubItem(dsItem, "Type: Image (" + QString::number(image->GetDataDimension()) + "D)");
				this->addSubItem(dsItem, "Components Per Voxel: " + QString::number(image->GetNumberOfScalarComponents()) + "");
				this->addSubItem(dsItem, "Scalar Type: " + QString::number(image->GetScalarType()) + "");
				if(image->GetDataDimension() == 3)
					this->addSubItem(dsItem, "Spacing: " + QString::number(image->GetSpacing()[0]) + " "+ QString::number(image->GetSpacing()[1]) + " " + QString::number(image->GetSpacing()[2]));
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
					qDebug() << "Item Text [" << col << "]: " << item->text( col ) << " Type:" << item->type() ;
					if(item->childCount() ==0)   {	
						qDebug() << "Item is a child, please select the data itself."<< endl;	
						return; 
					} 
					else
					{
						cout << "printing child columns" << endl;
						for( int i = 0; i < item->childCount(); ++i )  {
							for( int col = 0; col < item->columnCount(); ++col )
								cout << item->child(i)->text(col).toStdString() ;  cout  << endl;
						}
					}

				}

			}



			cout << "this->treeWidget->currentIndex().row():" << this->treeWidget->currentIndex().row() << " ";
			cout << this->treeWidget->currentIndex().column() << endl;

			DataSet * ds= this->dataSets[this->treeWidget->currentIndex().row()];
			QString name(ds->getName());  
			QString kind(ds->getKind()); 
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
			QStringRef fileNameExtention(&saveFileName,saveFileName.lastIndexOf(QString(".")),4 );
			this->hide();

			cout << "saveFileName:" << saveFileName.toStdString() << endl;
			cout << fileNameExtention.toString().toStdString() << endl;
			//cin.get();


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

			if(image && kind.contains("scalar volume"))// && (ds->getVtkImageData()->GetNumberOfScalarComponents() ==1 ))
			{
				qDebug() << "Writing the image data. No of scalar components is:" << image->GetNumberOfScalarComponents() << endl;
				//image->Print(cout);

				if((fileNameExtention.toString()==".vtk") || fileNameExtention.toString()==".vti" )
				{
					cout << "saving vtk or vti" << endl;
					writer->SetInput ( (vtkDataObject*)(image) );
					//save the transfer matrix along with the image
					writer->SetFileTypeToBinary();
					writer->SetFileName( saveFileName.toStdString().c_str() );
					writer->Write();
					this->saveTransferMatrix(saveFileName, ds ); 
				}
				else if( fileNameExtention.toString()==".nii" || fileNameExtention.toString()==".gz"  )
				{
					cout << "saving nifti" << endl;
					vtkObject * attObject = vtkObject::New();
			cout << "Get attribute transf mat. "<< endl;
			ds->getAttributes()->getAttribute("transformation matrix", attObject);
			 
					this->setNiftiFields(image,saveFileName.toStdString().c_str(),ds);
				//	this->getManager()->writeDataToFile(saveFileName, ds); // who will decide the data type supported extention writer can decide. Niftiwriter can decide.
				}


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
				writer->SetFileTypeToASCII();
				writer->SetFileName( saveFileName.toStdString().c_str() );
				writer->Write();
			}
			else 
			{
				qDebug() << "The data can not be saved due to data type."<< endl;	
				return; 
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
		// will be moved to
		void SaveDialog::setNiftiFields(vtkImageData * image, const QString saveFileName, data::DataSet *ds )
		{


			nifti_image * m_NiftiImage = new nifti_image;
			m_NiftiImage = nifti_simple_init_nim();
			//print for debug
			image->Print(cout);
			double dataTypeSize = 1.0;
			int dim[3];
			int wholeExtent[6];
			double spacing[3];
			double origin[3];
			image->Update();
			int numComponents = image->GetNumberOfScalarComponents();
			int imageDataType = image->GetScalarType();

			image->GetOrigin(origin);
			image->GetSpacing(spacing);
			image->GetDimensions(dim);
			image->GetWholeExtent(wholeExtent);
			m_NiftiImage->dt = 0;

			m_NiftiImage->ndim = 3;
			m_NiftiImage->dim[1] = wholeExtent[1] + 1;
			m_NiftiImage->dim[2] = wholeExtent[3] + 1;
			m_NiftiImage->dim[3] = wholeExtent[5] + 1;
			m_NiftiImage->dim[4] = 1;
			m_NiftiImage->dim[5] = 1;
			m_NiftiImage->dim[6] = 1;
			m_NiftiImage->dim[7] = 1;
			m_NiftiImage->nx =  m_NiftiImage->dim[1];
			m_NiftiImage->ny =  m_NiftiImage->dim[2];
			m_NiftiImage->nz =  m_NiftiImage->dim[3];
			m_NiftiImage->nt =  m_NiftiImage->dim[4];
			m_NiftiImage->nu =  m_NiftiImage->dim[5];
			m_NiftiImage->nv =  m_NiftiImage->dim[6];
			m_NiftiImage->nw =  m_NiftiImage->dim[7];

			//nhdr.pixdim[0] = 0.0 ;
			m_NiftiImage->pixdim[1] = spacing[0];
			m_NiftiImage->pixdim[2] = spacing[1];
			m_NiftiImage->pixdim[3] = spacing[2];
			m_NiftiImage->pixdim[4] = 0;
			m_NiftiImage->pixdim[5] = 1;
			m_NiftiImage->pixdim[6] = 1;
			m_NiftiImage->pixdim[7] = 1;
			m_NiftiImage->dx = m_NiftiImage->pixdim[1];
			m_NiftiImage->dy = m_NiftiImage->pixdim[2];
			m_NiftiImage->dz = m_NiftiImage->pixdim[3];
			m_NiftiImage->dt = m_NiftiImage->pixdim[4];
			m_NiftiImage->du = m_NiftiImage->pixdim[5];
			m_NiftiImage->dv = m_NiftiImage->pixdim[6];
			m_NiftiImage->dw = m_NiftiImage->pixdim[7];

			int numberOfVoxels = m_NiftiImage->nx;

			if(m_NiftiImage->ny>0){
				numberOfVoxels*=m_NiftiImage->ny;
			}
			if(m_NiftiImage->nz>0){
				numberOfVoxels*=m_NiftiImage->nz;
			}
			if(m_NiftiImage->nt>0){
				numberOfVoxels*=m_NiftiImage->nt;
			}
			if(m_NiftiImage->nu>0){
				numberOfVoxels*=m_NiftiImage->nu;
			}
			if(m_NiftiImage->nv>0){
				numberOfVoxels*=m_NiftiImage->nv;
			}
			if(m_NiftiImage->nw>0){
				numberOfVoxels*=m_NiftiImage->nw;
			}

			m_NiftiImage->nvox = numberOfVoxels;

			if(numComponents==1 || numComponents==6 ){
				switch(imageDataType)
				{
				case VTK_BIT://DT_BINARY:
					m_NiftiImage->datatype = DT_BINARY;
					m_NiftiImage->nbyper = 0;
					dataTypeSize = 0.125;
					break;
				case VTK_UNSIGNED_CHAR://DT_UNSIGNED_CHAR:
					m_NiftiImage->datatype = DT_UNSIGNED_CHAR;
					m_NiftiImage->nbyper = 1;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_SIGNED_CHAR://DT_INT8:
					m_NiftiImage->datatype = DT_INT8;
					m_NiftiImage->nbyper = 1;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_SHORT://DT_SIGNED_SHORT:
					m_NiftiImage->datatype = DT_SIGNED_SHORT;
					m_NiftiImage->nbyper = 2;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_UNSIGNED_SHORT://DT_UINT16:
					m_NiftiImage->datatype = DT_UINT16;
					m_NiftiImage->nbyper = 2;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_INT://DT_SIGNED_INT:
					m_NiftiImage->datatype = DT_SIGNED_INT;
					m_NiftiImage->nbyper = 4;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_UNSIGNED_INT://DT_UINT32:
					m_NiftiImage->datatype = DT_UINT32;
					m_NiftiImage->nbyper = 4;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_FLOAT://DT_FLOAT:
					m_NiftiImage->datatype = DT_FLOAT;
					m_NiftiImage->nbyper = 4;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_DOUBLE://DT_DOUBLE:
					m_NiftiImage->datatype = DT_DOUBLE;
					m_NiftiImage->nbyper = 8;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_LONG://DT_INT64:
					m_NiftiImage->datatype = DT_INT64;
					m_NiftiImage->nbyper = 8;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				case VTK_UNSIGNED_LONG://DT_UINT64:
					m_NiftiImage->datatype = DT_UINT64;
					m_NiftiImage->nbyper = 8;
					dataTypeSize = m_NiftiImage->nbyper;
					break;
				default:
					cout << "cannot handle this type" << endl ;
					break;
				}
			}
			// m_NiftiImage->data = image->GetPointData( // scalar pointer i ekle buraya !!!! yer ac? 
			m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;
			m_NiftiImage->data=const_cast<void *>( image->GetScalarPointer());

			m_NiftiImage->fname = nifti_makehdrname( saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0);
			m_NiftiImage->iname = nifti_makeimgname(saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0); // 0 is compressed
			m_NiftiImage->qform_code = 1;

			//Transformation and quaternion
			vtkObject * attObject = vtkObject::New();
			cout << "Get attribute transf mat. "<< endl;
			ds->getAttributes()->getAttribute("transformation matrix", attObject);
			cout << "Get attribute transf mat. OK "<< endl;
			if(attObject)
			{
				cout << "attObject"  << endl;
				vtkMatrix4x4 *matrix =  vtkMatrix4x4::New();
					matrix = vtkMatrix4x4::SafeDownCast(attObject);
				if(matrix)
				{
				matrix->Print(cout);
				cout << "attObject 1.1"  << endl;
				mat44 matrixf;
				for(int i=0;i<4;i++)
					for(int j=0;j<4;j++)
					{
						matrixf.m[i][j] = matrix->GetElement(i,j);
						cout <<  matrixf.m[i][j] << endl;
					}
					cout << "attObject 1.2"  << endl;
					nifti_mat44_to_quatern(matrixf, &( m_NiftiImage->quatern_b), &( m_NiftiImage->quatern_c), &( m_NiftiImage->quatern_d), 
						&( m_NiftiImage->qoffset_x), &(m_NiftiImage->qoffset_y), &(m_NiftiImage->qoffset_z), &(m_NiftiImage->dx) , &(m_NiftiImage->dy) ,&(m_NiftiImage->dz) , &(m_NiftiImage->qfac));


					cout << m_NiftiImage->quatern_b << " " << m_NiftiImage->quatern_c << " " << m_NiftiImage->quatern_d << " " << m_NiftiImage->qfac << " " << endl;
					cout << m_NiftiImage->qoffset_x << " " << m_NiftiImage->qoffset_y << " " << m_NiftiImage->qoffset_z <<endl;

					// in case the matrix is not pure transform, quaternion can not include scaling part. Therefore if the matris is not a pure transform matrix use scaling factor in spacing?
					float scaling[3];
					if(matrix->Determinant() != 1)
					{
						cout << "Determinant not 1. Find scaling." << endl;
						vtkTransform *transform = vtkTransform::New();
						transform->SetMatrix(matrix);
						transform->Scale(scaling);

						m_NiftiImage->pixdim[1] = spacing[0]*scaling[0];
						m_NiftiImage->pixdim[2] = spacing[1]*scaling[1];
						m_NiftiImage->pixdim[3] = spacing[2]*scaling[2];
						transform->Delete();
					}
				}
				else {
					cout << "Invalid   matrix \n";
				}
			}
			else
			{
				cout << "Invalid transformation object \n";
			}
			nifti_set_iname_offset(m_NiftiImage);
			// Write the image fiel 
			nifti_image_write( m_NiftiImage );
		}

	} // namespace gui


} // namespace bmia
