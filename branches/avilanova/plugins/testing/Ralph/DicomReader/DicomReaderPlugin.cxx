// Includes DTI tool
#include <DicomReaderPlugin.h>
#include <core/Core.h>

// Includes GDCM
#include <gdcmSystem.h>
#include <gdcmDirectory.h>
#include <gdcmScanner.h>
#include <gdcmDataSet.h>
#include <gdcmAttribute.h>

// Includes VTK
#include <vtkStringArray.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkImageData.h>
#include <vtkMath.h>

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	DicomReaderPlugin::DicomReaderPlugin() : plugin::Plugin( "DicomReaderPlugin" ),
		plugin::GUI(), data::Reader()
	{
		_widget = new QWidget;
	}

	///////////////////////////////////////////////////////////////////////////
	DicomReaderPlugin::~DicomReaderPlugin()
	{
		for( int i = 0; i < _readers.size(); ++i )
			_readers.at( i )->Delete();
		_readers.clear();
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * DicomReaderPlugin::getGUI()
	{
		return _widget;
	}

	///////////////////////////////////////////////////////////////////////////
	QStringList DicomReaderPlugin::getSupportedFileExtensions()
	{
		QStringList list;
		list.append( "dcm" );
		return list;
	}

	///////////////////////////////////////////////////////////////////////////
	QStringList DicomReaderPlugin::getSupportedFileDescriptions()
	{
		QStringList list;
		list.append( "DICOM files" );
		return list;
	}

	///////////////////////////////////////////////////////////////////////////
	void DicomReaderPlugin::loadDataFromFile( QString fileName )
	{
		Q_ASSERT( ! fileName.isEmpty() );

		// We assume one of the DICOM files in the directory was selected.
		// Obviously, we wish to load all DICOM files in the directory so
		// we strip the filename from the path
		int index = fileName.lastIndexOf( "/" );
		if( index < 0 )
			index = fileName.lastIndexOf( "\\" );
		Q_ASSERT( index > -1 );
		QString tmp = fileName;
		QString dirName = tmp.remove( index, fileName.length() - index );
		Q_ASSERT( gdcm::System::FileIsDirectory( dirName.toStdString().c_str() ) );

		gdcm::Directory dir;
		dir.Load( dirName.toStdString().c_str() );
		gdcm::Directory::FilenamesType const & fileNames = dir.GetFilenames();

		gdcm::Scanner scanner;
		scanner.AddTag( gdcm::Tag( 0x0020, 0x000d ) ); // Study Instance UID
		scanner.AddTag( gdcm::Tag( 0x0020, 0x000e ) ); // Series Instance UID
		scanner.AddTag( gdcm::Tag( 0x0010, 0x0010 ) ); // Patient's Name
		scanner.AddTag( gdcm::Tag( 0x0020, 0x0013 ) ); // Instance Number

		bool b = scanner.Scan( fileNames );
		Q_ASSERT( b );

		vtkStringArray * list = vtkStringArray::New();
		gdcm::Directory::FilenamesType dicomNames = scanner.GetKeys();
		gdcm::Directory::FilenamesType::const_iterator i = dicomNames.begin();
		for( ; i != dicomNames.end(); ++i )
		{
			list->InsertNextValue( (*i).c_str() );
		}

		vtkGDCMImageReader * reader = vtkGDCMImageReader::New();
//		this->core()->out()->startProgress(reader);
		reader->SetFileNames( list );
		reader->Update();

		Q_ASSERT( reader->GetOutput() );
//		this->core()->out()->stopProgress(reader);

		double position[3];
		reader->GetImagePositionPatient( position[0], position[1], position[2] );
		double orientation[3][3];
		reader->GetImageOrientationPatient( 
			orientation[0][0], orientation[0][1], orientation[0][2],
			orientation[1][0], orientation[1][1], orientation[1][2] );
		vtkMath::Cross( orientation[0], orientation[1], orientation[2] );

		vtkMatrix4x4 * rotation = vtkMatrix4x4::New();
		rotation->Identity();
		for( int i = 0; i < 3; ++i )
			for( int j = 0; j < 3; ++j )
				rotation->SetElement( i, j, orientation[i][j] );
		rotation->Transpose();

		vtkMatrix4x4 * translation = vtkMatrix4x4::New();
		translation->Identity();
		translation->SetElement( 0, 3, position[0] );
		translation->SetElement( 1, 3, position[1] );
		translation->SetElement( 2, 3, position[2] );

		vtkMatrix4x4 * compound = vtkMatrix4x4::New();
		vtkMatrix4x4::Multiply4x4( rotation, translation, compound );

		vtkTransform * transform = vtkTransform::New();
		transform->SetMatrix( compound );

		data::DataSet * data = new data::DataSet( fileName, "Dicom", reader->GetOutput() );
		data->getAttributes()->addAttribute( "transformation matrix",
				vtkObject::SafeDownCast( transform ) );
		this->core()->data()->addDataSet( data );

		QString text = "Loaded DICOM data from directory: ";
		text.append( dirName );
		this->core()->out()->showMessage( text );

		// WARNING: We should *not* delete the reader! Somehow, this
		// invalidates the image data

		_readers.append( reader );
		list->Delete();
	}
}

Q_EXPORT_PLUGIN2( libDicomReaderPlugin, bmia::DicomReaderPlugin )
