// Includes plugin
#include <DicomConverterPlugin.h>
#include <core/Core.h>

// Includes QT
#include <QFileDialog>
#include <QSettings>

// Includes DTIConv2
#include <DTIConfig.h>
#include <DTIDICOMReader.h>
#include <DTIGradientReader.h>
#include <DTIData2TensorConverter2.h>
#include <DTITensor2DtiToolConverter.h>
#include <DTIBootstrapConverter.h>
#include <DTIDICOMSliceGroup.h>
#include <DTISliceGroup.h>

// Includes GDCM
#include <gdcmSystem.h>
#include <gdcmGlobal.h>
#include <gdcmDict.h>
#include <gdcmDicts.h>
#include <gdcmPrivateTag.h>

// Includes GSL
#include <gsl/gsl_matrix.h>

// Includes STL
#include <vector>

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	DicomConverterPlugin::DicomConverterPlugin() :
		plugin::Plugin( "DicomConverterPlugin" ),
		plugin::GUI()
	{
		// Set configuration object to NULL
		_config = 0;

		// Create QT widget that will hold plugin's GUI
		_widget = new QWidget;

		// Create UI form that describes plugin's GUI, then
		// initialize widget with the form
		_form = new Ui::DicomConverterForm;
		_form->setupUi( _widget );

		// Setup signal/slot connections
		this->setupConnections();
	}

	///////////////////////////////////////////////////////////////////////////
	DicomConverterPlugin::~DicomConverterPlugin()
	{
		// Delete QT objects
		delete _widget;
		delete _form;

		// Delete configuration object. It is possible that it
		// is already NULL (if some loading went wrong) so
		// check for this
		if( _config != 0 ) delete _config;
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * DicomConverterPlugin::getGUI()
	{
		return _widget;
	}

	///////////////////////////////////////////////////////////////////////////
	void DicomConverterPlugin::setupConnections()
	{
		// Setup signal/slot connections
		this->connect( _form->buttonLoadConfig, SIGNAL( clicked() ), this, SLOT( loadConfig() ) );
		this->connect( _form->buttonConvert,    SIGNAL( clicked() ), this, SLOT( convert() ) );
	}

	///////////////////////////////////////////////////////////////////////////
	void DicomConverterPlugin::loadConfig()
	{
		// Show file dialog to let user select configuration file
		QString path = QFileDialog::getOpenFileName( 0, "Load Configuration File", "." );
		if( path.isNull() || path.isEmpty() )
			return;

		// If configuration object is not NULL, delete it
		// Then create new instance
		if( _config != 0 )
			delete _config;
		_config = new DTIConfig;

		// Create new DTI config object that allows us to load
		// the configuration file
		if( _config->LoadFile( path.toStdString().c_str())  ==  false )
		{
			std::cout << "DicomConverterPlugin::loadConfig() ";
			std::cout << "Could not open configuration file" << std::endl;
			delete _config;
			_config = 0;
			return;
		}

		_config->PrintKeyValuePairs( std::cout );

		std::cout << "DicomConverterPlugin::loadConfig() ";
		std::cout << "Successfully loaded config file" << std::endl;
	}

	///////////////////////////////////////////////////////////////////////////
	void DicomConverterPlugin::convert()
	{
		// Check we have a valid configuration object
		if( _config == 0 )
			return;

		// Get file format and check that it's the right one
		QString format = QString( _config->GetKeyValue( "Data.File.Format" ) );
		Q_ASSERT( ! format.isEmpty() );

		if( format.toLower() != "dicom" )
		{
			std::cout << "DicomConverterPlugin::convert() ";
			std::cout << "Illegal format provided. Must be DICOM!" << std::endl;
			return;
		}

		// Get scanner manufacturer. We need this because depending on
		// the manufacturer we set certain default parameters
		QString manufacturer = QString( _config->GetKeyValue( "Data.Manufacturer" ) );
		Q_ASSERT( ! manufacturer.isEmpty() );

		// Create DICOM reader object
		DTIDICOMReader * reader = new DTIDICOMReader;

		bool numberConversionOk = false;

		// If manufacturer is Philips we need some specific handling of the data.
		// If not, we use the default converting scheme
		if( manufacturer.toLower() == "philips" )
		{
			// We need to specify explicitly that the reader should
			// behave as a Philips reader. Note that the attributes being read in
			// part of the routine are not mandatory. The software has default
			// values for these attributes. However, in case Philips comes up
			// with different values suddenly, you have the option of providing
			// these values in the configuration file

			reader->SetPhilips( true );

			// Get Philips-specific image angulation anterior-posterior
			QString angulationAPGroup   = QString( _config->GetKeyValue( "Dicom.Tag.GroupID.ImageAngulationAP" ) );
			QString angulationAPElement = QString( _config->GetKeyValue( "Dicom.Tag.ElementID.ImageAngulationAP" ) );

			if( ! angulationAPGroup.isEmpty() && ! angulationAPElement.isEmpty() )
			{
				reader->SetImageAngulationAPGroupID  ( angulationAPGroup.toUInt() );
				reader->SetImageAngulationAPElementID( angulationAPElement.toUInt() );
			}

			// Get Philips-specific image angulation feet-head
			QString angulationFHGroup   = QString( _config->GetKeyValue( "Dicom.Tag.GroupID.ImageAngulationFH" ) );
			QString angulationFHElement = QString( _config->GetKeyValue( "Dicom.Tag.ElementID.ImageAngulationFH" ) );

			if( ! angulationFHGroup.isEmpty() && ! angulationFHElement.isEmpty() )
			{
				reader->SetImageAngulationFHGroupID  ( angulationFHGroup.toUInt() );
				reader->SetImageAngulationFHElementID( angulationFHElement.toUInt() );
			}

			// Get Philips-specific image angulation right-left
			QString angulationRLGroup   = QString( _config->GetKeyValue( "Dicom.Tag.GroupID.ImageAngulationRL" ) );
			QString angulationRLElement = QString( _config->GetKeyValue( "Dicom.Tag.ElementID.ImageAngulationRL" ) );

			if( ! angulationRLGroup.isEmpty() && ! angulationRLElement.isEmpty() )
			{
				reader->SetImageAngulationRLGroupID  ( angulationRLGroup.toUInt() );
				reader->SetImageAngulationRLElementID( angulationRLElement.toUInt() );
			}
		}

		// Get file path. This is really just the path, without the filename
		QString filePath = QString( _config->GetKeyValue("Data.File.Path") );
		Q_ASSERT( ! filePath.isEmpty() );
		reader->SetFilePath( filePath );

		// Get file prefix. This is the part of the filename that is the
		// same for all files in the set. We do not allow the prefix to be
		// empty. This would mean the filenames consist only of numbers
		QString filePrefix = QString( _config->GetKeyValue("Data.File.Prefix") );
		Q_ASSERT( ! filePrefix.isEmpty() );
		reader->SetFilePrefix( filePrefix );

		// Get file extension. It is possible that the files have no extension.
		// The user can indicate this in the config file by not specifying this
		// attribute at all, or setting it to '.'
		QString fileExtension = QString( _config->GetKeyValue("Data.File.Extension") );
		if( fileExtension.isNull() || fileExtension == "." )
			fileExtension = "";
		reader->SetFileExtension( fileExtension );

		// Get index of the first file in the dataset
		numberConversionOk = false;
		QString fileFirstIndex = QString( _config->GetKeyValue("Data.File.FirstIndex") );
		Q_ASSERT( ! fileFirstIndex.isEmpty() );
		int firstIndex = fileFirstIndex.toInt( & numberConversionOk );
		Q_ASSERT( numberConversionOk );
		reader->SetFileFirstIndex( firstIndex );

		// Get index of last file in the dataset
		numberConversionOk = false;
		QString fileLastIndex = QString( _config->GetKeyValue("Data.File.LastIndex") );
		Q_ASSERT( ! fileLastIndex.isEmpty() );
		int lastIndex = fileLastIndex.toInt( & numberConversionOk );
		Q_ASSERT( numberConversionOk );
		reader->SetFileLastIndex( lastIndex );

		// Get number of digits used for the file numbering. This will allow
		// us to do the right zero-padding later on. If the number of digits is
		// zero then the files have no zero-padding
		numberConversionOk = false;
		QString fileNumberOfDigits = QString( _config->GetKeyValue("Data.File.NumberOfDigits") );
		Q_ASSERT( ! fileNumberOfDigits.isEmpty() );
		int numberOfDigits = fileNumberOfDigits.toInt( & numberConversionOk );
		Q_ASSERT( numberConversionOk );
		reader->SetFileNumberOfDigits( numberOfDigits );

		// Get ordering. Usually files are ordered by filename, that is, the
		// filename ordering corresponds to the ordering in space. Sometimes
		// this is not the case and the ordering can be retrieved from the
		// instance number
		QString fileOrderedByInstanceNumber = QString( _config->GetKeyValue("Data.File.OrderedByInstanceNumber") );
		bool orderedByInstanceNumber = false;
		if( fileOrderedByInstanceNumber.toLower() == "true" )
			orderedByInstanceNumber = true;
		reader->SetOrderedByInstanceNumber( orderedByInstanceNumber );

		// Load the DICOM data
		if( reader->LoadData() == false )
		{
			std::cout << "DicomConverterPlugin::convert() ";
			std::cout << "Could not load DICOM data!" << std::endl;
			delete reader;
			return;
		}

		// Print info
		reader->PrintInfo( std::cout );

		// Create gradient reader for reading a gradient text file
		DTIGradientReader * gradientReader = new DTIGradientReader;

		// Check if the gradients are stored inside the DICOM files or
		// in a separate text file
		QString gradientsInDicom = QString( _config->GetKeyValue( "Data.GradientsInDicom" ) );

		if( gradientsInDicom.isEmpty() || gradientsInDicom.toLower() == "false" )
		{
			// Get gradient file path
			QString gradientFilePath = QString( _config->GetKeyValue("Gradients.File.Path") );
			Q_ASSERT( ! gradientFilePath.isEmpty() );
			gradientReader->SetFilePath( gradientFilePath );

			// Get gradient file name
			QString gradientFileName = QString( _config->GetKeyValue( "Gradients.File.Name" ) );
			Q_ASSERT( ! gradientFileName.isEmpty() );
			gradientReader->SetFileName( gradientFileName );

			// Get number of gradients
			numberConversionOk = false;
			QString tmp = QString( _config->GetKeyValue( "Gradients.Count" ) );
			Q_ASSERT( ! tmp.isEmpty() );
			int numberOfGradients = tmp.toInt( & numberConversionOk );
			Q_ASSERT( numberConversionOk );
			gradientReader->SetNumberOfGradients( numberOfGradients );

			// Load the gradients
			if( gradientReader->LoadData() == false )
			{
				std::cout << "DicomConverterPlugin::convert() ";
				std::cout << "Could not load DICOM data!" << std::endl;
				delete gradientReader;
				delete reader;
				return;
			}
		}
		else if( gradientsInDicom.toLower() == "true" )
		{
			QString gradientXGroup   = QString( _config->GetKeyValue( "Dicom.Tag.GroupID.GradientX" ) );
			Q_ASSERT( ! gradientXGroup.isEmpty() );
			reader->SetGradientXGroupID( gradientXGroup.toUInt() );

			QString gradientXElement = QString( _config->GetKeyValue( "Dicom.Tag.ElementID.GradientX" ) );
			Q_ASSERT( ! gradientXElement.isEmpty() );
			reader->SetGradientXElementID( gradientXElement.toUInt() );

			QString gradientYGroup = QString( _config->GetKeyValue( "Dicom.Tag.GroupID.GradientY" ) );
			Q_ASSERT( ! gradientYGroup.isEmpty() );
			reader->SetGradientYGroupID( gradientYGroup.toUInt() );

			QString gradientYElement = QString( _config->GetKeyValue( "Dicom.Tag.ElementID.GradientY" ) );
			Q_ASSERT( ! gradientYElement.isEmpty() );
			reader->SetGradientYElementID( gradientYElement.toUInt() );

			QString gradientZGroup = QString( _config->GetKeyValue( "Dicom.Tag.GroupID.GradientZ" ) );
			Q_ASSERT( ! gradientZGroup.isEmpty() );
			reader->SetGradientZGroupID( gradientZGroup.toUInt() );

			QString gradientZElement = QString( _config->GetKeyValue( "Dicom.Tag.ElementID.GradientZ" ) );
			Q_ASSERT( ! gradientZElement.isEmpty() );
			reader->SetGradientZElementID( gradientZElement.toUInt() );

			// Get number of gradients
			numberConversionOk = false;
			QString tmp = QString( _config->GetKeyValue( "Gradients.Count" ) );
			Q_ASSERT( ! tmp.isEmpty() );
			int numberOfGradients = tmp.toInt( & numberConversionOk );
			Q_ASSERT( numberConversionOk );

			// Let DICOM reader load the gradients
			if( reader->LoadGradients( numberOfGradients ) == false )
			{
				std::cout << "DicomConverterPlugin::convert() ";
				std::cout << "Could not load gradients from DICOM data!" << std::endl;
				delete gradientReader;
				delete reader;
				return;
			}

			// Pass gradients to gradient reader for further processing
			gradientReader->SetData( reader->GetGradients(), numberOfGradients );
		}
		else
		{
			std::cout << "DicomConverterPlugin::convert() ";
			std::cout << "Data.GradientsInDicom has illegal value!" << std::endl;
			delete gradientReader;
			delete reader;
			return;
		}

		// Optionally flip the X-axis of each gradient direction
		QString flipX = QString( _config->GetKeyValue( "Gradients.FlipX" ) );
		if( flipX.toLower() == "true" )
			gradientReader->FlipX();

		// Optionally flip the Y-axis of each gradient direction
		QString flipY = QString( _config->GetKeyValue( "Gradients.FlipY" ) );
		if( flipY.toLower() == "true" )
			gradientReader->FlipY();

		// Optionally flip the Z-axis of each gradient direction
		QString flipZ = QString( _config->GetKeyValue( "Gradients.FlipZ" ) );
		if( flipZ.toLower() == "true" )
			gradientReader->FlipZ();

		// Apply manufacturer-specific flips
		if( manufacturer.toLower() == "philips" )
		{
			// Philips requires a Z-flip by default
			gradientReader->FlipZ();
		}
		else if( manufacturer.toLower() == "ge" )
		{
			// GE requires a Y-flip by default
			gradientReader->FlipY();
		}

		// Apply the transforms to the gradients based on the image
		// position patient, image orientation patient and the optional
		// axis flips
		gsl_matrix * gradientTransform = reader->GetGradientTransform();
		gradientReader->Transform( gradientTransform );
		gradientReader->Normalize();

		// Get the transformed gradients
		gsl_matrix * gradients = gradientReader->GetOutput();

		// Get output type for conversion
		QString outputType = QString( _config->GetKeyValue( "Output.Type" ) );
		Q_ASSERT( ! outputType.isEmpty() );

		if( outputType.toLower() == "dtitool" )
		{
			// Create data to tensor converter
			DTIData2TensorConverter2 * converter = new DTIData2TensorConverter2;

			// Get number of B0 slices in the dataset
			QString numberB0Slices = QString( _config->GetKeyValue( "SliceGroup.B0Slice.Count" ) );
			Q_ASSERT( ! numberB0Slices.isEmpty() );
			converter->SetNumberOfB0Slices( numberB0Slices.toInt() );

			// Get first B0 slice index
			QString firstB0Index = QString( _config->GetKeyValue( "SliceGroup.B0Slice.FirstIndex" ) );
			Q_ASSERT( ! firstB0Index.isEmpty() );
			converter->SetB0SliceFirstIndex( firstB0Index.toInt() );

			// Get mask enabling attribute
			QString maskEnabled = QString( _config->GetKeyValue( "Data.MaskEnabled" ) );
			Q_ASSERT( ! maskEnabled.isEmpty() );
			if( maskEnabled.toLower() == "true" )
				converter->SetMaskEnabled( true );
			else
				converter->SetMaskEnabled( false );

			// Get mask value
			QString maskValue = QString( _config->GetKeyValue( "Data.MaskValue" ) );
			Q_ASSERT( ! maskValue.isEmpty() );
			converter->SetMaskValue( maskValue.toDouble() );

			// Get B-value
			QString bValue = QString( _config->GetKeyValue( "Data.BValue" ) );
			Q_ASSERT( ! bValue.isEmpty() );
			converter->SetBValue( bValue.toDouble() );

			// Pass output of dicom reader to converter
			converter->SetInput( (std::vector< DTISliceGroup * > *) reader->GetOutput() );
			converter->SetGradients( gradients );

			// Run the converter
			if( converter->Execute() == false )
			{
				std::cout << "DicomConverterPlugin::convert() ";
				std::cout << "Could not convert DICOM data!" << std::endl;
				delete gradientReader;
				delete reader;
				delete converter;
				return;
			}

			// Create tensor to DTI tool converter
			DTITensor2DtiToolConverter * outputConverter = new DTITensor2DtiToolConverter;
			outputConverter->SetInput( converter->GetOutput() );

			// Run the converter
			if( outputConverter->Execute() == false )
			{
				std::cout << "DicomConverterPlugin::convert() ";
				std::cout << "Could not convert to DTI tool!" << std::endl;
				delete gradientReader;
				delete reader;
				delete converter;
				delete outputConverter;
				return;
			}

			// Get output data type
			QString dataType = QString( _config->GetKeyValue( "Output.DTITool.DataType" ) );
			Q_ASSERT( ! dataType.isEmpty() );
			outputConverter->SetDataType( dataType );

			// Pass pixel spacing and slice thickness to output converter
			double * pixelSpacing = reader->GetPixelSpacing();
			double sliceThickness = reader->GetSliceThickness();
			outputConverter->SetPixelSpacing( pixelSpacing );
			outputConverter->SetSliceThickness( sliceThickness );

			// Set DTI tool version
			outputConverter->SetVersion( 2 );

			// Write tensor data to file
			if( outputConverter->Write() == false )
			{
				std::cout << "DicomConverterPlugin::convert() ";
				std::cout << "Could not write DTI tool data!" << std::endl;
				delete gradientReader;
				delete reader;
				delete converter;
				delete outputConverter;
				return;
			}

			std::cout << "DicomConverterPlugin::convert() ";
			std::cout << "Successfully converted DICOM to tensor data!" << std::endl;
		}
		else if( outputType.toLower() == "hardi" )
		{
			std::cout << "DicomConverterPlugin::convert() ";
			std::cout << "Successfully converted DICOM to hardi data!" << std::endl;
		}
		else if( outputType.toLower() == "bootstrap" )
		{
			// Create wild bootstrap converter. This converter reads in the
			// internally stored DICOM files, fit the tensor model, compute
			// the residuals of the fit and use them to generate multiple
			// random variations of the data
			DTIBootstrapConverter * converter = new DTIBootstrapConverter;

			// Set the converter's pixel spacing and slice thickness
			double * spacing = reader->GetPixelSpacing();
			converter->SetPixelSpacing( spacing[0], spacing[1] );
			converter->SetSliceThickness( reader->GetSliceThickness() );

			// Get number of B0 slices in the dataset
			QString numberB0Slices = QString( _config->GetKeyValue( "SliceGroup.B0Slice.Count" ) );
			Q_ASSERT( ! numberB0Slices.isEmpty() );
			converter->SetNumberOfB0Slices( numberB0Slices.toInt() );

			// Get first B0 slice index
			QString firstB0Index = QString( _config->GetKeyValue( "SliceGroup.B0Slice.FirstIndex" ) );
			Q_ASSERT( ! firstB0Index.isEmpty() );
			converter->SetB0SliceFirstIndex( firstB0Index.toInt() );

			// Get mask enabling attribute
			QString maskEnabled = QString( _config->GetKeyValue( "Data.MaskEnabled" ) );
			Q_ASSERT( ! maskEnabled.isEmpty() );
			if( maskEnabled.toLower() == "true" )
				converter->SetMaskEnabled( true );
			else
				converter->SetMaskEnabled( false );

			// Get mask value
			QString maskValue = QString( _config->GetKeyValue( "Data.MaskValue" ) );
			Q_ASSERT( ! maskValue.isEmpty() );
			converter->SetMaskValue( maskValue.toDouble() );

			// Get B-value
			QString bValue = QString( _config->GetKeyValue( "Data.BValue" ) );
			Q_ASSERT( ! bValue.isEmpty() );
			converter->SetBValue( bValue.toDouble() );

			// Get number of bootstrap volumes
			QString numberBootstraps = QString( _config->GetKeyValue( "Output.Bootstrap.NrVolumes" ) );
			Q_ASSERT( ! numberBootstraps.isEmpty() );
			converter->SetNumberOfBootstrapVolumes( numberBootstraps.toInt() );

			// Get starting index. This is the index that will be appended to the
			// first tensor volume generated. The other will be incrementally
			// numbered
			QString bootstrapFirstIndex = QString( _config->GetKeyValue( "Output.Bootstrap.FirstIndex" ) );
			Q_ASSERT( ! bootstrapFirstIndex.isEmpty() );
			converter->SetStartIndex( bootstrapFirstIndex.toInt() );

			// Pass output of dicom reader to converter
			converter->SetInput( (std::vector< DTISliceGroup * > *) reader->GetOutput() );
			converter->SetGradients( gradients );

			// Run the converter
			if( converter->Execute() == false )
			{
				std::cout << "DicomConverterPlugin::convert() ";
				std::cout << "Could not generate bootstrap tensor datasets!" << std::endl;
				delete gradientReader;
				delete reader;
				delete converter;
				return;
			}

			std::cout << "DicomConverterPlugin::convert() ";
			std::cout << "Successfully converted DICOM to many bootstrap datasets!" << std::endl;
		}
		else
		{
			std::cout << "DicomConverterPlugin::convert() ";
			std::cout << "Unknown output type!" << std::endl;
			delete gradientReader;
			delete reader;
			return;
		}
	}
}

Q_EXPORT_PLUGIN2( libDicomConverterPlugin, bmia::DicomConverterPlugin )
