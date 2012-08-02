#ifndef bmia_ImageTypeConverter_ImageTypeConverterPlugin_h
#define bmia_ImageTypeConverter_ImageTypeConverterPlugin_h

// Includes DTI tool
#include <DTITool.h>

// Includes QT
#include <QtGui>

// Includes VTK
#include <vtkTransform.h>
#include <vtkImageData.h>

namespace bmia
{
	class ImageTypeConverterPlugin :	public plugin::Plugin,
										public data::Consumer,
										public plugin::GUI
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::data::Consumer )
		Q_INTERFACES( bmia::plugin::GUI )

	public:

		/** Constructor and destructor */
		ImageTypeConverterPlugin();
		virtual ~ImageTypeConverterPlugin();

		/** Returns plugin's QT widget */
		QWidget * getGUI();

		/** Handle dataset events */
		void dataSetAdded  ( data::DataSet * dataset );
		void dataSetRemoved( data::DataSet * dataset );
		void dataSetChanged( data::DataSet * dataset );

	private slots:

		/** Executes conversion */
		void convert();

        /** Pads dataset to nearest power of two */
        void pad();

		/** Saves dataset to .VOL format */
		void save();

	private:

        /** Returns next power of two */
        int NextPowerOfTwo( int n );

		QComboBox   * _datasetBox;			// Combobox containing names of datasets
		QComboBox   * _typeBox;				// Combobox containing data types
		QPushButton * _button;				// Button for starting coordinate transformation
        QPushButton * _buttonSave;          // Button for saving dataset to .VOL
        QPushButton * _buttonPad;           // Button for padding dataset to nearest power of two
		QVBoxLayout * _layout;				// The layout of the widget
		QWidget     * _widget;				// Widget containing UI

		QString _kind;
		QList< vtkImageData * > _datasets;
	};
}

#endif
