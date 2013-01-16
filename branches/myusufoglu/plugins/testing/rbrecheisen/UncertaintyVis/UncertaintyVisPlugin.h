#ifndef bmia_UncertaintyVis_UncertaintyVisPlugin_h
#define bmia_UncertaintyVis_UncertaintyVisPlugin_h

#include "DTITool.h"

class vtkConfidenceIntervalMapper;
class vtkFiberConfidenceMapper;
class vtkConfidenceHistogram;
class vtkConfidenceInterval;
class vtkROIWidget;
class vtkActor;

class QPushButton;
class QSlider;
class QLabel;
class QCheckBox;
class QComboBox;
class QFrame;

class QConfidenceHistogramWidget;

namespace bmia
{
	class UncertaintyVisPlugin :	public plugin::AdvancedPlugin,
									public plugin::GUI,
									public plugin::Visualization
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::AdvancedPlugin )
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::GUI )
		Q_INTERFACES( bmia::plugin::Visualization )

	public:

		UncertaintyVisPlugin();
		virtual ~UncertaintyVisPlugin();

		vtkProp * getVtkProp();
		QWidget * getGUI();

	private slots:

		void loadHeader();
		void roiEnabled( bool );
		void blurringEnabled( bool );
		void noiseEnabled( bool );
		void activePropertyChanged( const QString );
		void renderModeChanged( const QString );
		void histogramOpacityChanged( int );
		void intervalRangeChanged( int );
		void intervalSliderReleased();
		void intervalChanged();
		void intervalSelected( int );
		void colorSelected();
		void outlineColorSelected();
		void baseColorSelected();
		void autoColorSchemeSelected( const QString );
		void staircaseApplied();
		void invertedStaircaseApplied();
		void equalizeToSelectedApplied();
		void subdivisionChanged( const QString );
		void numberOfIntervalsChanged( int );
		void saveProperties();
		void loadProperties();
		void loadTransform();
		void disableSelected();
		void makeScreenshot();

		void loadDAT();

		void startTiming();

	private:

		vtkFiberConfidenceMapper * mapper;
		vtkConfidenceHistogram * histogram;
		vtkConfidenceInterval * interval;
		vtkROIWidget * roiWidget;
		vtkActor * actor;

		QWidget * widget;
		QPushButton * loadHeaderButton;
		
		QSlider * histogramOpacitySlider;

		QCheckBox   * blurringCheckBox;
		QCheckBox   * noiseCheckBox;
		QCheckBox   * invertCheckBox;
		QCheckBox   * roiCheckBox;
		QComboBox   * activePropertyComboBox;
		QComboBox   * renderModeComboBox;
		QPushButton * selectColorButton;
		QPushButton * selectOutlineColorButton;
		QComboBox   * autoColorComboBox;
		QPushButton * selectBaseColorButton;
		QPushButton * applyStaircaseButton;
		QPushButton * applyInvertedStaircaseButton;
		QPushButton * applyEqualizeToSelectedButton;
		QPushButton * disableSelectedButton;
		QPushButton * savePropertiesButton;
		QPushButton * loadPropertiesButton;
		QComboBox   * subdivisionComboBox;
		QComboBox   * numberOfIntervalsComboBox;

		QLabel  * rangeLabel;
		QSlider * rangeSlider;

		QFrame * colorPatch;
		QFrame * baseColorPatch;
		QFrame * outlineColorPatch;

		QConfidenceHistogramWidget * histogramWidget;

		int selectedInterval;
		bool initialized;
	};
}

#endif