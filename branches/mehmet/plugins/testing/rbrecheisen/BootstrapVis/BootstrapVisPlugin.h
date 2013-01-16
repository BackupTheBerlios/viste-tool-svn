#ifndef bmia_BootstrapVis_BootstrapVisPlugin_h
#define bmia_BootstrapVis_BootstrapVisPlugin_h

#include "DTITool.h"

#include <QString>
#include <QMap>

#include <vector>
#include <string>

class vtkAssembly;
class vtkActor;
class vtkPolyData;
class vtkTubeFilter;
class vtkColor4;

class QComboBox;
class QSlider;
class QLabel;

namespace bmia
{
	class vtkFiberConfidenceMapper;
	class vtkFiberTubeMapper;
	class vtkDistanceTable;

	class BootstrapVisPlugin :	public plugin::Plugin,
								public plugin::Visualization,
								public plugin::GUI,
								public data::Consumer
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::Visualization )
		Q_INTERFACES( bmia::plugin::GUI )
		Q_INTERFACES( bmia::data::Consumer )

	public:

		BootstrapVisPlugin();
		~BootstrapVisPlugin();

		vtkProp * getVtkProp();
		QWidget * getGUI();

		void dataSetAdded( data::DataSet * ds );
		void dataSetChanged( data::DataSet * ds );
		void dataSetRemoved( data::DataSet * ds );

	private slots:

		void loadTable();
		void showMainFibers( int );
		void showConfidenceFibers( int );
		void showMainFibersAsStreamtubes( int );
		void selectMainFiberColor();
		void selectFillColor();
		void selectLineColor();
		void selectDensityColor();
		void enableDensityColoring( int );
		void enableDensitySmoothing( int );
		void enableDensityWeighting( int );
		void enableErosion( int );
		void levelChanged( int );
		void levelSelected( int );
		void showAsMainFibers();
		void showAsBootstrapFibers();
		void updateFibers();

	private:

		vtkPolyData * mainFibersPtr;
		vtkPolyData * mainFibers;
		vtkPolyData * bootstrapFibers;
		vtkAssembly * actor;
		vtkActor * mainActor;
		vtkActor * bootstrapActor;
		vtkFiberTubeMapper * mainMapper;
		vtkFiberConfidenceMapper * bootstrapMapper;
		vtkDistanceTable * table;
		vtkTubeFilter * tubeFilter;

		QMap< QString, data::DataSet * > datasets;
		std::vector< std::pair< std::string, float > > * levels;
		std::vector< std::pair< float, vtkColor4 > > * fillColors;
		std::vector< std::pair< float, vtkColor4 > > * lineColors;

		QComboBox * dataBox;
		QComboBox * levelBox;
		QWidget * widget;
		QSlider * slider;
		QLabel * sliderLabel;

		int selectedLevel;
	};
}

#endif
