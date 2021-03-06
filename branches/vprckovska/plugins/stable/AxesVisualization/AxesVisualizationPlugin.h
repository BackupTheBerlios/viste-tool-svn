/*
 * AxesVisualizationPlugin.h
 *
 * 2010-05-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2011-04-04	Evert van Aart
 * - Version 1.0.0.
 * - Completely new approach to drawing the axes: The axes widget is now added to
 *   the medical canvas by means of a "vtkOrientationMarkerWidget".
 *
 * 2011-06-08	Evert van Aart
 * - Version 1.0.1.
 * - Fixed a bug that caused crashes when this plugin was unloaded.
 *
 */


#ifndef bmia_AxesVisualization_AxesVisualizationPlugin_h
#define bmia_AxesVisualization_AxesVisualizationPlugin_h


/** Define the UI class */

namespace Ui 
{
	class AxesVisualizationForm;
} 


/** Includes - VTK */

#include <vtkAxesActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkCaptionActor2D.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkMatrix4x4.h>
#include <vtkMath.h>

/** Includes - Custom Files */

#include "gui/MetaCanvas/vtkMedicalCanvas.h"
#include "gui/MetaCanvas/vtkSubCanvas.h"
#include "gui/MetaCanvas/vtkMetaCanvasUserEvents.h"

/** Includes - GUI */

#include "ui_AxesVisualization.h"

/** Includes - Qt */

#include <QDebug>
#include <QString>
#include <QList>

/** Includes - C++ */

#include <string>

/** Includes - Main Header */

#include "DTITool.h"


namespace bmia {


class AxesVisualizationPlugin :	public plugin::AdvancedPlugin, 
								public plugin::GUI, 
								public data::Consumer
{
    Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::AdvancedPlugin)
	Q_INTERFACES(bmia::plugin::GUI)
	Q_INTERFACES(bmia::data::Consumer)

	public:

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */

		AxesVisualizationPlugin();

		/** Destructor */

		~AxesVisualizationPlugin();

		/** Initialize the plugin. */

		void init();

	void dataSetAdded(data::DataSet * ds);
	void dataSetChanged(data::DataSet * ds);
	void dataSetRemoved(data::DataSet * ds);

	QWidget * getGUI();

	void subcanvassesResized();

protected slots:

	void setPosToTL();
	void setPosToTR();
	void setPosToBR();
	void setPosToBL();
	void setPosToC();
	void changeSize(int newSize);
	void changeVisibility(bool show);
	void settingsToGUI();
	void showAll();
	void hideAll();
	void setTransformationMatrix();
	void setApplyTransformation(bool apply);

private: 
	
	QWidget * qWidget;
	Ui::AxesVisualizationForm * ui;

	class AxesCallback : public vtkCommand
	{
		public:

		/** Constructor Call */

		static AxesCallback * New() { return new AxesCallback; }

		/** Execute the callback. 
			@param caller		Not used.
			@param event		Event ID.
			@param callData		For BMIA_USER_EVENT_SUBCANVASSES_RESIZED, this
								will always be NULL. */

		void Execute(vtkObject * caller, unsigned long event, void * callData);

		/** Pointer to the axes visualization plugin. */

		AxesVisualizationPlugin * plugin;
};

	AxesCallback * callBack;

	enum MarkerPosition
	{
		MPOS_BL = 0,
		MPOS_TL,
		MPOS_TR,
		MPOS_BR,
		MPOS_C
	};

	struct AxesInfo
	{
		vtkAxesActor * actor;
		vtkOrientationMarkerWidget * marker;
		vtkSubCanvas * subcanvas;
		bool isVisible;
		double size;
		MarkerPosition pos;
		int matrixIndex;
		bool applyTransformation;
	};

	QList<AxesInfo> infoList;

	void updateMarker(AxesInfo info);

	void computeViewPort(double * mainViewPort, double * outViewPort, AxesInfo info);

	QList<vtkMatrix4x4 *> uniqueMatrices;
	QList<data::DataSet *> matrixDataSets;

	bool isMatrixUnique(vtkMatrix4x4 * m);

	bool areMatricesEqual(vtkMatrix4x4 * a, vtkMatrix4x4 * b);

	void initializeMatrices();

}; // class AxesVisualizationPlugin
} // namespace bmia
#endif // bmia_AxesVisualization_AxesVisualizationPlugin_h
