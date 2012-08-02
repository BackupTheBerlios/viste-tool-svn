/*
 * GeometryGlyphsPlugin.h
 *
 * 2011-04-20	Evert van Aart
 * - Version 1.0.0.
 * - First version
 *
 * 2011-05-09	Evert van Aart
 * - Version 1.1.0.
 * - Added additional support for coloring the glyphs.
 * - Added a glyph builder for Spherical Harmonics data.
 *
 * 2011-08-05	Evert van Aart
 * - Version 1.1.1.
 * - Fixed a major error in the computation of the unit vectors.
 * - Builder parameters are now correctly set when switching input data set.
 *
 */


#ifndef bmia_GeometryGlyphsPlugin_h
#define bmia_GeometryGlyphsPlugin_h


/** Define the UI class */

namespace Ui 
{
	class GeometryGlyphsForm;
}

/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_GeometryGlyphs.h"

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkMatrix4x4.h>
#include <vtkLookupTable.h>
#include <vtkScalarsToColors.h>

/** Includes - Qt */

#include <QMessageBox>

/** Includes - Custom Files */

#include "vtkGeometryGlyphBuilder.h"
#include "vtkGeometryGlyphFromSHBuilder.h"


namespace bmia {


class GeometryGlyphsPlugin : 	public plugin::Plugin,
								public data::Consumer,
								public plugin::Visualization,
								public plugin::GUI
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)
	Q_INTERFACES(bmia::plugin::Visualization)
	Q_INTERFACES(bmia::plugin::GUI)

	public:

		/** Current Version */

		QString getPluginVersion()
		{
			return "1.1.1";
		}

		/** Constructor */

		GeometryGlyphsPlugin();

		/** Destructor */

		~GeometryGlyphsPlugin();

		/** Initialize the plugin. */

		void init();

		/** Returns the VTK prop that renders all the geometry. This 
			implements the Visualization interface. */
    
		vtkProp * getVtkProp();

		/** Returns the Qt widget that gives the user control. This 
			implements the GUI interface. */
    
		QWidget * getGUI();

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

		void dataSetAdded(data::DataSet * ds);
    
		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. */

		void dataSetChanged(data::DataSet * ds);

		/** The data manager calls this function whenever an existing
			data set is removed.
			@param ds	Modified data set. */
   
		void dataSetRemoved(data::DataSet * ds);

	protected slots:

		/** Called when the selected discrete sphere function volume changes.
			@param index		Index of the new volume. */

		void inputDataChanged(int index);

		/** Called when the selected seed point set changes.
			@param index		Index of the new seed point set. */

		void seedDataChanged(int index);

		/** Change the normalization method for the glyphs.
			@param index		Index of the new normalization method. */

		void setNormalizationMethod(int index);

		/** Change the normalization scope for the glyphs.
			@param index		Index of the new normalization scope. */

		void setNormalizationScope(int index);

		/** Change the global scale of the glyphs.
			@param scale		New glyph scale. */

		void setScale(double scale);

		/** Change the exponent used for sharpening the glyphs. 
			@param exponent		New sharpening exponent. */

		void setSharpeningExponent(double exponent);

		/** Change the type of the glyphs (e.g., mesh or star).
			@param index		Index of the new glyph type. */

		void setGlyphType(int index);

		/** Turn glyph normalization on or off.	
			@param enable		Enable or disable glyph normalization. */

		void enableNormalization(bool enable);

		/** Turn glyph sharpening on or off.	
			@param enable		Enable or disable glyph sharpening. */

		void enableSharpening(bool enable);
		
		/** Turn glyph smoothing on or off.	
			@param enable		Enable or disable glyph smoothing. */

		void enableSmoothing(bool enable);

		/** Update the options for glyph smoothing. Called when the user clicks
			the "Update" button underneath the smoothing options. */

		void updateSmoothOptions();

		/** Change the coloring method of the geometry glyphs.
			@param index		Index of the desired coloring method. */

		void changeColorMethod(int index);

		/** Set the scalar volume used for coloring the glyphs.
			@param index		Index of the scalar volume data set. 
			@param update		If false, function will not call "render". */

		void setScalarVolume(int index, bool update = true);

		/** Set the Look-Up Table (Transfer Function) used for coloring the glyphs.
			@param index		Index of the LUT data set. */

		void setLUT(int index);

		/** Set the order of tessellation when using gemoetry glyphs for SH data. */

		void setTessellationOrder(int val);

	private:

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt Designer. */

		Ui::GeometryGlyphsForm * ui;

		/** List containing all discrete sphere function volumes that can be used
			to draw geometry glyphs. */

		QList<data::DataSet *> glyphDataSets;

		/** List of all available seed point data sets. */

		QList<data::DataSet *> seedDataSets;

		/** List of all scalar volume data sets (used for coloring). */

		QList<data::DataSet *> scalarDataSets;

		/** List of all available transfer functions (LUTs). */

		QList<data::DataSet *> lutDataSets;

		/** Geometry glyph builder, used to construct the polydata representing 
			the glyphs, using a discrete sphere function volume and a set of seed
			points as its input. */

		vtkGeometryGlyphBuilder * builder;

		/** Smoothing filter, used to smooth the output of "builder". */

		vtkSmoothPolyDataFilter * smoothFilter;

		/** Actor representing the glyphs. */

		vtkActor * actor;

		/** Polydata mapper used to draw the glyphs. */

		vtkPolyDataMapper * mapper;

}; // class GeometryGlyphsPlugin


} // namespace bmia


#endif // bmia_GeometryGlyphsPlugin_h
