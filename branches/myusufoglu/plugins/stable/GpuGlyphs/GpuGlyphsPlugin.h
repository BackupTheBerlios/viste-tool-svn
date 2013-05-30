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
 * GpuGlyphsPlugin.h
 *
 * 2010-07-13	Tim Peeters
 * - First version
 *
 * 2010-12-16	Evert van Aart
 * - Added support for HARDI SH glyphs.
 * - Implemented "dataSetRemoved" and "dataSetChanged".
 *
 * 2011-01-10	Evert van Aart
 * - Added support for the Cylindrical Harmonics HARDI mapper.
 * - Added fused visualization.
 * - Automatically initialize fiber ODF when selected.
 * - Cleaned up the code, added comments.
 *
 * 2011-02-08	Evert van Aart
 * - Version 1.0.0.
 * - Added support for coloring DTI glyphs.
 *
 * 2011-03-28	Evert van Aart
 * - Version 1.0.1.
 * - Made the two glyph actors non-pickable. This will prevent these actors from
 *   interfering with the Fiber Cutting plugin.
 *
 * 2011-03-29	Evert van Aart
 * - Version 1.0.2.
 * - Fused glyphs now correctly update when moving the planes. 
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.0.3.
 * - Correctly update when seed points are changed.
 *
 * 2011-06-20	Evert van Aart
 * - Version 1.1.0.
 * - Added LUTs for SH glyphs. Removed the range clamping and "SQRT" options for
 *   coloring, as both things can be achieved with the Transfer Functions.
 * - Measure values for SH glyphs coloring are now computed only once per seed
 *   point set and stored in an array, rather than at every render pass. This 
 *   should smoothen the camera movement for SH glyphs.
 * - Implemented "dataSetChanged" and "dataSetRemoved" for transfer functions.
 *
 * 2011-07-07	Evert van Aart
 * - Version 1.1.1.
 * - After rendering the glyphs, put GL options that were disabled/enabled during
 *   rendering back to their original state. In particular, failing to re-enable
 *   blending cause problems with text rendering.
 *
 */


#ifndef bmia_GpuGlyphsPlugin_h
#define bmia_GpuGlyphsPlugin_h


/** Define the UI class */

namespace Ui 
{
    class GpuGlyphsForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkPropAssembly.h>
#include <vtkPointSet.h>
#include <vtkPoints.h>
#include <vtkVolume.h>
#include <vtkUnstructuredGrid.h>
#include <vtkMatrix4x4.h>
#include <vtkColorTransferFunction.h>
#include <vtkScalarsToColors.h>

/** Includes - Custom Files */

#include "vtkThresholdFilter.h"
#include "HARDI/HARDIMeasures.h"

/** Includes - GUI */

#include "ui_GpuGlyphs.h"

/** Includes - Qt */

#include <QDebug>
#include <QMessageBox>


namespace bmia {


/** Forward Class Declarations */

class vtkDTIGlyphMapperVA;
class vtkGlyphMapperVA;
class vtkSHGlyphMapper;
class vtkCHGlyphMapper;

/** This plugin visualizes DTI and HARDI data using glyphs. Both glyph types
	use seed point data sets (stored as "vtkPointSet" objects) to determine the
	position of the glyphs. Eigensystem image data sets are used to create the 
	DTI glyphs; for HARDI glyphs, image data sets containing Spherical Harmonics
	coefficients are used. The plugin can also visualize DTI and HARDI glyphs
	at the same time; in this case, a HARDI measure and a threshold value are
	used to determine which input seed points should have a HARDI glyphs, and 
	which should have a DTI glyphs. GPU mappers are used to draw both glyph types.
*/

class GpuGlyphsPlugin : 	public plugin::AdvancedPlugin,
							public plugin::GUI,
	                        public data::Consumer,
							public plugin::Visualization
							
							
{
    Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::AdvancedPlugin)
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
    
		GpuGlyphsPlugin();

		/** Destructor */

		~GpuGlyphsPlugin();

		/** Initialize the plugin. */

		void init();

		/** Returns the VTK prop that renders all the geometry. This 
			implements the Visualization interface. In this case, the VTK
			prop is an assembly containing props for both the DTI glyphs
			and the HARDI glyphs (for fused visualization). */
    
		vtkProp * getVtkProp();

		/** Returns the Qt widget that gives the user control. This 
			implements the GUI interface. */
    
		QWidget * getGUI();

		/** Define behavior when new data sets are added to the data manager, and
			when existing sets are removed or modified. This defines the consumer 
			interface of the plugin. */

		void dataSetAdded(data::DataSet * ds);
		void dataSetChanged(data::DataSet * ds);
		void dataSetRemoved(data::DataSet * ds);

	protected slots:
    
		/** Set the scale of the glyphs (DTI and HARDI).
			@param scale	Desired scale. */

		void setScale(double scale);

		/** Update the settings of the HARDI glyph mapper. */

		void changeSHSettings();

		/** Clear the array of pre-computed measure scalar values for SH glyphs.
			Called when the coloring options of these glyphs change, as this 
			makes re-computation of these values necessary. */

		void clearHARDIScalars();

		/** Change the coloring method for SH glyphs. By default just calls 
			"changeSHSettings"; however, if a user tries to switch to LUT-based
			coloring while there are no LUTs available, this function will show
			a warning dialog and switching the coloring method back to RGB.
			@param index		Index of the new coloring method. */

		void changeSHColoringMethod(int index);

		/** Enable or disable GUI controls based on the current GUI settings. */

		void enableControls();

		/** Update the DTI glyphs. The optional seed point data set is used
			for fused visualization, when this function is called from "showFused". 
			If "seeds" is NULL, the seed point set defined by the GUI combo
			box is used instead.
			@param seeds	Optional seed points. */

		void changeDTIData(vtkPointSet * seeds = NULL);

		/** Update the HARDI glyphs. The optional seed point data set is used
			for fused visualization, when this function is called from "showFused". 
			If "seeds" is NULL, the seed point set defined by the GUI combo
			box is used instead.
			@param seeds	Optional seed points. */

		void changeHARDIData(vtkPointSet * seeds = NULL);

		/** Show the DTI glyphs, and hide the HARDI glyphs. */

		void showDTIOnly();

		/** Show the HARDI glyphs, and hide the DTI glyphs. */
	
		void showHARDIOnly();

		/** Show both the DTI and HARDI glyphs. A "vtkThresholdFilter" object is used
			to split the input seed point data set into two sets, one for DTI glyphs
			and one for HARDI glyphs. Called when switching to fused visualization,
			and when changing one of the fused visualization options. */

		void showFused();

		/** Called when the seed points change. */

		void seedsChanged();

	private:

		/** Add a DTI Eigensystem image to the plugin. 
			@param ds		New eigensystem data set. */

		void dtiImageAdded(data::DataSet * ds);

		/** Add a Spherical Harmonics image to the plugin. 
			@param ds		New HARDI SH data set. */

		void hardiImageAdded(data::DataSet * ds);

		/** Add a seed point data set to the plugin.
			@param ds		New seed point data set. */

		void seedsAdded(data::DataSet * ds);

		/** Add a scalar volume data set to the plugin.
			@param ds		New scalar volume data set. */

		void scalarsAdded(data::DataSet * ds);

		/** Add a transfer function data set to the plugin.
			@param ds		New transfer function data set. */

		void lutAdded(data::DataSet * ds);

		/** The Qt widget to be returned by "getGUI". */
    
		QWidget * widget;

		/** The Qt form created with Qt Designer. */
    
		Ui::GpuGlyphsForm * ui;

		/** Lists containing all DTI Eigensystem, HARDI Spherical
			Harmonics, seed point, scalar volume and transfer function 
			data sets that have been added to this plugin. */

		QList<data::DataSet *> dtiImageDataSets;
		QList<data::DataSet *> hardiImageDataSets;
		QList<data::DataSet *> seedDataSets;
		QList<data::DataSet *> scalarDataSets;
		QList<data::DataSet *> lutDataSets;

		/** GPU mappers for DTI and HARDI glyphs. */
    
		vtkDTIGlyphMapperVA * DTIMapper;
		vtkSHGlyphMapper * HARDIMapper;

		/** The VTK volumes that do the rendering for DTI and HARDI glyphs. */
    
		vtkVolume * DTIGlyphs;
		vtkVolume * HARDIGlyphs;

		/** The assembly contains both "vtkVolume" objects, allowing them to be
			rendered at the same time (used for fused visualization). */

		vtkPropAssembly * assembly;

		/** Filter used to split the input seed point data set into two sets: one
			for DTI glyphs, and one for HARDI glyphs. */

		vtkThresholdFilter * seedFilter;

		/** Seed points for DTI and HARDI when using fused visualization. */
	
		vtkPointSet * dtiSeeds;
		vtkPointSet * hardiSeeds;

		/** Previously selected HARDI mapper. The plugin supports both the Spherical
			Harmonics mapper, and the Cylindrical Harmonics mapper. This variable is used
			to check for changes in the HARDI mapper type. */

		int prevHARDIMapper;

}; // class GpuGlyphsPlugin


} // namespace bmia


#endif // bmia_GpuGlyphsPlugin_h
