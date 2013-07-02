#ifndef bmia_ConnectivityMeasurePlugin_vtkFiberScoringMeasuresFilter_h
#define bmia_ConnectivityMeasurePlugin_vtkFiberScoringMeasuresFilter_h


/** Includes - VTK */

#include <vtkPolyDataToPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkObjectFactory.h>
#include <vtkDataArray.h>
#include <vtkCellArray.h>
#include <vtkCell.h>
#include <vtkIdList.h>

/** Includes - Qt */

#include <QMap>

/** Includes - Custom Files */

#include "ScoringMeasuresTypes.h"

namespace bmia {

/** This class is used to
*/

class vtkFiberScoringMeasuresFilter : public vtkPolyDataToPolyDataFilter
{
	public:

		/** Constructor Call */

		static vtkFiberScoringMeasuresFilter * New();

		/** VTK Macro */

		vtkTypeMacro(vtkFiberScoringMeasuresFilter, vtkPolyDataToPolyDataFilter);


	protected:

		/** Main entry point of the filter. */

		virtual void Execute();

		/** Constructor. */

		vtkFiberScoringMeasuresFilter();

		/** Destructor. */

		~vtkFiberScoringMeasuresFilter();

}; // class vtkFiberScoringMeasuresFilter


} // namespace bmia


#endif // bmia_ConnectivityMeasurePlugin_vtkFiberScoringMeasuresFilter_h
