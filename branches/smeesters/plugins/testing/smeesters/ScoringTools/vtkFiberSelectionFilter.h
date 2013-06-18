#ifndef bmia_ConnectivityMeasurePlugin_vtkFiberSelectionFilter_h
#define bmia_ConnectivityMeasurePlugin_vtkFiberSelectionFilter_h


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

#include "ScoringTools.h"


namespace bmia {


/** This class is used to
*/

class vtkFiberSelectionFilter : public vtkPolyDataToPolyDataFilter
{
	public:

		/** Constructor Call */

		static vtkFiberSelectionFilter * New();

		/** VTK Macro */

		vtkTypeMacro(vtkFiberSelectionFilter, vtkPolyDataToPolyDataFilter);

		/** Average score thresholding range */

		void SetAverageScoreRange(double range[2])
		{
            averageScoreRange = range;
		}

		/** Select scalar value type */

		void SetScalarType(int index)
		{
            scalarType = index;
		}

	protected:


		/** Main entry point of the filter. */

		virtual void Execute();

		/** Constructor. */

		vtkFiberSelectionFilter();

		/** Destructor. */

		~vtkFiberSelectionFilter();

		/** Scoring parameters. */

		double* averageScoreRange;

        /** Selected scalar value type */

		int scalarType;

		/** Evaluates if the considered fiber should be included in the output polydata
		    based on scoring thresholding criteria. */

		bool EvaluateFiber(vtkCell* cell, vtkDataArray* inputScalars);

}; // class vtkFiberSelectionFilter


} // namespace bmia


#endif // bmia_ConnectivityMeasurePlugin_vtkFiberSelectionFilter_h
