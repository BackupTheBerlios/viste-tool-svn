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

typedef struct
{
    bool set;
    double averageScore[2];
} ThresholdSettings;

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

//		/** Average score thresholding range */
//
//		void SetAverageScoreRange(double range[2])
//		{
//            averageScoreRange = range;
//		}
//
//		/** Select active scalar value type */
//
//		void SetScalarType(int index)
//		{
//            scalarType = index;
//		}

//        void SetThresholdSettings(QList<ThresholdSettings*> settings)
//        {
//            thresholdSettings = settings;
//        }

        void AddThresholdSetting(bool set, double range[2])
        {
            ThresholdSettings* ts = new ThresholdSettings;
            ts->set = set;
            ts->averageScore[0] = range[0];
            ts->averageScore[1] = range[1];
            thresholdSettings.append(ts);
        }

	protected:



		/** Main entry point of the filter. */

		virtual void Execute();

		/** Constructor. */

		vtkFiberSelectionFilter();

		/** Destructor. */

		~vtkFiberSelectionFilter();

		/** Scoring parameters. */

        QList<ThresholdSettings*> thresholdSettings;
		double* averageScoreRange;

        /** Selected scalar value type */

		int scalarType;

		/** Evaluates if the considered fiber should be included in the output polydata
		    based on scoring thresholding criteria. */

		bool EvaluateFiber(vtkCell* cell, vtkPointData* inputPD);

}; // class vtkFiberSelectionFilter


} // namespace bmia


#endif // bmia_ConnectivityMeasurePlugin_vtkFiberSelectionFilter_h
