/*
 * vtkFiberRankingFilter.h
 *
 * 2011-05-13	Evert van Aart
 * - First version.
 *
 * 2011-08-22	Evert van Aart
 * - Fixed a bug caused by incorrect iteration through the fiber map.
 * - Added a progress bar.
 *
 */


#ifndef bmia_ConnectivityMeasurePlugin_vtkFiberRankingFilter_h
#define bmia_ConnectivityMeasurePlugin_vtkFiberRankingFilter_h


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

#include "ConnectivityMeasuresPlugin.h"


namespace bmia {


/** This class is used to rank fibers based on their connectivity measure values.
	The filter will select only the X 'strongest' fibers (or the Y% 'strongest'
	fiber), where the strength of a fiber depends on the CM values. For example,
	the strength can be defined as the average CM value, or as the CM value at 
	the last fiber point (which is especially useful for region-to-region
	connectivity). The input of this filter should be the output of a 
	connectivity measure filter (i.e., a subclass of "vtkGenericConnectivity-
	MeasureFilter".
*/

class vtkFiberRankingFilter : public vtkPolyDataToPolyDataFilter
{
	public:

		/** Constructor Call */

		static vtkFiberRankingFilter * New();

		/** VTK Macro */

		vtkTypeMacro(vtkFiberRankingFilter, vtkPolyDataToPolyDataFilter);

		/** Set the measure used to determines the strength of a fiber. 
			@param rM		Desired ranking measure. */

		void setMeasure(ConnectivityMeasuresPlugin::RankingMeasure rM)
		{
			measure = rM;
		}

		/** Set the output method (e.g., output only the best X fibers or
			the best Y% of fibers). Should not be "RO_AllFibers", since this
			filter will simply be bypassed if we should output all fibers. 
			@param rO		Desired output method. */

		void setOutputMethod(ConnectivityMeasuresPlugin::RankingOutput rO)
		{
			outputMethod = rO;
		}

		/** Set the number of fibers to be added to the output. Only relevant if
			the output method is "RO_BestNumber".
			@param rNOF		Desired number of fibers. */

		void setNumberOfFibers(int rNOF)
		{
			numberOfFibers = rNOF;
		}

		/** Set the percentage of input fibers (1-100) to be added to the output.
			Only relevant if the output method is "RO_BestPercentage".
			@param rPerc	Desired percentage of input fibers. */

		void setPercentage(int rPerc)
		{
			percentage = rPerc;
		}

		/** Set whether or not to use a single scalar for each point of a fiber.
			@param rUse		Use single scalar value. */

		void setUseSingleValue(bool rUse)
		{
			useSingleValue = rUse;
		}

	protected:

		/** Measure used for determining the strength of a fiber. Uses the Connectivity
			Measure values computed for each fiber point by the connectivity measure 
			filter. The ranking measure determines which CM value is uses for ranking
			purposes, e.g., the CM value of the last fiber point or the average CM value. */

		ConnectivityMeasuresPlugin::RankingMeasure measure;

		/** output method (e.g., output only the best X fibers or the best Y% of 
			fibers). Should not be "RO_AllFibers", since this filter will simply 
			be bypassed if we should output all fibers. */
			
		ConnectivityMeasuresPlugin::RankingOutput outputMethod;

		/** Number of fibers to be added to the output. */

		int numberOfFibers;

		/** Percentage of input fibers to be added to the output. */

		int percentage;

		/** If true, the scalar value used for ranking the fibers is copied to
			every point of the fiber; otherwise, the input values (one unique 
			value per fiber point) are copied to the output. */

		bool useSingleValue;

		/** Main entry point of the filter. */

		virtual void Execute();

		/** Constructor. */

		vtkFiberRankingFilter();

		/** Destructor. */

		~vtkFiberRankingFilter();

}; // class vtkFiberRankingFilter


} // namespace bmia


#endif // bmia_ConnectivityMeasurePlugin_vtkFiberRankingFilter_h