/*
 * vtk2DROIFiberFilter.h
 *
 * 2010-11-02	Evert van Aart
 * - First version
 *
 * 2010-11-22	Evert van Aart
 * - Fixed an error where the output would contain no fibers if all
 *   ROIs were "NOT" ROIs.
 *
 */


#ifndef bmia_vtk2DROIFiberFilter_h
#define bmia_vtk2DROIFiberFilter_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkPolyDataToPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkDataObject.h>
#include <vtkObjectFactory.h>
#include <vtkPolygon.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkCellData.h>
#include <vtkPointData.h>

/** Includes - STL */

#include <list>


namespace bmia {


/** This class filters a set of input fibers through a number of 
	Regions of Interest (ROIs). ROIs can have the "NOT" property, i.e.,
	fibers cannot pass through this region. Fibers should pass through
	all ROIs that do not have the "NOT" property (AND-rule). If a single
	ROI object contains more than one polygon (which is not the case
	by default), the OR-rule is applied to the individual polygons (i.e.,
	if a fiber passes through one polygon, it has passed through the
	overreaching ROI object. Both input and output are "vtkPolyData" 
	objects; cell- and point data is copied between input and output. */

class vtk2DROIFiberFilter : public vtkPolyDataToPolyDataFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtk2DROIFiberFilter, vtkPolyDataToPolyDataFilter);

		/** Constructor Call */

		static vtk2DROIFiberFilter * New();

		/** Add a Region of Interest to the list.
			@param rROI		New Region of Interest.
			@param rNOT		If true, fibers crossing this ROI are excluded. */

		void addROI(vtkPolyData * rROI, bool rNOT);

		vtkSetMacro(CutFibersAtROI, bool);

	protected:

		/** Constructor */

		vtk2DROIFiberFilter();

		/** Destructor */

		~vtk2DROIFiberFilter();

		/** Point of entry of the filter. */

		void Execute();

	private:

		/** For each ROI, we maintain a structure containing a pointer to the
			original poly data object, a list of polygons constructed from this 
			poly data, and a boolean which indicates whether or not this is
			a "NOT" ROI (i.e., fibers through this ROI are excluded from the output. */

		struct ROISettings
		{
			vtkPolyData * ROI;
			std::list<vtkPolygon *> ROIPolygons;
			bool bNOT;
		};

		/** List of all ROIs that have been added to the filter. Fibers must pass through
			all ROIs in the list for which "bNOT" is false, and may not pass through a
			ROI for which "bNOT" is true. */

		std::list<ROISettings> ROIList;

		/** Create the polygon(s) for a ROI. By default, each ROI contains only one polygon,
			but they can also contains multiple ROIs. All polygons are added to the
			"ROIPolygons" list of the "ROISettings" object. 
			@param newROI	Information about the new ROI. */

		bool createPolygons(ROISettings * newROI);

		/** Write a single fiber to the output data set. Copies point data for each point,
			and cell data for the entire fiber. If "CutFibersAtROI" is true, it also adds
			the "firstX" and "lastX" points to the fiber, and only copies points that lie
			between the first and last ROI.
			@param numberOfPoints	Number of points in the input fiber.
			@param pointList		List of point IDs of the input fiber.
			@param lineId			ID of the input fiber. */

		void writeFiberToOutput(vtkIdType numberOfPoints, vtkIdType * pointList, vtkIdType lineID);

		/** Returns "true" if the line between "p1" and "p2" intersect the polygon stored
			in "currentROI". The point of intersection is returned in the vector "iX".
			@param p1	First point.
			@param p2	Second point.
			@param iX	Point of intersection. */

		bool lineIntersectsROI(double p1[], double p2[], double * iX);

		/** Input/Output poly data (fibers). */

		vtkPolyData * input;
		vtkPolyData * output;

		/** Point data of tyhe input/output. */

		vtkPointData * inPD;
		vtkPointData * outPD;

		/** Cell data of the input/output. */

		vtkCellData * inCD;
		vtkCellData * outCD;

		/** Points/Lines of the output. */

		vtkPoints *    outputPoints;
		vtkCellArray * outputLines;

		/** Polygon of the current ROI. */

		vtkPolygon * currentROI;

		/** If "CutFibersAtROI" is true, and the number of non-"NOT" ROIs is
			at least two, the output fibers will start at the intersection
			point of the first ROI encountered ("firstX"), and end at the 
			intersection point of the last ROI encountered ("lastX"). We 
			store these intersection points, and the IDs of the associated 
			fiber points, in these four variables. */

		double firstX[3];
		double lastX[3];
		vtkIdType firstID;
		vtkIdType lastID;

		/** If "CutFibersAtROI" is true, and the number of non-"NOT" ROIs is
			at least two, the output fibers will start at the intersection
			point of the first ROI encountered ("firstX"), and end at the 
			intersection point of the last ROI encountered ("lastX"). */

		bool CutFibersAtROI;

		/** Number of added ROIs for which "NOT" is true. */

		int numberOfNOTs;

}; // class vtk2DROIFiberFilter


} // namespace bmia


#endif // bmia_vtk2DROIFiberFilter_h