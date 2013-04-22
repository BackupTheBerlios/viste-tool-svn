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

/**
 * FiberOutput.cxx
 *
 * 2008-01-17	Jasper Levink
 * - First version
 *
 * 2010-12-15	Evert van Aart
 * - Ported to DTITool3, refactored some code
 *
 */


#ifndef bmia_FiberOutput_h
#define bmia_FiberOutput_h


/** Includes - Custom Files */

#include "TensorMath/ScalarMeasures.h"

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkPointData.h> 
#include <vtkCellArray.h> 
#include <vtkCellData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkObject.h>

/** Includes - C++ */

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <set>
#include <vector>

/** Includes - Qt */

#include <QProgressDialog>


namespace bmia {


/** Output information about fibers or Regions of Interest. The information
	can consist of various scalar measures, tensor values, tensor eigenvectors,
	fiber length, and fiber volume. For each information field, the mean and
	variance can be calculated. This superclass takes care of collecting the 
	data and computing the mean/variance/fiber length/volume; the actual output
	is written in the subclasses. There are two subclasses, one for ".txt" files
	and one for ".xml" files.
*/

class FiberOutput
{
	public:
	
		/** Constructor */

		FiberOutput();

		/** Destructor */
	
		~FiberOutput();

		/** Enumeration for the data source, which is either Regions of
			Interest (0) or fibers (1). */

		enum DataSourceType
		{
			DS_ROI = 0,
			DS_Fibers
		};

		/** Add the input VTK object to one of the three input lists (scalar images,
			seed points, or fibers), as well as its name. 
			@param data		VTK data object.
			@param name		Name of the data. */

		void addScalarImage(vtkObject * data, std::string name);
		void addSeedPoints(vtkObject * data, std::string name);
		void addFibers(vtkObject * data, std::string name);

		/** Store pointer and name of the DTI tensor image, and the image containing
			the eigensystem derived from these tensors. */

		void setTensorImage(vtkObject * data, std::string name);
		void setEigenImage(vtkObject * data, std::string name);

		/** Set the optional outputs: Tensor components, Eigenvectors, fiber length,
			and fiber volume. All are false by default. 
			@param b		Input boolean value. */

		void setOutputTensor(bool b)
		{
			selectedTensorOutput = b;
		}

		void setOutputEigenvector(bool b)
		{
			selectedEigenVectorOutput = b;
		}

		void setOutputFiberLength(bool b)
		{
			selectedFiberLengthOutput = b;
		}

		void setOutputFiberVolume(bool b)
		{
			selectedFiberVolumeOutput = b;
		}

		/** Write the selected data to the output. Before calling this function, 
			we should add the tensor image, the eigensystem image, one or more
			scalar measure images (optional), one or more ROIs (when using ROIs 
			as data source), and/or a fiber set (when using fibers as data source.
			@param fileName		Output file.
			@param ds			Required data source (ROIs or fibers).
			@param rPerVoxel	Determine if we output data for each voxel.
			@param rMeanAndVar	Determine if we output the means and variances. */

		virtual std::string saveData(char * fileName, DataSourceType ds, bool rPerVoxel, bool rMeanAndVar);

	protected:
		
		/** Structure for storing input information (VTK Object and name). */

		struct InputInfo
		{
			vtkObject * data;
			std::string name;
		};

		/** Lists containing input information of scalar images (used for scalar measures), 
			seed points (used as data source), and fibers (used as data source). Functions using 
			the data stored in these lists should take care of casting the pointers to the
			correct type ("vtkImageData" for the scalar images, "vtkUnstructuredGrid" for the
			seed points, and "vtkPolyData" for the fibers. */

		std::vector<InputInfo> scalarImageList;
		std::vector<InputInfo> seedList;
		std::vector<InputInfo> fiberList;

		/** Input information for the main DTI image and the image containing its eigensystem. */

		InputInfo tensorImageInfo;
		InputInfo eigenImageInfo;

		/** Check if the input data sets are valid. Returns false if invalid data sets are encountered. */

		bool checkInputs();

		/** Run at the start of the output. Implemented in child classes. */
	
		virtual void outputInit();

		/** Write the header. Implemented in child classes. */

		virtual void outputHeader();

		/** Start a new worksheet. Implemented in child classes. 
			@param titel			Optional title of the worksheet. */
	
		virtual void outputInitWorksheet(std::string titel = "");

		/** Write a row of strings. Implemented in child classes. 
			@param content			Array of output strings. 
			@param contentSize		Number of strings.
			@param styleID			Output style, used for ".xml" files. */

		virtual void outputWriteRow(std::string * content = &(std::string)"", int contentSize = 1, int styleID = 0);

		/** Write a row of doubles with an optional label. Implemented in child classes. 
			@param content			Array of output doubles. 
			@param contentSize		Number of doubles.
			@param label			Optional row label.
			@param styleID			Output style, used for ".xml" files. */

		virtual void outputWriteRow(double * content, int contentSize = 1, std::string label = "", int styleID = 0); 
	
		/** Finalize current worksheet. Implemented in child classes. */
	
		virtual void outputEndWorksheet();
	
		/** Finalize output. Implemented in child classes. */
	
		virtual void outputEnd();
	
		/** Write data for each point in the set "points". Also computes the mean values.
			@param means			Array of mean values, computed in this function.
			@param points			Set of input points, either seed points or fiber points.
			@param numberOfColumns	Number of columns in the output table.
			@param label			Optional row label index. */

		void outputDataPerVoxel(double * means, vtkPoints * points, int numberOfColumns, int label = -1);

		/** Calculate the variances for all points. One variance value is computed for each
			scalar measure (plus, optionally, for the tensor values and eigenvectors).
			@param vars				Array of variances, computed in this function.
			@param means			Array of mean values, used to compute the variances.
			@param points			Set of input points, either seed points or fiber points.
			@param divPoints		Number of points used for division. */

		void calculateVars(double * vars, double * means, vtkPoints * points, int divPoints);

		/** Calculate the mean values for all points. One mean value is computed for each
			scalar measure (plus, optionally, for the tensor values and eigenvectors). 
			@param means			Array of mean values, computed in this function.
			@param points			Set of input points, either seed points or fiber points. */

		void calculateMeans(double * means, vtkPoints * point);

		/** Output the means, the variances, and (if "selectedFiberLengthOutput" is true), the
			fiber lengths. Means and variances are stored as 2D array with one row per element
			(ROI or fiber), and one column per measure. 
			@param means			Matrix containing mean values.
			@param vars				Matrix containing variance values.
			@param numberOfElements	Number of ROIs or fibers.
			@param fullAllMean		Mean values for all ROIs or fibers combined.
			@param fullAllVar		Variances for all ROIs or fibers combined.
			@param length			Fiber length array. */

		virtual void outputMeanVarAndLength(double ** means, double ** vars, int numberOfElements, double * fullAllMean, double * fullAllVar, double * lengths = NULL);

		/** Calculate the fibers volume by counting the number of voxels crossed by a fiber set.
			@param fiber			Input fiber set. */
	
		int calculateVolume(vtkPolyData * fibers);

		/** Output the volume of a fiber set. */
	
		void outputVolume();

		/** Compute the number of columns that are needed in the output tables. Depends
			on "numberOfSelectedMeasures", and on "selectedTensorOutput" and "selected-
			EigenVectorOutput". */

		void computeNumberOfColumnsNeededForMeasures();

		/** Add measure names to the header row, starting at the specified column.
			@param headerRow		Array of strings for the header row.
			@param currentColumn	Starting column for printing the names. */

		void addColumnHeaders(std::string * headerRow, int currentColumn);

		/** Compute the length of a fiber.
			@param Points			Point set containing all points of one fiber. */

		double computeFiberLength(vtkPoints * Points);

		/** Used for file writing by the child classes. */
		
		ofstream outfile;

		/** Data source, either "DS_ROI" for ROIs, or "DS_Fibers" for fibers. */

		DataSourceType dataSource;

		/** Should we output data for each point? */
	
		bool perVoxel;

		/** Should we output the means and variances? */

		bool meanAndVar;

		/** Number of ROIs. */
	
		int numberOfSelectedROIs;

		/** Number of scalar measures; equal to the size of "scalarImageList". */
	
		int numberOfSelectedMeasures;

		/** Output filename. */

		char * fileName;

		/** Optional output options. By setting these to true, we can output the
			tensor values, the tensor eigenvectors, the fiber lengths, and/or
			the volume occupied by the fiber set. */
	
		bool selectedTensorOutput;
		bool selectedEigenVectorOutput;
		bool selectedFiberVolumeOutput;
		bool selectedFiberLengthOutput;

		/** Number of columns used for the measures in the output tables. */

		int numberOfColumnsNeededForMeasures;
	
}; //class FiberOutput


} //namespace bmia


#endif