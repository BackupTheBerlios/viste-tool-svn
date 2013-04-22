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
 * vtkHARDIReader.h
 *
 * 2010-11-18	Evert van Aart
 * - First version. 
 *
 * 2011-04-19	Evert van Aart
 * - Changed the reader so that the output has the same format as the discrete
 *   sphere functions read in by the NIfTI reader. This way, the Geometry Glyphs
 *   plugin can be used to visualize the HARDI data read by this plugin.
 * - Added triangulation for the input HARDI data.
 *
 * 2011-04-26	Evert van Aart
 * - Improved progress reporting.
 *
 */


#ifndef bmia_vtkHARDIReader_h
#define bmia_vtkHARDIReader_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - C++ */

#include <list>
#include <string>
#include <stdio.h>
#include <stdlib.h>

/** Includes - VTK */

#include <vtkImageReader2.h>
#include <vtkImageData.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkObjectFactory.h>
#include <vtkMath.h>
#include <vtkIntArray.h>

/** Includes - Custom Files */

#include "Helpers/vtkBetterDataReader.h"
#include "Helpers/vtkDTIComponentReader.h"
#include "HARDI/SphereTriangulator.h"

/** Includes - Qt */

#include <QFile>


namespace bmia {

/** Forward Class Declaration */

class vtkDTIComponentReader;


/** This class reads HARDI data formatted as a single ".hardi" header files and N
	".dat" files, where N is the number of the gradients. The output is a "vtkImageData"
	object with four point data arrays: 
	
	- An array containing one scalar value per voxel for each gradient component with 
	  non-zero gradient, named "Vectors".
	- An array containing one scalar value per voxel with the average scalar value 
	  of all images with zero gradient (i.e., the gradient vector does not contain 
	  any non-zero elements), named "Zero Gradient Average".
    - An array containing all gradients described an angle pair (zenith and azimuth),
	  named "Spherical Directions".
    - An array containing triangle definition (i.e., 3-component tuples containing
	  the point indices of points that form a triangle). Triangles are calculated
	  using the "SphereTriangulator" class. The array is named "Triangles".
	The class can also output an array with all gradients (including the zero 
	gradients), described as 3D vectors, as well as the B-value.
	
	The "vtkDTIComponentReader" class is used to read the individual ".dat" files; 
	the header is parsed in this class. 
*/	
	
class vtkHARDIReader : public vtkImageReader2
{
	public:
	
		/** Constructor Call */
		
		static vtkHARDIReader * New();
		
		/** Test whether the specified file can be opened. 
			@param fname	Desired filename. */
			
		virtual int CanReadFile(const char * fname);
		
		/** Return supported file extension. */

		virtual const char * GetFileExtensions()
		{
			return ".hardi";
		}

		/** Return description of the reader. */
		
		virtual const char * GetDescription()
		{
			return "HARDI";
		}

		/** Return array containing all gradient directions. */

		vtkDoubleArray * getGradientArray()
		{
			return gradientArray;
		}

		/** Return the B-value of the input data set. */

		double getB()
		{
			return b;
		}

	protected:
	
		/** Constructor */
		
		vtkHARDIReader();
		
		/** Destructor */
		
		~vtkHARDIReader();
		
		/** Reset parameters, free up allocated memory. Called when any part of the 
			reading process fails, and when it finishes successfully. */
			
		void CleanUp();

		/** Reads header file (".hardi"). */
		
		virtual void ExecuteInformation();
		
		/** Reads all component data files (".dat").
			@param out	Output data. Not used. */
			
		virtual void ExecuteData(vtkDataObject * out); 
		
		/** Read the next line of the input file. */
		
		virtual bool NextLine();
		
		/** Try to read the header; return true on succes. */
		
		bool ReadHeader();
		
		/** Try to read the data type; return true on succes. */
		
		bool ReadDataType();
				
		/** Try to read the voxel size; return true on succes. */
		
		bool ReadVoxelSize();
		
		/** Try to read the B-value; return true on succes. */
		
		bool ReadBValue();
		
		/** Try to read the number of gradients; return true on succes. */
		
		bool ReadNumberOfGradients();
		
		/** Try to read the dimensions; return true on succes. */
		
		bool ReadDimensions();
		
		/** Try to read the filenames of the ".dat" files; return true on succes. */
				
		bool ReadGradientFileNamesDat();

		/** Set to true if "ExecuteInformation" has completed successfully. */
		
		bool InformationExecuted;
		
		/** If true, we read the image dimensions from the ".hardi" file instead of
			from the ".dat" files. True if the first line of the ".hardi" file is "HARDI03". */
			
		bool readDimensions;

		/** Current line of the input file. */
		
		std::string CurrentLine;
		
		/** Input stream used to read the file. */
		
		ifstream * IStream;

		/** Data type of the ".dat" files. Either VTK_UNSIGNED_SHORT or VTK_FLOAT. */
		
		int DataType;

		/** Number of gradients in the input. */
		
		int numberOfGradients;
		
		/** Dimensions of the input image. */
		
		int Dimensions[3];
		
		/** Array containing all gradient directions, described as 3D vectors. */

		vtkDoubleArray * gradientArray;

		/** Array containing only non-zero gradients, described as zenith-azimuth pairs. */

		vtkDoubleArray * anglesArray;

		/** Number of zero gradients. */
		
		int zeroGradientCounter;
		
		/** B-value of the HARDI data. */
		
		double b;

		/** List of filenames of the ".dat" files. */
		
		std::list<std::string> ComponentFileNames;

		/** Output data arrays. */
		
		vtkDataArray * grayValuesArray;
		vtkDataArray * zeroGradientsArray;
		
		/** Used during the reading of the ".dat" files. */
		
		vtkDTIComponentReader * componentReader;
		vtkImageData * componentImageData;
		vtkPointData * componentPointData;
		vtkDataArray * componentScalars;
		
}; // class vtkHARDIReader


} // namespace bmia


#endif // bmia_vtkHARDIReader_h






