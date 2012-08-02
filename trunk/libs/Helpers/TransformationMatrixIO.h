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
 * TransformationMatrixIO.h
 *
 * 2011-01-24	Evert van Aart
 * - First version. 
 *
 */


#ifndef BMIA_TRANSFORMATIONMATRIXIO_H
#define BMIA_TRANSFORMATIONMATRIXIO_H


/** Includes - VTK */

#include <vtkMatrix4x4.h>

/** Includes - C++ */

#include <string>

/** Includes - Qt */

#include <QString>
#include <QStringList>
#include <QTextStream>
#include <QFile>
#include <QFileInfo>


namespace bmia {

/** Class used to write and read transformation matrix files. This is used to
	store and load the transformation matrix of data sets to and from file formats
	that do not natively contain this matrix. For example: Fibers are stored as a
	".fbs" file, which is simply the output of a VTK polydata writer. If we want
	to store the transformation matrix for a set of fibers (to align it with the
	corresponding DTI image), we can write it to a seperate file with the same base
	name as the ".fbs" file, i.e., we'd have "fibers.fbs" and "fibers.tfm". The
	".tfm" file contains the 4x4 matrix, formatted as ASCII.
*/

class TransformationMatrixIO
{
	public:

		static vtkMatrix4x4 * readMatrix(std::string filename, std::string & errorMessage);

		static bool writeMatrix(std::string filename, vtkMatrix4x4 * m, std::string & errorMessage);

}; // class TransformationMatrixIO


} // namespace bmia


#endif // BMIA_TRANSFORMATIONMATRIXIO_H