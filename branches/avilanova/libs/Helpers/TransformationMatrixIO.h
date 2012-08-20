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