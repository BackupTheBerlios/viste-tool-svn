#ifndef __DTIGradientReader_h
#define __DTIGradientReader_h

#include "DTIConfig.h"
#include "DTIUtils.h"

#include "gsl/gsl_linalg.h"

#include <QString>

//---------------------------------------------------------------------------
//! \file   DTIGradientReader.h
//! \class  DTIGradientReader
//! \author Ralph Brecheisen
//! \brief  Reads gradient vectors from file.
//---------------------------------------------------------------------------
class DTIGradientReader
{
public:

	//---------------------------------------------------------------------------
	//! Constructor.
	//---------------------------------------------------------------------------
	DTIGradientReader()
	{
		this->GradientMatrix               = 0;
		this->GradientTransformationMatrix = 0;
		this->NumberOfGradients            = 0;
	}

	//---------------------------------------------------------------------------
	//! Destructor.
	//---------------------------------------------------------------------------
	~DTIGradientReader()
	{
		if(this->GradientMatrix != 0)
			gsl_matrix_free(this->GradientMatrix);

		if(this->GradientTransformationMatrix != 0)
			gsl_matrix_free(this->GradientTransformationMatrix);
	}

	//---------------------------------------------------------------------------
	//! Set/get/is methods.
	//---------------------------------------------------------------------------
	void SetFilePath( const QString path )
	{
		this->FilePath = path;
	};

	void SetFileName( const QString name )
	{
		this->FileName = name;
	};

	__DTIMACRO_SETGET(NumberOfGradients, int);

	//---------------------------------------------------------------------------
	//! Loads gradient vectors from file. If a gradient transformation is enabled,
	//! the corresponding gradient transformation matrix is also loaded from file
	//! and applied to the gradients.
	//! Further options are to flip one or more of the axes.
	//---------------------------------------------------------------------------
	bool LoadData()
	{
		const char *func = "DTIGradientReader::LoadData";

		// Append slash to directory.
		if( this->FilePath.contains( '/' ) )
		{
			if( this->FilePath.endsWith( '/' ) == false )
				this->FilePath.append( '/' );
		}
		else if( this->FilePath.contains( '/' ) )
		{
			if( this->FileName.endsWith( '\\' ) == false )
				this->FileName.append( '\\' );
		}
		else
		{
			std::cout << func << " filepath has no slashes, weird....";
			std::cout << "getting out of here!" << std::endl;
			return false;
		}

		// Check if number of gradients was specified. If not, quit.
		if(this->NumberOfGradients == 0)
		{
			__DTIMACRO_LOG(func << ": Number of gradients not specified" << endl,
				ERROR,
				DTIUtils::LogLevel);
			return false;
		}

		// Check if gradient (and possibly transformation) matrix not already loaded
		// before. If so, delete them first.
		if(this->GradientMatrix != 0)
			gsl_matrix_free(this->GradientMatrix);

		// Allocate memory in gradient matrix.
		this->GradientMatrix = gsl_matrix_calloc(this->NumberOfGradients, 3);

		// Start loading the gradients and store the value in the nx3 array. For internal
		// linear algebra computations we use GSL matrix.
		QString fileName = this->FilePath;
		fileName.append( this->FileName );

		FILE *f = fopen(fileName.toStdString().c_str(), "rt");

		if(f == 0)
		{
			__DTIMACRO_LOG(func << ": Error opening file " << fileName.toStdString() << endl,
				ERROR, 
				DTIUtils::LogLevel);
			return false;
		}

		int  n = 0;
		int  linenr = 0;
		char *line = new char[128];

		__DTIMACRO_LOG(func << ": Loading gradients from file " << fileName.toStdString() << endl,
			ERROR, 
			DTIUtils::LogLevel);

		while(!feof(f))
		{
			// Clear previous line contents.
			memset(line, 0, 128);

			// Read line and check for errors.
			fgets(line, 128, f);

			if(ferror(f))
			{
				__DTIMACRO_LOG(func << ": Error reading line " << linenr << endl, 
					ERROR, 
					DTIUtils::LogLevel);
				delete [] line;
				return false;
			}

			linenr++;

			// Check if line can be skipped, i.e. is empty, commented out or something else.
			if(line[0] == '#' || line[0] == '\n' || line == "" || line[0] == '\0' || line[0] == '\t' || line[0] == '\r' )
				continue;

			// Retrieve the string tokens and trim any leading or trailing spaces.
			double x = DTIUtils::StringToDouble(DTIUtils::Trim(strtok(line, ",")));
			double y = DTIUtils::StringToDouble(DTIUtils::Trim(strtok(0,    ",")));
			double z = DTIUtils::StringToDouble(DTIUtils::Trim(strtok(0,    ",")));

			__DTIMACRO_LOG(func << ": (" << x << "," << y << "," << z << ")" << endl,
				ALWAYS,
				DTIUtils::LogLevel);
			
			// Create new gradient vector
			gsl_matrix_set(this->GradientMatrix, n, 0, x);
			gsl_matrix_set(this->GradientMatrix, n, 1, y);
			gsl_matrix_set(this->GradientMatrix, n, 2, z);
			n++;
		}

		fclose(f);

		//// Print gradients.
		//cout<<endl;
		//for(int i = 0; i < this->NumberOfGradients; i++)
		//{
		//	double grad0 = gsl_matrix_get(this->GradientMatrix, i, 0);
		//	double grad1 = gsl_matrix_get(this->GradientMatrix, i, 1);
		//	double grad2 = gsl_matrix_get(this->GradientMatrix, i, 2);

		//	cout<<grad0<<","<<grad1<<","<<grad2<<endl;
		//}

		// Check whether we encountered as many gradients in the file as were
		// specified in the configuration.
		if(n != this->NumberOfGradients)
		{
			__DTIMACRO_LOG(func << ": Number of gradients found does not match number specified" << endl,
				ERROR,
				DTIUtils::LogLevel);

			if(abs(n - this->NumberOfGradients) == 1)
				__DTIMACRO_LOG(func << ": Make sure the last line in the gradients file is terminated with a newline (return) character" << endl,
				ERROR,
				DTIUtils::LogLevel);

			return false;
		}

		return true;
	}

	//---------------------------------------------------------------------------
	//! Sets gradient data directly (without having to load it).
	//---------------------------------------------------------------------------
	virtual void SetData(gsl_matrix *gradients, int numberofgradients)
	{
		if(this->GradientMatrix != 0)
			gsl_matrix_free(this->GradientMatrix);

		this->GradientMatrix = 0;
		this->GradientMatrix = gradients;

		this->NumberOfGradients = numberofgradients;
	}

	//---------------------------------------------------------------------------
	//! Apply transformation matrix to the gradients. This is done according to
	//! mathematical convention, i.e. A x = b.
	//---------------------------------------------------------------------------
	void Transform(gsl_matrix *transform)
	{
		const char *func = "DTIGradientReader::Transform";

		if(this->GradientMatrix == 0)
		{
			__DTIMACRO_LOG(func << ": Gradients not loaded yet" << endl,
				ERROR,
				DTIUtils::LogLevel);
			return;
		}

		// Apply transform.
		// Copy transpose input matrix and delete the original
		gsl_matrix *tmp = gsl_matrix_calloc(3, this->NumberOfGradients);
		gsl_matrix_transpose_memcpy(tmp, this->GradientMatrix);
		gsl_matrix_free(this->GradientMatrix);

		// Multiply transposed input matrix with transformation matrix
		gsl_matrix *tmp1 = gsl_matrix_calloc(3, this->NumberOfGradients);
		gsl_linalg_matmult(transform, tmp, tmp1);
		gsl_matrix_free(tmp);

		// Back transpose the output matrix
		gsl_matrix *output = gsl_matrix_calloc(this->NumberOfGradients, 3);
		gsl_matrix_transpose_memcpy(output, tmp1);
		gsl_matrix_free(tmp1);

		this->GradientMatrix = output;

		// Keep transformation matrix for later purposes.
		if(this->GradientTransformationMatrix != 0)
			gsl_matrix_free(this->GradientTransformationMatrix);

		this->GradientTransformationMatrix = 0;
		this->GradientTransformationMatrix = transform;
	}

	//---------------------------------------------------------------------------
	//! Flip X-axis.
	//---------------------------------------------------------------------------
	void FlipX()
	{
		const char *func = "DTIGradientReader::FlipX";

		if(this->GradientMatrix == 0)
		{
			__DTIMACRO_LOG(func << ": Gradients not loaded yet" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		// Flip the X-component of each gradient vector.
		for(int i = 0; i < this->NumberOfGradients; i++)
			gsl_matrix_set(this->GradientMatrix, i, 0, -gsl_matrix_get(this->GradientMatrix, i, 0));
	}

	//---------------------------------------------------------------------------
	//! Flip Y-axis.
	//---------------------------------------------------------------------------
	void FlipY()
	{
		const char *func = "DTIGradientReader::FlipY";

		if(this->GradientMatrix == 0)
		{
			__DTIMACRO_LOG(func << ": Gradients not loaded yet" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		// Flip the Y-component of each gradient vector.
		for(int i = 0; i < this->NumberOfGradients; i++)
			gsl_matrix_set(this->GradientMatrix, i, 1, -gsl_matrix_get(this->GradientMatrix, i, 1));
	}

	//---------------------------------------------------------------------------
	//! Flip Z-axis.
	//---------------------------------------------------------------------------
	void FlipZ()
	{
		const char *func = "DTIGradientReader::FlipZ";

		if(this->GradientMatrix == 0)
		{
			__DTIMACRO_LOG(func << ": Gradients not loaded yet" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		// Flip the Z-component of each gradient vector.
		for(int i = 0; i < this->NumberOfGradients; i++)
			gsl_matrix_set(this->GradientMatrix, i, 2, -gsl_matrix_get(this->GradientMatrix, i, 2));
	}

	//---------------------------------------------------------------------------
	//! Normalizes each gradient vector.
	//---------------------------------------------------------------------------
	void Normalize()
	{
		const char *func = "DTIGradientReader::Normalize";

		// Check if we have a gradient matrix to begin with.
		if(this->GradientMatrix == 0)
		{
			__DTIMACRO_LOG(func << ": Gradients not loaded yet" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		// Run through the rows of the matrix. These are the vectors we wish
		// to normalize.
		for(int i = 0; i < ((int) this->GradientMatrix->size1); i++)
		{
			// Get the vector
			double vec[3];
			vec[0] = gsl_matrix_get(this->GradientMatrix, i, 0);
			vec[1] = gsl_matrix_get(this->GradientMatrix, i, 1);
			vec[2] = gsl_matrix_get(this->GradientMatrix, i, 2);

			if(vec[0] == 0.0 && vec[1] == 0.0 && vec[2] == 0.0)
				continue;

			// Calculate the vector length
			double len = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

			// Normalize it
			gsl_matrix_set(this->GradientMatrix, i, 0, vec[0] / len);
			gsl_matrix_set(this->GradientMatrix, i, 1, vec[1] / len);
			gsl_matrix_set(this->GradientMatrix, i, 2, vec[2] / len);
		}
	}

	//---------------------------------------------------------------------------
	//! Returns output of this reader as GSL matrix.
	//---------------------------------------------------------------------------
	gsl_matrix *GetOutput()
	{
		const char *func = "DTIGradientReader::GetOutput";

		if(this->GradientMatrix == 0)
		{
			__DTIMACRO_LOG(func << ": Gradients not loaded yet" << endl,
				ERROR,
				DTIUtils::LogLevel);
		}

		return this->GradientMatrix;
	}

	//---------------------------------------------------------------------------
	//! Prints contents of gradient matrix.
	//---------------------------------------------------------------------------
	void PrintInfo(ostream &ostr=cout)
	{
		const char *func = "DTIGradientReader::PrintInfo";

		// Check if gradient matrix filled.
		if(this->GradientMatrix == 0)
		{
			__DTIMACRO_LOG(func << ": Nothing to print" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		ostr << func << endl;
		ostr << "    Gradient Matrix:" << endl;

		for(int i = 0; i < this->NumberOfGradients; i++)
		{
			ostr	<< "    ("
				<< gsl_matrix_get(this->GradientMatrix, i, 0) << ","
				<< gsl_matrix_get(this->GradientMatrix, i, 1) << ","
				<< gsl_matrix_get(this->GradientMatrix, i, 2) << ")" << endl;
		}
	}

private:

	// Gradient and gradient transformation matrices.
	gsl_matrix *GradientMatrix;
	gsl_matrix *GradientTransformationMatrix;

	QString FilePath;
	QString FileName;

	int   NumberOfGradients;
};

#endif
