#ifndef __DTITensor2DtiToolConverter_h
#define __DTITensor2DtiToolConverter_h

#include "DTITensorSlice.h"
#include <vector>

#include <QString>

using namespace std;

//---------------------------------------------------------------------------
//! \file   DTITensor2DtiToolConverter.h
//! \class  DTITensor2DtiToolConverter
//! \author Ralph Brecheisen
//! \brief  Writes tensor data to DTI tool volumes.
//---------------------------------------------------------------------------
class DTITensor2DtiToolConverter
{
public:

	//-------------------------------------------------------------------
	//! Constructor.
	//-------------------------------------------------------------------
	DTITensor2DtiToolConverter()
	{
		this->Input          = 0;
		this->Volumes        = 0;
		this->Rows           = 0;
		this->Columns        = 0;
		this->NumberOfSlices = 0;
		this->FileName       = 0;
		this->Version        = 0;
		this->DataType       = 0;
		this->SliceThickness = 0;
		this->PixelSpacing   = 0;
	}

	//-------------------------------------------------------------------
	//! Destructor.
	//-------------------------------------------------------------------
	~DTITensor2DtiToolConverter()
	{
		if(this->Input != 0)
		{
			for(unsigned int i = 0; i < this->Input->size(); i++)
			{
				DTITensorSlice * slice = this->Input->at(i);
				delete slice;
			}
			delete this->Input;
		}

		if(this->Volumes != 0)
		{
			for ( unsigned int i = 0; i < this->Volumes->size(); i++ )
			{
				std::vector<gsl_matrix *> * list = this->Volumes->at(i);
				for ( unsigned int j = 0; j < list->size(); j++ )
				{
					gsl_matrix * matrix = list->at(j);
					gsl_matrix_free ( matrix );
				}

				delete list;
			}

			delete this->Volumes;
		}
	}

	//-------------------------------------------------------------------
	//! Sets the input data.
	//-------------------------------------------------------------------
	virtual void SetInput(vector<DTITensorSlice *> *input)
	{
		if(this->Input != 0)
		{
			this->Input->clear();
			delete this->Input;
		}

		this->Input = 0;
		this->Input = input;
	}

	//-------------------------------------------------------------------
	//! Sets output filename prefix.
	//-------------------------------------------------------------------
	virtual void SetFileName(char *filename)
	{
		if(this->FileName != 0)
			delete [] this->FileName;

		this->FileName = 0;
		this->FileName = filename;
	}

	//-------------------------------------------------------------------
	//! Sets DTI tool version.
	//-------------------------------------------------------------------
	virtual void SetVersion(int version)
	{
		this->Version = version;
	}

	//-------------------------------------------------------------------
	//! Sets DTI tool data type.
	//-------------------------------------------------------------------
	virtual void SetDataType( const QString dataType )
	{
		const char *func = "DTITensor2DtiToolConverter::SetDataType";

		if( dataType.toLower() == "float" )
		{
			this->DataType = DTIUtils::FLOAT;
		}
		else if( dataType.toLower() == "ushort" )
		{
			this->DataType = DTIUtils::USHORT;
		}
		else
		{
			__DTIMACRO_LOG(func << "Illegal datatype " << dataType.toStdString() << endl, ERROR, DTIUtils::LogLevel);
			return;
		}
	}

	//-------------------------------------------------------------------
	//! Sets pixel spacing in X-, Y- and Z-direction.
	//-------------------------------------------------------------------
	virtual void SetPixelSpacing(double *pixelspacing)
	{
		this->PixelSpacing = pixelspacing;
	}

	//-------------------------------------------------------------------
	//! Sets slice thickness.
	//-------------------------------------------------------------------
	virtual void SetSliceThickness(double slicethickness)
	{
		this->SliceThickness = slicethickness;
	}

	//-------------------------------------------------------------------
	//! Executes converter.
	//-------------------------------------------------------------------
	virtual bool Execute()
	{
		const char *func = "DTITensor2DtiToolConverter::Execute";

		// Check if we have all required properties and data.
		if(this->Input == 0)
		{
			__DTIMACRO_LOG(func << ": No input set" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		// Get rows, columns and number of tensor slices.
		this->Rows           = this->Input->at(0)->GetRows();
		this->Columns        = this->Input->at(0)->GetColumns();
		this->NumberOfSlices = this->Input->size();

		// If we already created a volume list before, clear it and create an empty
		// one.
		if(this->Volumes != 0)
		{
			this->Volumes->clear();
			this->Volumes = 0;
		}

		this->Volumes = new vector<vector<gsl_matrix *> *>;

		// For tensor value (1 out of 6) create a separate volume.
		for(int k = 0; k < 6; k++)
		{
			// Create empty volume as a list of GSL matrices.
			vector<gsl_matrix *> *volume = new vector<gsl_matrix *>;

			for(int index = 0; index < this->NumberOfSlices; index++)
			{
				// Create temporary matrix to store slice data.
				gsl_matrix *slice = gsl_matrix_calloc(this->Rows, this->Columns);

				// Get tensor slice from the input data. We have to reverse the order in the
				// Z-direction because DTI tool expects the slice to be ordered that way. So,
				// instead of this->Input->at(index) we have this->Input->at(... - index).
				DTITensorSlice *tensorslice = this->Input->at((this->NumberOfSlices - 1) - index);

				// For each element in the tensor slice (at position (i,j)) we extract the k-th
				// tensor value and store it in our current volume.
				for(int i = 0; i < this->Rows; i++)
				{
					for(int j = 0; j < this->Columns; j++)
					{
						// Get the (6-valued) tensor.
						gsl_matrix *tensor = tensorslice->GetTensorAt(i, j);

//						std::cout
//							<< gsl_matrix_get( tensor, 0, 0 ) << " "
//							<< gsl_matrix_get( tensor, 1, 0 ) << " "
//							<< gsl_matrix_get( tensor, 2, 0 ) << " "
//							<< gsl_matrix_get( tensor, 3, 0 ) << " "
//							<< gsl_matrix_get( tensor, 4, 0 ) << " "
//							<< gsl_matrix_get( tensor, 5, 0 ) << std::endl;

						// Start with the last row index. DTI tool expects this, so instead of
						// (i, j) we have ((this->Rows-1-i), j).
						gsl_matrix_set(slice, ((this->Rows - 1) - i), j, gsl_matrix_get(tensor, k, 0));
					}
				}

				// Add the slice to our current volume.
				volume->push_back(slice);
			}

			// Add the volume to the list of volumes.
			this->Volumes->push_back(volume);
		}

		// We now have 6 volumes for the DTI tool. What remains is creating the identity
		// volume.
		vector<gsl_matrix *> *volumeXX = this->Volumes->at(0);
		vector<gsl_matrix *> *volumeYY = this->Volumes->at(1);
		vector<gsl_matrix *> *volumeZZ = this->Volumes->at(2);

		// Create empty identity volume.
		vector<gsl_matrix *> *volumeII = new vector<gsl_matrix *>;

		vector<gsl_matrix *>::iterator iterXX = volumeXX->begin();
		vector<gsl_matrix *>::iterator iterYY = volumeYY->begin();
		vector<gsl_matrix *>::iterator iterZZ = volumeZZ->begin();

		for( ; iterXX != volumeXX->end(); iterXX++, iterYY++, iterZZ++)
		{
			gsl_matrix *tmpII = gsl_matrix_calloc(this->Rows, this->Columns);

			gsl_matrix *tmpXX = (*iterXX);
			gsl_matrix *tmpYY = (*iterYY);
			gsl_matrix *tmpZZ = (*iterZZ);

			for(int i = 0; i < this->Rows; i++)
			{
				for(int j = 0; j < this->Columns; j++)
				{
					double value =	(gsl_matrix_get(tmpXX, i, j) +
							 gsl_matrix_get(tmpYY, i, j) +
							 gsl_matrix_get(tmpZZ, i, j)) / 3.0;

					gsl_matrix_set(tmpII, i, j, value);
				}
			}

			// Store the identity slice in the II volume.
			volumeII->push_back(tmpII);
		}

		// Again add the identity volume to the end of the volume list.
		this->Volumes->push_back(volumeII);
		return true;
	}

	//-------------------------------------------------------------------
	//! Write converted data to file.
	//-------------------------------------------------------------------
	virtual bool Write()
	{
		const char *func = "DTITensor2DtiToolConverter::Write";

		// Check for a number of things.
		if(this->Volumes == 0)
		{
			__DTIMACRO_LOG(func << ": No data converted yet" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		if(this->DataType == 0)
		{
			__DTIMACRO_LOG(func << ": No output data type specified" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		if(this->PixelSpacing == 0)
		{
			__DTIMACRO_LOG(func << ": No pixel spacing specified" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		if(this->SliceThickness == 0)
		{
			__DTIMACRO_LOG(func << ": No slice thickness specified" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		// Build filenames based on user-defined filename prefix.

		std::string filenames[7];

		if ( this->FileName != NULL )
		{
			filenames[0] = std::string(this->FileName).append("_XX.dat");
			filenames[1] = std::string(this->FileName).append("_YY.dat");
			filenames[2] = std::string(this->FileName).append("_ZZ.dat");
			filenames[3] = std::string(this->FileName).append("_XY.dat");
			filenames[4] = std::string(this->FileName).append("_YZ.dat");
			filenames[5] = std::string(this->FileName).append("_XZ.dat");
			filenames[6] = std::string(this->FileName).append("_II.dat");
		}
		else
		{
			filenames[0] = std::string("XX.dat");
			filenames[1] = std::string("YY.dat");
			filenames[2] = std::string("ZZ.dat");
			filenames[3] = std::string("XY.dat");
			filenames[4] = std::string("YZ.dat");
			filenames[5] = std::string("XZ.dat");
			filenames[6] = std::string("II.dat");
		}

		vector<vector<gsl_matrix *> *>::iterator iter;
		int index = 0;

		for(iter = this->Volumes->begin(); iter != this->Volumes->end(); iter++)
		{
			// Get current volume, which is a list of GSL matrices (the slices)
			vector<gsl_matrix *> *volume = (*iter);

			// Open volume file for writing.
			FILE *f = fopen(filenames[index].c_str(), "wb");
			if(f == 0)
			{
				__DTIMACRO_LOG(func << ": Could not open file " << filenames[index] << " for writing" << endl, ERROR, DTIUtils::LogLevel);
				return false;
			}

			// Write volume dimensions to file first.
			fwrite(&this->Columns,        sizeof(unsigned short), 1, f);
			fwrite(&this->Rows,           sizeof(unsigned short), 1, f);
			fwrite(&this->NumberOfSlices, sizeof(unsigned short), 1, f);

			// Go through each slice and append its data to the newly created .DAT file.
			vector<gsl_matrix *>::iterator sliceiter;

			for(sliceiter = volume->begin(); sliceiter != volume->end(); sliceiter++)
			{
				gsl_matrix *slice = (*sliceiter);

				for(int i = 0; i < this->Rows; i++)
				{
					for(int j = 0; j < this->Columns; j++)
					{
						if(this->DataType == DTIUtils::FLOAT)
						{
							float value = (float) gsl_matrix_get(slice, i, j);
							fwrite(&value, sizeof(float), 1, f);
						}
						else if(this->DataType == DTIUtils::USHORT)
						{
							unsigned short value = (unsigned short) gsl_matrix_get(slice, i, j);
							fwrite(&value, sizeof(unsigned short), 1, f);
						}
						else
						{
						}
					}
				}
			}

			// Print that we successfully wrote the .DAT volume to file.
			__DTIMACRO_LOG(func << ": Successfully written " << filenames[index] << endl, ALWAYS, DTIUtils::LogLevel);

			fclose(f);
			index++;
		}

		if(this->Version == 1)
		{
			// First write old version DTI tool header.
			FILE *f = fopen("DTITOOL1.dti", "wt");
			if(f == 0)
			{
				__DTIMACRO_LOG(func << ": Could not open DTITOOL1.dti for writing" << endl, ERROR, DTIUtils::LogLevel);
				return false;
			}

			fprintf(f, "/* DTI BMT format */\n");
			fprintf(f, "T\n");

			if(this->DataType == DTIUtils::FLOAT)
				fprintf(f, "float\n");
			else if(this->DataType == DTIUtils::USHORT)
				fprintf(f, "ushort\n");

			fprintf(f, "XX XX.dat\n");
			fprintf(f, "YY YY.dat\n");
			fprintf(f, "ZZ ZZ.dat\n");
			fprintf(f, "XY XY.dat\n");
			fprintf(f, "XZ YZ.dat\n");
			fprintf(f, "YZ XZ.dat\n");
			fprintf(f, "I  II.dat\n");
			fprintf(f, "%f %f %f\n", (float) this->PixelSpacing[0], (float) this->PixelSpacing[1], (float) this->SliceThickness);
			fclose(f);

			__DTIMACRO_LOG(func << ": Successfully written DTITOOL1.dti header file" << endl, ALWAYS, DTIUtils::LogLevel);
		}
		else
		{
			if ( this->FileName == NULL )
				this->FileName = (char *) "tensorvolume";

			std::string fname(this->FileName);
			fname.append(".dti");

			FILE *f = fopen(fname.c_str(), "wt");
			if(f == 0)
			{
				__DTIMACRO_LOG(func << ": Could not open *.dti for writing" << endl, ERROR, DTIUtils::LogLevel);
				return false;
			}

			fprintf(f, "/* DTI BMT format */\n");
			fprintf(f, "T\n");

			if(this->DataType == DTIUtils::FLOAT)
				fprintf(f, "float\n");
			else if(this->DataType == DTIUtils::USHORT)
				fprintf(f, "ushort\n");

			fprintf(f, "XX %s\n", filenames[0].c_str());
			fprintf(f, "YY %s\n", filenames[1].c_str());
			fprintf(f, "ZZ %s\n", filenames[2].c_str());
			fprintf(f, "XY %s\n", filenames[3].c_str());
			fprintf(f, "XZ %s\n", filenames[4].c_str());
			fprintf(f, "YZ %s\n", filenames[5].c_str());
			fprintf(f, "I  %s\n", filenames[6].c_str());
			fprintf(f, "%f %f %f\n", (float) this->PixelSpacing[0], (float) this->PixelSpacing[1], (float) this->SliceThickness);
			fclose(f);

			__DTIMACRO_LOG(func << ": Successfully written *.dti header file" << endl, ALWAYS, DTIUtils::LogLevel);
		}

		// free stuff

		return true;
	}

protected:

	vector<DTITensorSlice *>       *Input;
	vector<vector<gsl_matrix *> *> *Volumes;

	char   *FileName;
	double  SliceThickness;
	double *PixelSpacing;
	int     Version;
	int     Rows;
	int     Columns;
	int     NumberOfSlices;
	int     DataType;
};

#endif
