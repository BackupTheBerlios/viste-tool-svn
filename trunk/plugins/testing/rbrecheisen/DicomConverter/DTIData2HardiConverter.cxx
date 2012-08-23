#include "DTIData2HardiConverter.h"
#include "DTIUtils.h"

///////////////////////////////////////////////
DTIData2HardiConverter::DTIData2HardiConverter()
{
	this->Input = NULL;
	this->Gradients = NULL;
	this->NumberOfGradients = 0;
	this->Rows = 0;
	this->Columns = 0;
	this->NumberOfSlices = 0;
	this->DataType = DTIUtils::USHORT;
	this->Version = 0;
	this->PixelSpacing = NULL;
	this->SliceThickness = 0.0;
}

///////////////////////////////////////////////
DTIData2HardiConverter::~DTIData2HardiConverter()
{
}

///////////////////////////////////////////////
void DTIData2HardiConverter::SetInput(vector<DTISliceGroup *> *input)
{
	if(this->Input != 0)
	{
		this->Input->clear();
		delete this->Input;
	}

	this->Input = 0;
	this->Input = input;

	DTISlice *slice = this->Input->at(0)->GetSliceAt(0);
	this->Rows = slice->GetRows();
	this->Columns = slice->GetColumns();
	this->NumberOfSlices = this->Input->size();
	this->PixelSpacing = slice->GetPixelSpacing();
	this->SliceThickness = slice->GetSliceThickness();

	printf("Rows=%d, Columns=%d, NumberOfSlices=%d\n", this->Rows, this->Columns, this->NumberOfSlices);
}

///////////////////////////////////////////////
void DTIData2HardiConverter::SetGradients(gsl_matrix * gradients)
{
	if(this->Gradients != 0)
		gsl_matrix_free(this->Gradients);
	this->Gradients = gradients;

	this->NumberOfGradients = (int) this->Gradients->size1;
}

///////////////////////////////////////////////
void DTIData2HardiConverter::SetDataType(char *type)
{
	const char *func = "DTIData2HardiConverter::SetDataType";

	if(strcmp(strlwr(type), "ushort") == 0)
		this->DataType = DTIUtils::USHORT;
	else if(strcmp(strlwr(type), "float") == 0)
		this->DataType = DTIUtils::FLOAT;
	else
		__DTIMACRO_LOG(func << ": Data type " << type << " unknown" << endl, ERROR, DTIUtils::LogLevel);
}

///////////////////////////////////////////////
void DTIData2HardiConverter::SetVersion(int version)
{
	this->Version = version;
}

///////////////////////////////////////////////
void DTIData2HardiConverter::SetBValue(double value)
{
	this->BValue = value;
}

///////////////////////////////////////////////
bool DTIData2HardiConverter::Execute()
{
	const char *func = "DTIData2HardiConverter::Execute";

	// Write gradient .dat files.
	FILE **f = new FILE*[this->NumberOfGradients];

	for(int i = 0; i < this->NumberOfGradients; i++)
	{
		ostringstream str;
		if(i < 10) str << "g00" << i << ".dat";
		else if(i >= 10 && i < 100) str << "g0" << i << ".dat";
		else if(i >= 100) str << "g" << i << ".dat";
		else
		{
			__DTIMACRO_LOG(func << ": Number of gradients too big" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		f[i] = fopen(str.str().c_str(), "wb");
		if(f[i] == NULL)
		{
			__DTIMACRO_LOG(func << ": Could not open file " << str.str().c_str() << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		// Write dimensions to file first.
		fwrite(&this->Columns,        sizeof(unsigned short), 1, f[i]);
		fwrite(&this->Rows,           sizeof(unsigned short), 1, f[i]);
		fwrite(&this->NumberOfSlices, sizeof(unsigned short), 1, f[i]);
	}

	int bla = 0;
	vector<DTISliceGroup *>::iterator iter;
	for(iter = this->Input->begin(); iter != this->Input->end(); iter++, bla++)
	{
		DTISliceGroup *slicegroup = (*iter);

		for(int n = 0; n < this->NumberOfGradients; n++)
		{
			std::cout << "slicegroup " << bla << " has " << slicegroup->GetSize() << " number of slices" << std::endl;
			unsigned short *pixels = slicegroup->GetSliceAt(n)->GetData();
			fwrite(pixels, sizeof(unsigned short), this->Rows*this->Columns, f[n]);
		}
	}

	cout << endl;
	__DTIMACRO_LOG(func << ": Successfully written gradient volumes." << endl, ALWAYS, DTIUtils::LogLevel);
	for(int i = 0; i < this->NumberOfGradients; i++)
		fclose(f[i]);

	// Write .hardi file.
	FILE *file = fopen("g.hardi", "wt");
	if(file == NULL)
	{
		__DTIMACRO_LOG(func << ": Could not open file g.hardi" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	ostringstream str;
	str << "HARDI";
	if(this->Version < 10) str << "0" << this->Version << endl;
	else str << this->Version << endl;
	if(this->DataType == DTIUtils::FLOAT) str << "type float" << endl;
	else str << "type ushort" << endl;
	str << "b " << this->BValue << endl;

	// Write voxel size.
	str << this->PixelSpacing[0] << " " << this->PixelSpacing[1] << " " << this->SliceThickness << endl;

	// Write number of gradients.
	str << this->NumberOfGradients << endl;

	for(int i = 0; i < this->NumberOfGradients; i++)
	{
		if(i < 10) str << "g00" << i;
		else if(i >= 10 && i < 100) str << "g0" << i;
		else str << "g" << i;

		str << " ";

		str << gsl_matrix_get(this->Gradients, i, 0) << " ";
		str << gsl_matrix_get(this->Gradients, i, 1) << " ";
		str << gsl_matrix_get(this->Gradients, i, 2) << " ";

		if(i < 10) str << "g00" << i << ".dat";
		else if(i >= 10 && i < 100) str << "g0" << i << ".dat";
		else str << "g" << i << ".dat";

		str << endl;
	}

	fprintf(file, str.str().c_str());
	fclose(file);

	__DTIMACRO_LOG(func << ": Successfully written .hardi file." << endl, ALWAYS, DTIUtils::LogLevel);
	return true;
}
