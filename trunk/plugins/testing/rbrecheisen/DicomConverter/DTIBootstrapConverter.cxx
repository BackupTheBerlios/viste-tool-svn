#include "DTIBootstrapConverter.h"
#include "DTITensor2DtiToolConverter.h"
#include <time.h>

/////////////////////////////////////////////////////////////////
DTIBootstrapConverter::DTIBootstrapConverter ()
{
	this->ADCVectorSlices = NULL;
	this->ResidualVectorSlices = NULL;
	this->NumberOfBootstrapVolumes = 1;
	this->SliceThickness = 0.0;
	this->StartIndex = 0;
	this->PixelSpacing[0] = 0.0;
	this->PixelSpacing[1] = 0.0;
}

/////////////////////////////////////////////////////////////////
DTIBootstrapConverter::~DTIBootstrapConverter ()
{
	if ( this->ADCVectorSlices )
		delete this->ADCVectorSlices;
	if ( this->ResidualVectorSlices )
		delete this->ResidualVectorSlices;
}

/////////////////////////////////////////////////////////////////
void DTIBootstrapConverter::SetNumberOfBootstrapVolumes ( int nrvolumes )
{
	this->NumberOfBootstrapVolumes = nrvolumes;
	if ( this->NumberOfBootstrapVolumes < 1 ) this->NumberOfBootstrapVolumes = 1;
}

/////////////////////////////////////////////////////////////////
void DTIBootstrapConverter::SetPixelSpacing ( double x, double y )
{
	this->PixelSpacing[0] = x;
	this->PixelSpacing[1] = y;
}

/////////////////////////////////////////////////////////////////
void DTIBootstrapConverter::SetSliceThickness ( double slicethickness )
{
	this->SliceThickness = slicethickness;
}

/////////////////////////////////////////////////////////////////
bool DTIBootstrapConverter::Execute ()
{
	unsigned long start = clock ();

	// mask slices

	double masksetting = 0.0;
	if(this->MaskEnabled)
	{
		masksetting = this->CalculateMaskSetting();
	}

	gsl_matrix *gradientsinverse = this->CalculatePseudoInverse(this->GradientsExtended);

	// compute adc and residual volumes

	vector<DTISliceGroup *>::iterator iter;
	int count = 0;

	for ( iter = this->Input->begin(); iter != this->Input->end(); iter++ )
	{
		DTISliceGroup *slicegroup = (*iter);
		int numberofslices = slicegroup->GetSize ();

		DTISlice *b0slice = slicegroup->GetSliceAt(this->B0SliceFirstIndex);
		unsigned short *b0pixels = b0slice->GetData();

		if(this->MaskEnabled)
		{
			for(int k = 0; k < numberofslices; k++)
			{
				double grad0 = gsl_matrix_get(this->Gradients, k, 0);
				double grad1 = gsl_matrix_get(this->Gradients, k, 1);
				double grad2 = gsl_matrix_get(this->Gradients, k, 2);

				if(grad0 == 0.0 && grad1 == 0.0 && grad2 == 0.0)
				{
					continue;
				}

				unsigned short *gradientpixels = slicegroup->GetSliceAt(k)->GetData();
				this->ApplyMask(gradientpixels, b0pixels, masksetting);
			}
		}

		DTIVectorSlice * adcslice = new DTIVectorSlice ();
		adcslice->SetRows ( this->Rows );
		adcslice->SetColumns ( this->Columns );

		DTIVectorSlice * residualslice = new DTIVectorSlice ();
		residualslice->SetRows ( this->Rows );
		residualslice->SetColumns ( this->Columns );

		for ( int i = 0; i < this->Rows; i++ )
		{
			for ( int j = 0; j < this->Columns; j++ )
			{
				gsl_matrix * adcvector = NULL, * residualvector = NULL; 
				this->ComputeADCAndResidualVectors ( i, j, slicegroup, b0pixels, this->GradientsExtended, 
					gradientsinverse, adcvector, residualvector );

				adcslice->SetVectorAt ( adcvector, i, j );
				residualslice->SetVectorAt ( residualvector, i, j );
			}
		}

		if ( this->ADCVectorSlices == NULL )
			this->ADCVectorSlices = new std::vector<DTIVectorSlice *>;
		this->ADCVectorSlices->push_back ( adcslice );

		if ( this->ResidualVectorSlices == NULL )
			this->ResidualVectorSlices = new std::vector<DTIVectorSlice *>;
		this->ResidualVectorSlices->push_back ( residualslice );

		count++;

		__DTIMACRO_LOG("Created ADC and residual slice " << count << endl, ALWAYS, DTIUtils::LogLevel);
	}

	// compute bootstrap tensor volumes

	for ( int volNr = 0; volNr < this->NumberOfBootstrapVolumes; volNr++ )
	{
		std::vector<DTITensorSlice *> * tensorslices = new std::vector<DTITensorSlice *>;

		for ( unsigned int sliceNr = 0; sliceNr < this->ADCVectorSlices->size (); sliceNr++ )
		{
			DTIVectorSlice * adcslice = this->ADCVectorSlices->at ( sliceNr );
			DTIVectorSlice * residualslice = this->ResidualVectorSlices->at ( sliceNr );

			DTITensorSlice * tensorslice = new DTITensorSlice;
			tensorslice->SetRows ( this->Rows );
			tensorslice->SetColumns ( this->Columns );

			for ( int i = 0; i < this->Rows; i++ )
			{
				for ( int j = 0; j < this->Columns; j++ )
				{
					gsl_matrix * adcvector = adcslice->GetVectorAt ( i, j );
					gsl_matrix * residualvector = residualslice->GetVectorAt ( i, j );
					gsl_matrix * tensor = this->ComputeRandomTensor ( adcvector, residualvector, gradientsinverse );
					tensorslice->SetTensorAt ( tensor, i, j );
				}
			}

			tensorslices->push_back ( tensorslice );
		}

		// write tensor volume

		this->Write ( volNr, tensorslices );
	}

	// free stuff

	gsl_matrix_free(gradientsinverse);
	unsigned long stop = clock ();
	std::cout << "Time elapsed: " << (stop-start)/CLOCKS_PER_SEC << std::endl;
	return true;
}

/////////////////////////////////////////////////////////////////
void DTIBootstrapConverter::Write ( int index, std::vector<DTITensorSlice *> * tensorslices )
{
	// create filename based on index. if a start index greater than zero
	// was specified, the filename will start at the start index. this can
	// be useful for repeated runs.

	char fname[32];
	sprintf(fname, "volume%d", index + this->StartIndex);
	DTITensor2DtiToolConverter * converter = new DTITensor2DtiToolConverter;

	converter->SetInput ( tensorslices );
	converter->Execute ();
	converter->SetDataType ( (char *) "float" );
	converter->SetPixelSpacing ( this->PixelSpacing );
	converter->SetSliceThickness ( this->SliceThickness );
	converter->SetFileName(fname);
	converter->Write ();

	delete converter;
}

/////////////////////////////////////////////////////////////////
void DTIBootstrapConverter::ComputeADCAndResidualVectors ( int i, int j, DTISliceGroup * slicegroup, unsigned short * b0pixels,
	gsl_matrix * gradients, gsl_matrix * gradientsinverse, gsl_matrix *& adcvector, gsl_matrix *& residualvector )
{
	int nrslices = slicegroup->GetSize ();
	int position = i * this->Columns + j;
	adcvector = gsl_matrix_calloc ( this->NumberOfGradients - this->NumberOfB0Slices, 1 );
	

	int index = 0;
	for(int k = 0; k < nrslices; k++)
	{
		double grad0 = gsl_matrix_get(this->Gradients, k, 0);
		double grad1 = gsl_matrix_get(this->Gradients, k, 1);
		double grad2 = gsl_matrix_get(this->Gradients, k, 2);

		if(grad0 == 0.0 && grad1 == 0.0 && grad2 == 0.0)
			continue;

		unsigned short *gradientpixels = slicegroup->GetSliceAt(k)->GetData();
		double adcvalue = 0.0;

		if(gradientpixels[position] != 0 && b0pixels[position] != 0)
		{
			adcvalue = -(1.0 / this->BValue) * log(gradientpixels[position] / ((double) b0pixels[position]));
		}

		gsl_matrix_set(adcvector, index, 0, adcvalue);
		index++;
	}

	gsl_matrix * tensor = gsl_matrix_calloc ( 6, 1 );
	gsl_linalg_matmult ( gradientsinverse, adcvector, tensor );

	gsl_matrix * approximatedadcvector = gsl_matrix_calloc ( this->NumberOfGradients - this->NumberOfB0Slices, 1 );
	gsl_linalg_matmult ( gradients, tensor, approximatedadcvector );

	residualvector = gsl_matrix_calloc ( this->NumberOfGradients - this->NumberOfB0Slices, 1 );
	index = 0;
	for ( int k = 0; k < nrslices; k++ )
	{
		double grad0 = gsl_matrix_get ( this->Gradients, k, 0 );
		double grad1 = gsl_matrix_get ( this->Gradients, k, 1 );
		double grad2 = gsl_matrix_get ( this->Gradients, k, 2 );

		if ( grad0 == 0.0 && grad1 == 0.0 && grad2 == 0.0 )
			continue;

		double x0 = gsl_matrix_get ( adcvector, index, 0 );
		double x1 = gsl_matrix_get ( approximatedadcvector, index, 0 );
		gsl_matrix_set ( residualvector, index, 0, x1 - x0 );
		index++;
	}

	gsl_matrix_free ( approximatedadcvector );
	gsl_matrix_free ( tensor );
	return;
}

/////////////////////////////////////////////////////////////////
gsl_matrix * DTIBootstrapConverter::ComputeRandomTensor ( gsl_matrix * adcvector, gsl_matrix * residualvector, 
	gsl_matrix * gradientsinverse )
{
	int size = adcvector->size1;
	gsl_matrix * randomizedadcvector = gsl_matrix_calloc ( size, 1 );

	srand ( (unsigned int) time ( NULL ) );

	for ( int i = 0; i < size; i++ )
	{
		double residual = gsl_matrix_get ( residualvector, i, 0 );
		double adcvalue = gsl_matrix_get ( adcvector, i, 0 );
		double flipvalue = (rand () % 2 == 0) ? 1 : -1;
		gsl_matrix_set ( randomizedadcvector, i, 0, adcvalue + residual * flipvalue );
	}

	gsl_matrix * tensor = gsl_matrix_calloc ( 6, 1 );
	gsl_linalg_matmult ( gradientsinverse, randomizedadcvector, tensor );

	//printf ( "adc=" );
	//for ( int i = 0; i < size; i++ )
	//	printf ( "%f,", gsl_matrix_get ( adcvector, i, 0 ) );
	//printf ( "\nrandomadc=" );
	//for ( int i = 0; i < size; i++ )
	//	printf ( "%f,", gsl_matrix_get ( randomizedadcvector, i, 0 ) );
	//printf ( "\n" );

	gsl_matrix_free ( randomizedadcvector );
	return tensor;
}
