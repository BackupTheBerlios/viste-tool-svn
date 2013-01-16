#include "DTIPARRECReader.h"
#include "DTIUtils.h"

////////////////////////////////////////////////////
DTIPARRECReader::DTIPARRECReader()
{
	this->Version = 4;

	// General information.
	this->PatientName = NULL;
	this->ExaminationName = NULL;
	this->ProtocolName = NULL;
	this->ExaminationDateTime = NULL;
	this->SeriesType = NULL;
	this->ReconstructionNr = 0;
	this->ScanDuration = 0;
	this->NrCardiacPhases = 0;
	this->NrEchoes = 0;
	this->NrSlices = 0;
	this->NrDynamics = 0;
	this->NrMixes = 0;
	this->PatientPosition = NULL;
	this->PreparationDirection = NULL;
	this->ScanResolution = NULL;
	this->ScanMode = NULL;
	this->RepetitionTime = 0;
	this->FieldOfView = NULL;
	this->WaterFatShift = 0;
	this->AngulationMidSlice = NULL;
	this->OffCentreMidSlice = NULL;
	this->FlowCompensation = false;
	this->PreSaturation = false;
	this->PhaseEncodingVelocity = NULL;
	this->MTC = false;
	this->SPIR = false;
	this->EPIFactor = 0;
	this->DynamicScan = false;
	this->Diffusion = false;
	this->DiffusionEchoTime = 0.0;
	this->SliceNrs = NULL;
	this->TotalNrSlices = 0;
	this->SliceNrsBufferSize = 256;

	this->SliceGroups = NULL;

	// Slice parameters.
	this->SliceOrientation = NULL;
	this->PixelSize = 16;
	this->ReconResolution = NULL;
	this->ImageAngulation = NULL;
	this->ImageOffcentre = NULL;
	this->PixelSpacing = NULL;
	this->SliceThickness = 0.0;
	this->BFactor = 0.0;
	this->FlipAngle = 0.0;

	this->NrGradients = 0;
	this->GradientMatrix = NULL;
}

////////////////////////////////////////////////////
DTIPARRECReader::~DTIPARRECReader()
{
}

////////////////////////////////////////////////////
void DTIPARRECReader::LoadPAR(const char *filename)
{
	if(this->Version == 3)
	{
		this->LoadPAR3(filename);
	}
	else if(this->Version == 4)
	{
		this->LoadPAR4(filename);
	}
	else
	{
		printf("DTIPARRECReader::LoadPAR() unknown version\n");
		return;
	}
}

////////////////////////////////////////////////////
void DTIPARRECReader::LoadPAR3(const char *filename)
{
}

////////////////////////////////////////////////////
void DTIPARRECReader::LoadPAR4(const char *filename)
{
	// Open PAR file.
	FILE *f = fopen(filename, "rt");
	if(f == NULL)
	{
		printf("DTIPARRECReader::LoadPAR() could not open PAR(v4) file %s\n", filename);
		return;
	}

	int nrline = 0;
	bool beginDataDescriptionFile = false;
	bool generalInformation = false;
	bool pixelValues = false;
	bool imageInformationDefinition = false;
	bool imageInformation = false;
	bool endDataDescriptionFile = false;

	// Parse file line by line.
	while( ! feof(f))
	{
		char *line = (char *) calloc(256, sizeof(char));
		if(line == NULL)
		{
			printf("DTIPARRECReader::LoadPAR() out of memory allocating string\n");
			return;
		}

		// Read line from file.
		fgets(line, 256, f);
		if(ferror(f))
		{
			printf("DTIPARRECReader::LoadPar() error reading line %d\n", nrline);
			return;
		}

		nrline++;

		//printf("%s", line);

		if(line[0] == '\n')
		{
			continue;
		}
		else if(line[0] == '#')
		{
			// If line contains nothing more, continue.
			if(line[1] == '\n')
			{
				continue;
			}

			if(strstr(strlwr(line), "data description file") != NULL)
			{
				generalInformation = pixelValues = imageInformationDefinition = imageInformation = endDataDescriptionFile = false;
				beginDataDescriptionFile = true;
				continue;
			}
			else if(strstr(strlwr(line), "general information") != NULL)
			{
				beginDataDescriptionFile = pixelValues = imageInformationDefinition = imageInformation = endDataDescriptionFile = false;
				generalInformation = true;
				continue;
			}
			else if(strstr(strlwr(line), "pixel values") != NULL)
			{
				beginDataDescriptionFile = generalInformation = imageInformationDefinition = imageInformation = endDataDescriptionFile = false;
				pixelValues = true;
				continue;
			}
			else if(strstr(strlwr(line), "image information definition") != NULL)
			{
				beginDataDescriptionFile = generalInformation = pixelValues = imageInformation = endDataDescriptionFile = false;
				imageInformationDefinition = true;
				continue;
			}
			else if(strstr(strlwr(line), "image information") != NULL)
			{
				beginDataDescriptionFile = generalInformation = pixelValues = imageInformationDefinition = endDataDescriptionFile = false;
				imageInformation = true;
				continue;
			}
			else if(strstr(strlwr(line), "end of data description file") != NULL)
			{
				beginDataDescriptionFile = generalInformation = pixelValues = imageInformationDefinition = imageInformation = false;
				endDataDescriptionFile = true;
				continue;
			}
			else {}

			if(imageInformationDefinition)
			{
				// Retrieve image information definitions.
			}
			else
			{
				continue;
			}
		}
		else if(line[0] == '.')
		{
			if(generalInformation)
			{
				// Retrieve general information.
				if(strstr(strlwr(line), "patient name") != NULL)			this->PatientName = this->GetTagString(line);
				else if(strstr(strlwr(line), "examination name") != NULL)		this->ExaminationName = this->GetTagString(line);
				else if(strstr(strlwr(line), "protocol name") != NULL)			this->ProtocolName = this->GetTagString(line);
				else if(strstr(strlwr(line), "examination date/time") != NULL)		this->ExaminationDateTime = this->GetTagString(line);
				else if(strstr(strlwr(line), "series type") != NULL)			this->SeriesType = this->GetTagString(line);
				else if(strstr(strlwr(line), "reconstruction nr") != NULL)		this->ReconstructionNr = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "scan duration") != NULL)			this->ScanDuration = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "number of cardiac phases") != NULL)	this->NrCardiacPhases = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "number of echoes") != NULL)		this->NrEchoes = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "number of slices/locations") != NULL)	this->NrSlices = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "number of dynamics") != NULL)		this->NrDynamics = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "number of mixes") != NULL)		this->NrMixes = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "patient position") != NULL)		this->PatientPosition = this->GetTagStringArray(line, 3);
				else if(strstr(strlwr(line), "preparation direction") != NULL)		this->PreparationDirection = this->GetTagString(line);
				else if(strstr(strlwr(line), "scan resolution") != NULL)		this->ScanResolution = this->GetTagIntegerArray(line, 2);
				else if(strstr(strlwr(line), "scan mode") != NULL)			this->ScanMode = this->GetTagString(line);
				else if(strstr(strlwr(line), "repetition time") != NULL)		this->RepetitionTime = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "fov") != NULL)				this->FieldOfView = this->GetTagDoubleArray(line, 3);
				else if(strstr(strlwr(line), "water fat shift") != NULL)		this->WaterFatShift = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "angulation midslice") != NULL)		this->AngulationMidSlice = this->GetTagDoubleArray(line, 3);
				else if(strstr(strlwr(line), "off centre midslice") != NULL)		this->OffCentreMidSlice = this->GetTagIntegerArray(line, 3);
				else if(strstr(strlwr(line), "flow compensation") != NULL)		this->FlowCompensation = this->GetTagBool(line);
				else if(strstr(strlwr(line), "presaturation") != NULL)			this->PreSaturation = this->GetTagBool(line);
				else if(strstr(strlwr(line), "phase encoding velocity") != NULL)	this->PhaseEncodingVelocity = this->GetTagDoubleArray(line, 3);
				else if(strstr(strlwr(line), "mtc") != NULL)				this->MTC = this->GetTagBool(line);
				else if(strstr(strlwr(line), "spir") != NULL)				this->SPIR = this->GetTagBool(line);
				else if(strstr(strlwr(line), "epi factor") != NULL)			this->EPIFactor = this->GetTagInteger(line);
				else if(strstr(strlwr(line), "dynamic scan") != NULL)			this->DynamicScan = this->GetTagBool(line);
				else if(strstr(strlwr(line), "diffusion") != NULL)			this->Diffusion = this->GetTagBool(line);
				else if(strstr(strlwr(line), "diffusion echo time") != NULL)		this->DiffusionEchoTime = this->GetTagDouble(line);

				continue;
			}
		}

		if(imageInformation)
		{
			// Retrieve image slice parameters and store slice number in separate list.
			int sliceNr, echoNr, dynScanNr, cardPhaseNr, typeNr, scanSeq, indexInREC, pixSize, 
				scanPerc, reconRes[2], winCenter, winWidth, imgDispOrient, sliceOrient, fMRIStat, imgTypeEDES, 
				nrAvgs, cardFreq, minRRInt, maxRRInt, turboFact;
			float rescIntercept, rescSlope, scaleSlope, imgAng[3], imgOffcentre[3], sliceThick, sliceGap, pixSpacing[2],
				echoTime, dynScanBeginTime, trigTime, diffBFact, imgFlipAng, invDelay;

			sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d %f %f %f %d %d %f %f %f %f %f %f %f %f %d %d %d %d %f %f %f %f %f %f %d %f %d %d %d %d %f", 
				&sliceNr, &echoNr, &dynScanNr, &cardPhaseNr, &typeNr, &scanSeq, &indexInREC, &pixSize, &scanPerc,
				&reconRes[0], &reconRes[1], &rescIntercept, &rescSlope, &scaleSlope, &winCenter, &winWidth, 
				&imgAng[0], &imgAng[1], &imgAng[2], &imgOffcentre[0], &imgOffcentre[1], &imgOffcentre[2], &sliceThick,
				&sliceGap, &imgDispOrient, &sliceOrient, &fMRIStat, &imgTypeEDES, &pixSpacing[0], &pixSpacing[1],
				&echoTime, &dynScanBeginTime, &trigTime, &diffBFact, &nrAvgs, &imgFlipAng, &cardFreq, &minRRInt, 
				&maxRRInt, &turboFact, &invDelay);

			// Store global parameters.
			if(this->ReconResolution == NULL)
			{
				this->ReconResolution = new int[2];
				if(this->ImageAngulation == NULL) this->ImageAngulation = new double[3];
				if(this->ImageOffcentre == NULL) this->ImageOffcentre = new double[3];
				if(this->PixelSpacing == NULL) this->PixelSpacing = new double[2];

				this->PixelSize = pixSize;
				this->ReconResolution[0] = reconRes[0];
				this->ReconResolution[1] = reconRes[1];
				this->ImageAngulation[0] = imgAng[0];
				this->ImageAngulation[1] = imgAng[1];
				this->ImageAngulation[2] = imgAng[2];
				this->ImageOffcentre[0] = imgOffcentre[0];
				this->ImageOffcentre[1] = imgOffcentre[1];
				this->ImageOffcentre[2] = imgOffcentre[2];
				this->SliceThickness = sliceThick;
				this->PixelSpacing[0] = pixSpacing[0];
				this->PixelSpacing[1] = pixSpacing[1];
				this->BFactor = diffBFact;
				this->FlipAngle = imgFlipAng;
				
				switch(sliceOrient)
				{
					case 0: this->SliceOrientation = "TRA"; break;
					case 1: this->SliceOrientation = "SAG"; break;
					case 2: this->SliceOrientation = "COR"; break;
					default: break;
				}
			}

			// Store the slice number of each slice.
			if(this->SliceNrs == NULL)
			{
				this->SliceNrs = new int[this->SliceNrsBufferSize];
				
			}

			if(this->TotalNrSlices == this->SliceNrsBufferSize)
			{
				// Increase buffer size.
				int size = 2 * this->SliceNrsBufferSize;
				int *tmp = new int[size];
				memcpy(tmp, this->SliceNrs, this->SliceNrsBufferSize * sizeof(int));
				delete [] this->SliceNrs;

				this->SliceNrs = tmp;
				this->SliceNrsBufferSize = size;
			}

			this->SliceNrs[this->TotalNrSlices] = sliceNr;
			this->TotalNrSlices++;
		}
	}

	//for(int i = 0; i < this->TotalNrSlices; i++)
	//{
	//	printf("slice number: %d\n", this->SliceNrs[i]);
	//}

	fclose(f);
}

////////////////////////////////////////////////////
void DTIPARRECReader::LoadREC(const char *filename)
{
	if(this->SliceNrs == NULL)
	{
		printf("DTIPARRECReader::LoadREC() load PAR file first\n");
		return;
	}

	FILE *f = fopen(filename, "rb");
	if(f == NULL)
	{
		printf("DTIPARRECReader::LoadREC() could not open REC file %s\n", filename);
		return;
	}

	if(this->SliceGroups != NULL)
	{
		printf("DTIPARRECReader::LoadREC() slices already loaded\n");
		return;
	}

	for(int i = 0; i < this->TotalNrSlices; i++)
	{
		int sliceNumber = this->SliceNrs[i];
		int size = this->ReconResolution[0] * this->ReconResolution[1];
		unsigned short *pixels = new unsigned short[size];

		// Read image pixels. Quit if not all pixels could be read.
		int n = fread(pixels, sizeof(unsigned short), size, f);
		if(n != size)
		{
			printf("DTIPARRECReader::LoadREC() could load only %d out of %d pixel values\n", n, size);
			break;
		}

		this->AddSlice(pixels, sliceNumber);
		printf("DTIPARRECReader::LoadREC() added slice %d\n", i);
	}

	fclose(f);
}

////////////////////////////////////////////////////
void DTIPARRECReader::LoadGradients(const char *filename, int nr)
{
	if(this->GradientMatrix != NULL)
	{
		gsl_matrix_free(this->GradientMatrix);
		this->GradientMatrix = NULL;
	}

	this->GradientMatrix = gsl_matrix_calloc(nr, 3);

	FILE *f = fopen(filename, "rt");
	if(f == NULL)
	{
		printf("DTIPARRECReader::LoadGradients() could not open gradients file %s\n", filename);
		return;
	}

	int n = 0;
	int nrline = 0;

	while( ! feof(f))
	{
		char *line = (char *) calloc(128, sizeof(char));
		if(line == NULL)
		{
			printf("DTIPARRECReader::LoadGradients() out of memory allocating string\n");
			break;
		}

		fgets(line, 128, f);
		if(ferror(f))
		{
			printf("DTIPARRECReader::LoadGradients() error reading line %d\n", nrline);
			break;
		}

		nrline++;

		if(line[0] == '#' || line[0] == '\n' || line == "" || line[0] == '\0' || line[0] == '\t')
		{
			continue;
		}

		double x = DTIUtils::StringToDouble(DTIUtils::Trim(strtok(line, ",")));
		double y = DTIUtils::StringToDouble(DTIUtils::Trim(strtok(0, ",")));
		double z = DTIUtils::StringToDouble(DTIUtils::Trim(strtok(0, ",")));

		printf("DTIPARRECReader::LoadGradients() %f,%f,%f\n", x, y, z);

		// Normalize.
		double len = sqrt(x*x + y*y + z*z);
		if(len > 0.0)
		{
			x = x / len;
			y = y / len;
			z = z / len;
		}

		gsl_matrix_set(this->GradientMatrix, n, 0, x);
		gsl_matrix_set(this->GradientMatrix, n, 1, y);
		gsl_matrix_set(this->GradientMatrix, n, 2, z);
		n++;

	}

	fclose(f);
}

////////////////////////////////////////////////////
vector<DTISliceGroup *> *DTIPARRECReader::GetOutput()
{
	if(this->SliceGroups == NULL)
	{
		printf("DTIPARRECReader::GetOutput() no output available\n");
	}

	return this->SliceGroups;
}

////////////////////////////////////////////////////
gsl_matrix *DTIPARRECReader::GetGradients()
{
	if(this->GradientMatrix == NULL)
	{
		printf("DTIPARRECReader::GetGradients() gradients not loaded yet\n");
		return NULL;
	}

	return this->GradientMatrix;
}

////////////////////////////////////////////////////
gsl_matrix *DTIPARRECReader::GetGradientTransform()
{
	gsl_matrix *m = gsl_matrix_calloc(3, 3);
	
	// Identity matrix.
	gsl_matrix_set(m, 0, 0, 1.0);
	gsl_matrix_set(m, 0, 1, 0.0);
	gsl_matrix_set(m, 0, 2, 0.0);
	gsl_matrix_set(m, 1, 0, 0.0);
	gsl_matrix_set(m, 1, 1, 1.0);
	gsl_matrix_set(m, 1, 1, 0.0);
	gsl_matrix_set(m, 2, 0, 0.0);
	gsl_matrix_set(m, 2, 1, 0.0);
	gsl_matrix_set(m, 2, 2, 1.0);

	return m;
}

////////////////////////////////////////////////////
void DTIPARRECReader::AddSlice(unsigned short *pixels, int slicenr)
{
	if(this->SliceGroups == NULL)
	{
		this->SliceGroups = new vector<DTISliceGroup *>;
	}

	// Create new slice object.
	DTISlice *slice = new DTISlice();
	slice->SetRows(this->ReconResolution[0]);	// Use reconstruction resolution instead of scan resolution.
	slice->SetColumns(this->ReconResolution[1]);	// Idem.
	slice->SetPixelSpacing(this->PixelSpacing[0], this->PixelSpacing[1]);
	slice->SetData(pixels);
	slice->SetSliceLocation((double) slicenr);
	slice->SetSliceThickness(this->SliceThickness);

	// Find slice group with given slice number (location).
	DTISliceGroup *slicegroup = this->FindSliceGroup(slicenr);
	
	if(slicegroup == NULL)
	{
		slicegroup = new DTISliceGroup();
		slicegroup->SetSliceLocation((double) slicenr);

		this->SliceGroups->push_back(slicegroup);
	}

	// Slice group was either found or created, so add the slice.
	slicegroup->AddSlice(slice);
}

////////////////////////////////////////////////////
DTISliceGroup *DTIPARRECReader::FindSliceGroup(int slicenr)
{
	DTISliceGroup *slicegroup = NULL;
	vector<DTISliceGroup *>::iterator iter;

	for(iter = this->SliceGroups->begin(); iter != this->SliceGroups->end(); iter++)
	{
		DTISliceGroup *tmp = (DTISliceGroup *) (*iter);

		if(tmp->GetSliceLocation() == ((double) slicenr))
		{
			slicegroup = tmp;
			break;
		}
	}

	return slicegroup;
}

////////////////////////////////////////////////////
int DTIPARRECReader::GetTagInteger(char *line, char *delim)
{
	char *token = strtok(line, delim);
	token = strtok(0, delim);
	return DTIUtils::StringToInteger(DTIUtils::Trim1(token));
}

////////////////////////////////////////////////////
int *DTIPARRECReader::GetTagIntegerArray(char *line, int size, char *delim)
{
	char *token = strtok(line, delim);
	token = strtok(0, delim);
	return DTIUtils::StringToIntegerArray(DTIUtils::Trim1(token), " ");
}

////////////////////////////////////////////////////
double DTIPARRECReader::GetTagDouble(char *line, char *delim)
{
	char *token = strtok(line, delim);
	token = strtok(0, delim);
	return DTIUtils::StringToDouble(DTIUtils::Trim1(token));
}

////////////////////////////////////////////////////
double *DTIPARRECReader::GetTagDoubleArray(char *line, int size, char *delim)
{
	char *token = strtok(line, delim);
	token = strtok(0, delim);
	return DTIUtils::StringToDoubleArray(DTIUtils::Trim1(token), " ");
}

////////////////////////////////////////////////////
char *DTIPARRECReader::GetTagString(char *line, char *delim)
{
	char *token = strtok(line, delim);
	token = strtok(0, delim);
	return DTIUtils::Trim1(token);
}

////////////////////////////////////////////////////
char **DTIPARRECReader::GetTagStringArray(char *line, int size, char *delim)
{
	char *token = strtok(line, delim);
	token = strtok(0, delim);
	return DTIUtils::StringToStringArray(DTIUtils::Trim1(token), " ");
}

////////////////////////////////////////////////////
bool DTIPARRECReader::GetTagBool(char *line, char *delim)
{
	char *token = strtok(line, delim);
	token = strtok(0, delim);
	return DTIUtils::StringToBool(DTIUtils::Trim1(token));
}

////////////////////////////////////////////////////
void DTIPARRECReader::PrintSelf()
{
	if(this->PatientName != NULL)
	{
		printf("PAR DATA\n");
		printf("========\n");
		printf("PatientName: %s\n", this->PatientName);
		printf("ExaminationName: %s\n", this->ExaminationName);
		printf("ProtocolName: %s\n", this->ProtocolName);
		printf("ExaminationDateTime: %s\n", this->ExaminationDateTime);
		printf("SeriesType: %s\n", this->SeriesType);
		printf("ReconstructionNr: %d\n", this->ReconstructionNr);
		printf("ScanDuration: %d\n", this->ScanDuration);
		printf("NrCardiacPhases: %d\n", this->NrCardiacPhases);
		printf("NrEchoes: %d\n", this->NrEchoes);
		printf("NrSlices: %d\n", this->NrSlices);
		printf("NrDynamics: %d\n", this->NrDynamics);
		printf("NrMixes: %d\n", this->NrMixes);
		printf("PatientPosition: %s %s %s\n", this->PatientPosition[0], this->PatientPosition[1], this->PatientPosition[2]);
		printf("PreparationDirection: %s\n", this->PreparationDirection);
		printf("ScanResolution: %d %d\n", this->ScanResolution[0], this->ScanResolution[1]);
		printf("ScanMode: %s\n", this->ScanMode);
		printf("RepetitionTime: %d\n", this->RepetitionTime);
		printf("FieldOfView: %f %f %f\n", this->FieldOfView[0], this->FieldOfView[1], this->FieldOfView[2]);
		printf("WaterFatShift: %d\n", this->WaterFatShift);
		printf("AngulationMidSlice: %f %f %f\n", this->AngulationMidSlice[0], this->AngulationMidSlice[1], this->AngulationMidSlice[2]);
		printf("OffCentreMidSlice: %d %d %d\n", this->OffCentreMidSlice[0], this->OffCentreMidSlice[1], this->OffCentreMidSlice[2]);
		printf("FlowCompensation: %d\n", this->FlowCompensation);
		printf("PreSaturation: %d\n", this->PreSaturation);
		printf("PhaseEncodingVelocity: %f %f %f\n", this->PhaseEncodingVelocity[0], this->PhaseEncodingVelocity[1], this->PhaseEncodingVelocity[2]);
		printf("MTC: %d\n", this->MTC);
		printf("SPIR: %d\n", this->SPIR);
		printf("EPIFactor: %d\n", this->EPIFactor);
		printf("DynamicScan: %d\n", this->DynamicScan);
		printf("Diffusion: %d\n", this->Diffusion);
		printf("DiffusionEchoTime: %f\n", this->DiffusionEchoTime);
		printf("\n");
		printf("TotalNrSlices: %d\n", this->TotalNrSlices);
		printf("\n");
		printf("PixelSize: %d\n", this->PixelSize);
		printf("ReconResolution: %d %d\n", this->ReconResolution[0], this->ReconResolution[1]);
		printf("ImageAngulation: %f %f %f\n", this->ImageAngulation[0], this->ImageAngulation[1], this->ImageAngulation[2]);
		printf("ImageOffcentre: %f %f %f\n", this->ImageOffcentre[0], this->ImageOffcentre[1], this->ImageOffcentre[2]);
		printf("SliceThickness: %f\n", this->SliceThickness);
		printf("PixelSpacing: %f %f\n", this->PixelSpacing[0], this->PixelSpacing[1]);
		printf("BFactor: %f\n", this->BFactor);
		printf("FlipAngle: %f\n", this->FlipAngle);
		printf("SliceOrientation: %s\n", this->SliceOrientation);
		printf("\n");
	}

	if(this->SliceGroups != NULL && this->SliceGroups->size() > 0)
	{
		printf("REC DATA\n");
		printf("========\n");
		printf("NrSliceGroups: %d\n", this->SliceGroups->size());
		
		vector<DTISliceGroup *>::iterator iter;
		for(iter = this->SliceGroups->begin(); iter != this->SliceGroups->end(); iter++)
		{
			DTISliceGroup *slicegroup = (DTISliceGroup *) (*iter);
			printf("SliceGroup%d: %d slices\n", ((int) slicegroup->GetSliceLocation()), slicegroup->GetSize());
		}
	}
}

////////////////////////////////////////////////////
void DTIPARRECReader::SetVersion(int version)
{
	if(version != 3 || version != 4)
	{
		printf("DTIPARRECReader::SetVersion: version should be 3 or 4\n");
		return;
	}

	this->Version = version;
}
