#ifndef __DTIPARRECReader_h
#define __DTIPARRECReader_h

#include "DTISliceGroup.h"
#include "DTISlice.h"
#include <vector>
#include "gsl/gsl_linalg.h"

using namespace std;

//---------------------------------------------------------------------------
//! \file   DTIPARRECReader.h
//! \class  DTIPARRECReader
//! \author Ralph Brecheisen
//! \brief  Reads Philips PAR/REC Diffusion Tensor Imaging files.
//---------------------------------------------------------------------------
class DTIPARRECReader
{
public:
	//! Constructor.
	DTIPARRECReader();
	//! Destructor.
	~DTIPARRECReader();

	//! Loads PAR header from file.
	virtual void LoadPAR(const char *filename);
	//! Loads REC data.
	virtual void LoadREC(const char *filename);
	//! Loads gradient directions.
	virtual void LoadGradients(const char *filename, int nr);

	//! Set version.
	virtual void SetVersion(int version);
	//! Print.
	virtual void PrintSelf();

	//! Returns output data as collection of DTISliceGroup objects.
	virtual vector<DTISliceGroup *> *GetOutput();
	//! Returns gradient matrix needed for tensor calculation.
	virtual gsl_matrix *GetGradients();
	//! Returns gradient transformation matrix.
	virtual gsl_matrix *GetGradientTransform();

protected:
	//! Loads version 3 PAR header from file.
	virtual void LoadPAR3(const char *filename);
	//! Loads version 4 PAR header from file.
	virtual void LoadPAR4(const char *filename);

	//! Adds slice to list of slice groups.
	virtual void AddSlice(unsigned short *pixels, int slicenr);
	//! Finds slice group with given slice number (location).
	virtual DTISliceGroup *FindSliceGroup(int slicenr);

	//! Get tag value (integer).
	virtual int GetTagInteger(char *line, char *delim = ":");
	//! Get tag value (integer array).
	virtual int *GetTagIntegerArray(char *line, int size, char *delim = ":");
	//! Get tag value (double).
	virtual double GetTagDouble(char *line, char *delim = ":");
	//! Get tag value (double array).
	virtual double *GetTagDoubleArray(char *line, int size, char *delim = ":");
	//! Get tag value (string).
	virtual char *GetTagString(char *line, char *delim = ":");
	//! Get tag value (string array).
	virtual char **GetTagStringArray(char *line, int size, char *delim = ":");
	//! Get tag value (bool).
	virtual bool GetTagBool(char *line, char *delim = ":");

	// Other parameters.
	int TotalNrSlices;
	int Version;
	int SliceNrsBufferSize;
	int *SliceNrs;
	// PAR/REC parameters.
	int ReconstructionNr;
	int ScanDuration;
	int NrCardiacPhases;
	int NrEchoes;
	int NrSlices;
	int NrDynamics;
	int NrMixes;
	int WaterFatShift;
	int RepetitionTime;
	int EPIFactor;
	int *ScanResolution;
	int *OffCentreMidSlice;
	char *PatientName;
	char *ExaminationName;
	char *ProtocolName;
	char *ExaminationDateTime;
	char *SeriesType;
	char *PreparationDirection;
	char *ScanMode;
	char **PatientPosition;
	double *FieldOfView;
	double *AngulationMidSlice;
	double *PhaseEncodingVelocity;
	bool FlowCompensation;
	bool PreSaturation;
	bool MTC;
	bool SPIR;
	bool DynamicScan;
	bool Diffusion;
	double DiffusionEchoTime;

	vector<DTISliceGroup *> *SliceGroups;

	char *SliceOrientation;
	int PixelSize;
	int *ReconResolution;
	double *ImageAngulation;
	double *ImageOffcentre;
	double *PixelSpacing;
	double SliceThickness;
	double BFactor;
	double FlipAngle;

	int NrGradients;
	gsl_matrix *GradientMatrix;
};

#endif