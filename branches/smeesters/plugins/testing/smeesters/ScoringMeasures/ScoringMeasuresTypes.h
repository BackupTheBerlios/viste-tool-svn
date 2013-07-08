#ifndef bmia_ScoringMeasuresTypes_h
#define bmia_ScoringMeasuresTypes_h

namespace bmia {

/** Holding parameter settings */
typedef struct
{
    bool useGlyphData;
    bool normalizeScalars;

    double lambda;
    double beta;
    double muu;
    int typeOfCurve;

} ParameterSettings;

}

#endif  // bmia_ScoringMeasuresTypes_h
