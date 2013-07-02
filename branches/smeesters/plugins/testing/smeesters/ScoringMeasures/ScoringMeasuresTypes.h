#ifndef bmia_ScoringMeasuresTypes_h
#define bmia_ScoringMeasuresTypes_h

namespace bmia {

/** Holding parameter settings */
typedef struct
{
    bool useGlyphData;
    double lambda;
    double beta;
    double muu;
    int typeOfCurve;

} ParameterSettings;

enum TypeOfCurve
{
    CURVE_TYPE_GEODESIC = 0,
    CURVE_TYPE_ELASTICA
};

}

#endif  // bmia_ScoringMeasuresTypes_h
