#ifndef bmia_ScoringTypes_h
#define bmia_ScoringTypes_h

namespace bmia {

/** Holding threshold settings */
typedef struct
{
    bool set;
    double averageScore[2];
    double scalarRange[2];
    double globalSetting[2];

} ThresholdSettings;

}

#endif  // bmia_ScoringTypes_h
