#include <DistanceConfiguration.h>

#include <vtkLookupTable.h>
#include <vtkTexture.h>
#include <vtkMath.h>

namespace bmia
{
///////////////////////////////////////////////////////////////////////////
	DistanceConfiguration::DistanceConfiguration( QString name )
	{
		_name = name;
		_selectedVoronoiData = 0;
		_selectedThreshold = -1;
		_selectedDistance = -1;
		_selectedColorScaleName = "Grayscale";
		_minimumDistanceRange[0] = -1;
		_minimumDistanceRange[1] = -1;
		_maximumDistanceRange[0] = -1;
		_maximumDistanceRange[1] = -1;
		_selectedPointMarkerPosition[0] = -1;
		_selectedPointMarkerPosition[1] = -1;
		_selectedPointMarkerPosition[2] = -1;
		_tumorCentroid[0] = -1;
		_tumorCentroid[1] = -1;
		_tumorCentroid[2] = -1;
		_fiberCentroid[0] = -1;
		_fiberCentroid[1] = -1;
		_fiberCentroid[2] = -1;
		_tumorMinMaxRadius[0] = -1;
		_tumorMinMaxRadius[1] = -1;
		_thresholdRange[0] = -1;
		_thresholdRange[1] = -1;
		_riskRadius = -1;
		_spacing[0] = -1;
		_spacing[1] = -1;
		_spacing[2] = -1;
		_projectionEnabled = false;
		_riskRadiusEnabled = false;
		_contoursEnabled = false;
		_fiberData = 0;
		_tumorData = 0;
		_tumorActor = 0;
		_tumorVoxelPositions = 0;
		_fiberVoxelPositions = 0;
		_numberOfTumorVoxelPositions = -1;
		_numberOfFiberVoxelPositions = -1;
		_numberOfContours = -1;
		_dimensions[0] = -1;
		_dimensions[1] = -1;
		_dimensions[2] = -1;
	}

	///////////////////////////////////////////////////////////////////////////
	DistanceConfiguration::~DistanceConfiguration()
	{
	}

	///////////////////////////////////////////////////////////////////////////
	QString DistanceConfiguration::getName()
	{
		return _name;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setThresholdRange( double min, double max )
	{
		_thresholdRange[0] = min;
		_thresholdRange[1] = max;
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getThresholdMin()
	{
		return _thresholdRange[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getThresholdMax()
	{
		return _thresholdRange[1];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setSelectedDistance( double distance )
	{
		_selectedDistance = distance;
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSelectedDistance()
	{
		return _selectedDistance;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setSelectedThreshold( double threshold )
	{
		_selectedThreshold = threshold;
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSelectedThreshold()
	{
		return _selectedThreshold;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setSelectedColorScaleName( QString name )
	{
		_selectedColorScaleName = name;
	}

	///////////////////////////////////////////////////////////////////////////
	QString DistanceConfiguration::getSelectedColorScaleName()
	{
		return _selectedColorScaleName;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setSelectedVoronoiData( vtkImageData * voronoiData )
	{
		if( _selectedVoronoiData )
			_selectedVoronoiData->Delete();
		_selectedVoronoiData = vtkImageData::New();
		_selectedVoronoiData->DeepCopy( voronoiData );
	}

	///////////////////////////////////////////////////////////////////////////
	vtkImageData * DistanceConfiguration::getSelectedVoronoiData()
	{
		return _selectedVoronoiData;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setSelectedPointMarkerPosition( double position[3] )
	{
		_selectedPointMarkerPosition[0] = position[0];
		_selectedPointMarkerPosition[1] = position[1];
		_selectedPointMarkerPosition[2] = position[2];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSelectedPointMarkerPositionX()
	{
		return _selectedPointMarkerPosition[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSelectedPointMarkerPositionY()
	{
		return _selectedPointMarkerPosition[1];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSelectedPointMarkerPositionZ()
	{
		return _selectedPointMarkerPosition[2];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setMinimumDistanceRange( double min, double max )
	{
		_minimumDistanceRange[0] = min;
		_minimumDistanceRange[1] = max;
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getMinimumDistanceMin()
	{
		return _minimumDistanceRange[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getMinimumDistanceMax()
	{
		return _minimumDistanceRange[1];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setMaximumDistanceRange( double min, double max )
	{
		_maximumDistanceRange[0] = min;
		_maximumDistanceRange[1] = max;
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getMaximumDistanceMin()
	{
		return _maximumDistanceRange[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getMaximumDistanceMax()
	{
		return _maximumDistanceRange[1];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setTumorCentroid( double centroid[3] )
	{
		_tumorCentroid[0] = centroid[0];
		_tumorCentroid[1] = centroid[1];
		_tumorCentroid[2] = centroid[2];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getTumorCentroidX()
	{
		return _tumorCentroid[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getTumorCentroidY()
	{
		return _tumorCentroid[1];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getTumorCentroidZ()
	{
		return _tumorCentroid[2];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setFiberCentroid( double centroid[3] )
	{
		_fiberCentroid[0] = centroid[0];
		_fiberCentroid[1] = centroid[1];
		_fiberCentroid[2] = centroid[2];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getFiberCentroidX()
	{
		return _fiberCentroid[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getFiberCentroidY()
	{
		return _fiberCentroid[1];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getFiberCentroidZ()
	{
		return _fiberCentroid[2];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setTumorMinMaxRadius( double minRadius, double maxRadius )
	{
		_tumorMinMaxRadius[0] = minRadius;
		_tumorMinMaxRadius[1] = maxRadius;
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getTumorMinRadius()
	{
		return _tumorMinMaxRadius[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getTumorMaxRadius()
	{
		return _tumorMinMaxRadius[1];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setRiskRadius( double radius )
	{
		_riskRadius = radius;
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getRiskRadius()
	{
		return _riskRadius;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setRiskRadiusEnabled( bool enabled )
	{
		_riskRadiusEnabled = enabled;
	}

	///////////////////////////////////////////////////////////////////////////
	bool DistanceConfiguration::isRiskRadiusEnabled()
	{
		return _riskRadiusEnabled;
	}

    ///////////////////////////////////////////////////////////////////////////
    void DistanceConfiguration::setRiskRadiusUncertaintyEnabled( bool enabled )
    {
        _riskRadiusUncertaintyEnabled = enabled;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool DistanceConfiguration::isRiskRadiusUncertaintyEnabled()
    {
        return _riskRadiusUncertaintyEnabled;
    }

    ///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setProjectionEnabled( bool enabled )
	{
		_projectionEnabled = enabled;
	}

	///////////////////////////////////////////////////////////////////////////
	bool DistanceConfiguration::isProjectionEnabled()
	{
		return _projectionEnabled;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setContoursEnabled( bool enabled )
	{
		_contoursEnabled = enabled;
	}

	///////////////////////////////////////////////////////////////////////////
	bool DistanceConfiguration::isContoursEnabled()
	{
		return _contoursEnabled;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setNumberOfContours( int numberOfContours )
	{
		_numberOfContours = numberOfContours;
	}

	///////////////////////////////////////////////////////////////////////////
	int DistanceConfiguration::getNumberOfContours()
	{
		return _numberOfContours;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setDimensions( int dimensions[3] )
	{
		_dimensions[0] = dimensions[0];
		_dimensions[1] = dimensions[1];
		_dimensions[2] = dimensions[2];
	}

	///////////////////////////////////////////////////////////////////////////
	int DistanceConfiguration::getDimensionX()
	{
		return _dimensions[0];
	}

	///////////////////////////////////////////////////////////////////////////
	int DistanceConfiguration::getDimensionY()
	{
		return _dimensions[1];
	}

	///////////////////////////////////////////////////////////////////////////
	int DistanceConfiguration::getDimensionZ()
	{
		return _dimensions[2];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setSpacing( double spacing[3] )
	{
		_spacing[0] = spacing[0];
		_spacing[1] = spacing[1];
		_spacing[2] = spacing[2];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSpacingX()
	{
		return _spacing[0];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSpacingY()
	{
		return _spacing[1];
	}

	///////////////////////////////////////////////////////////////////////////
	double DistanceConfiguration::getSpacingZ()
	{
		return _spacing[2];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setTumorData( vtkImageData * tumorData )
	{
		_tumorData = tumorData;
	}

	///////////////////////////////////////////////////////////////////////////
	vtkImageData * DistanceConfiguration::getTumorData()
	{
		return _tumorData;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setFiberData( vtkImageData * fiberData )
	{
		_fiberData = fiberData;
	}

	///////////////////////////////////////////////////////////////////////////
	vtkImageData * DistanceConfiguration::getFiberData()
	{
		return _fiberData;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setTumorActor( vtkActor * tumorActor )
	{
		_tumorActor = tumorActor;
	}

	///////////////////////////////////////////////////////////////////////////
	vtkActor * DistanceConfiguration::getTumorActor()
	{
		return _tumorActor;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setTumorVoxelPositions( int ** positions, int numberOfPositions )
	{
		_tumorVoxelPositions = positions;
		_numberOfTumorVoxelPositions = numberOfPositions;
	}

	///////////////////////////////////////////////////////////////////////////
	int ** DistanceConfiguration::getTumorVoxelPositions()
	{
		return _tumorVoxelPositions;
	}

	///////////////////////////////////////////////////////////////////////////
	int DistanceConfiguration::getNumberOfTumorVoxelPositions()
	{
		return _numberOfTumorVoxelPositions;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::setFiberVoxelPositions( int ** positions, int numberOfPositions )
	{
		_fiberVoxelPositions = positions;
		_numberOfFiberVoxelPositions = numberOfPositions;
	}

	///////////////////////////////////////////////////////////////////////////
	int ** DistanceConfiguration::getFiberVoxelPositions()
	{
		return _fiberVoxelPositions;
	}

	///////////////////////////////////////////////////////////////////////////
	int DistanceConfiguration::getNumberOfFiberVoxelPositions()
	{
		return _numberOfFiberVoxelPositions;
	}

	///////////////////////////////////////////////////////////////////////////
	QList< double > & DistanceConfiguration::getMinimumDistances()
	{
		return _minimumDistances;
	}

	///////////////////////////////////////////////////////////////////////////
	QList< double > & DistanceConfiguration::getMaximumDistances()
	{
		return _maximumDistances;
	}

	///////////////////////////////////////////////////////////////////////////
	QList< QPointF > & DistanceConfiguration::getThresholdDistancePoints()
	{
		return _thresholdDistancePoints;
	}

	///////////////////////////////////////////////////////////////////////////
	QList< double * > & DistanceConfiguration::getMinimumDistanceStartPoints()
	{
		return _minimumDistanceStartPoints;
	}

	///////////////////////////////////////////////////////////////////////////
	QList< double * > & DistanceConfiguration::getMinimumDistanceEndPoints()
	{
		return _minimumDistanceEndPoints;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::applyTexture( vtkTexture * texture )
	{
		Q_ASSERT( _tumorActor );
		_tumorActor->SetTexture( texture );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::updateTexture()
	{
		vtkTexture * texture = _tumorActor->GetTexture();
		if( texture == 0 )
			return;
		this->updateTextureLookupTable( texture );
		this->applyTexture( texture );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::updateTextureLookupTable( vtkTexture * texture )
	{
		Q_ASSERT( texture );

		vtkLookupTable * lut = (vtkLookupTable *) texture->GetLookupTable();
		Q_ASSERT( lut );

		double * range = lut->GetRange();
		double step = 256 / _numberOfContours;
		int index = 255 * (_riskRadius - range[0]) / (range[1] -range[0]);

		for( int i = 0; i < 256; ++i )
		{
			if( i < index && _riskRadiusEnabled )
			{
				lut->SetTableValue( i, 1.0, 0.0, 0.0 );
			}
			else
			{
                double value = (double) i;

				if( _contoursEnabled )
				{
					int level = (int) floor( i / step );
                    value = level * step;
                }

                if( _selectedColorScaleName == "Grayscale" )
                {
                    lut->SetTableValue( i, value / 255.0, value / 255.0, value / 255.0 );
                }
                else if( _selectedColorScaleName == "HeatMap" )
                {
                    // Make step size little bit smaller so we don't
                    // wrap around to red color agains but stop at blue
                    double HSV[3];
                    HSV[0] = value / (255.0 / 0.75);
                    HSV[1] = 1.0;
                    HSV[2] = 1.0;

                    double RGB[3];
                    vtkMath::HSVToRGB( HSV, RGB );

                    lut->SetTableValue( i, RGB[0], RGB[1], RGB[2] );
                }
                else if( _selectedColorScaleName == "CoolToWarm" )
                {
                    double blueRGB[3], blueHSV[3];
                    blueRGB[0] = 0.0;
                    blueRGB[1] = 102.0 / 255.0;
                    blueRGB[2] = 255.0 / 255.0;
                    vtkMath::RGBToHSV( blueRGB, blueHSV );

                    double orangeRGB[3], orangeHSV[3];
                    orangeRGB[0] = 255.0 / 255.0;
                    orangeRGB[1] = 102.0 / 255.0;
                    orangeRGB[2] = 0.0;
                    vtkMath::RGBToHSV( orangeRGB, orangeHSV );

                    double RGB[3];

                    if( value <= 127.0 )
                    {
                        double HSV[3];
                        HSV[0] = orangeHSV[0];
                        HSV[1] = orangeHSV[1] - value * orangeHSV[1] / 127.0;
                        HSV[2] = orangeHSV[2] + value * (1.0 - orangeHSV[2]) / 127.0;
                        vtkMath::HSVToRGB( HSV, RGB );
                    }
                    else
                    {
                        double HSV[3];
                        HSV[0] = blueHSV[0];
                        HSV[1] = (value - 127.0) * (blueHSV[1] / 127.0);
                        HSV[2] = 1.0 - (value - 127.0) * (1.0 - blueHSV[2]) / 127.0;
                        vtkMath::HSVToRGB( HSV, RGB );
                    }

                    lut->SetTableValue( i, RGB[0], RGB[1], RGB[2] );
                }

//				}
//				else
//				{
//					if( _selectedColorScaleName == "Grayscale" )
//					{
//						lut->SetTableValue( i, i / 255.0, i / 255.0, i / 255.0 );
//					}
//					else
//					{
//						// Make step size little bit smaller so we don't
//						// wrap around to red color agains but stop at blue
//						double HSV[3];
//						HSV[0] = i / (255.0 / 0.75);
//						HSV[1] = 1.0;
//						HSV[2] = 1.0;

//						double RGB[3];
//						vtkMath::HSVToRGB( HSV, RGB );

//						lut->SetTableValue( i, RGB[0], RGB[1], RGB[2] );
//					}
//				}
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////
	bool DistanceConfiguration::isValid()
	{
		if( this->getName().isEmpty() ) return false;
		if( this->getMinimumDistanceMin() < 0 ) return false;
		if( this->getMinimumDistanceMax() < 0 ) return false;
		if( this->getMaximumDistanceMin() < 0 ) return false;
		if( this->getMaximumDistanceMax() < 0 ) return false;
		if( this->getTumorCentroidX() < 0 ) return false;
		if( this->getTumorCentroidY() < 0 ) return false;
		if( this->getTumorCentroidZ() < 0 ) return false;
		if( this->getFiberCentroidX() < 0 ) return false;
		if( this->getFiberCentroidY() < 0 ) return false;
		if( this->getFiberCentroidZ() < 0 ) return false;
		if( this->getTumorMinRadius() < 0 ) return false;
		if( this->getTumorMaxRadius() < 0 ) return false;
		if( this->getThresholdMin() < 0 ) return false;
		if( this->getThresholdMax() < 0 ) return false;
		if( this->getRiskRadius() < 0 ) return false;
		if( this->getSpacingX() < 0 ) return false;
		if( this->getSpacingY() < 0 ) return false;
		if( this->getSpacingZ() < 0 ) return false;

		if( this->getFiberData() == 0 ) return false;
		if( this->getTumorData() == 0 ) return false;
		if( this->getTumorActor() == 0 ) return false;
		if( this->getTumorVoxelPositions() == 0 ) return false;
		if( this->getFiberVoxelPositions() == 0 ) return false;

		if( this->getNumberOfContours() < 0 ) return false;
		if( this->getNumberOfFiberVoxelPositions() < 0 ) return false;
		if( this->getNumberOfTumorVoxelPositions() < 0 ) return false;
		if( this->getDimensionX() == 0 ) return false;
		if( this->getDimensionY() == 0 ) return false;
		if( this->getDimensionZ() == 0 ) return false;

		if( this->getThresholdDistancePoints().size() == 0 ) return false;
		if( this->getMinimumDistanceStartPoints().size() == 0 ) return false;
		if( this->getMinimumDistanceEndPoints().size() == 0 ) return false;
		if( this->getMinimumDistances().size() == 0 ) return false;
		if( this->getMaximumDistances().size() == 0 ) return false;

		if( this->getSelectedDistance() < 0 ) return false;
		if( this->getSelectedThreshold() < 0 ) return false;

		return true;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceConfiguration::printConfig()
	{
	}
}
