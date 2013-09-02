/** Class for detecting the maxima in a glyph */

#include "MaximumFinder.h"
 
//--------------------------[ Find maxima for discrete sphere data]--------------------------\\

namespace bmia {

	void MaximumFinder::getOutputDS(double* pDarraySH, int shOrder,double treshold, std::vector<double*> anglesArray,  std::vector<int>& indexesOfMaxima, std::vector<int> &input)
	{
		//cout << "Max Finder Get indexesOfMaxima with treshold starts. tresh:" << treshold<<  endl;

		//clear the indexesOfMaxima
		indexesOfMaxima.clear();

		//get radii.
		//this->radii = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &anglesArray, shOrder);

		for (int i=0; i< shOrder; i++) {

			this->radii.push_back(pDarraySH[i]); // was zero, I made i
			//cout << pDarraySH[i] << " " ;
		}

		//list with neighborhood
		std::vector<int> neighborslist;

		//find maximum and minimum radii
		double min = *min_element((this->radii).begin(), (this->radii).end());
		double max = *max_element((this->radii).begin(), (this->radii).end());

		//min-max normalization
		//for all points on the sphere
		(this->radii_norm).clear();
		for (unsigned int i = 0; i < (this->radii).size(); ++i)
		{
			//dont divide by 0
			if (min != max)
			{
				//add normalized radius
				this->radii_norm.push_back(((this->radii[i])-min)/(max-min));
			}
			//in case of sphere (SH-order = 0)
			else
			{	//add 1.0 (sphere)    
				this->radii_norm.push_back(1.0);
			}
		}

		//for all points on the sphere
		for (unsigned int i = 0; i < (input.size()); ++i)
		{
			//get current radius
			double currentPointValue = this->radii_norm[(input[i])];
			//cout << "radii-N[" << i << "]:"<<  currentPointValue ;
			//if the value is high enough
			if (currentPointValue > (treshold))
			{
				//get the neighbors 
				getNeighbors(input[i], 1, neighborslist); // dene 1 komsuluk yeterli mi
				double currentMax = 0.0;

				//find the highest valued neighbor
				for (unsigned int j = 0; j < neighborslist.size(); ++j)
				{
					if ((this->radii_norm)[(neighborslist[j])]>currentMax)
					{
						currentMax = this->radii_norm[(neighborslist[j])];
					}	
				}
				//if the current point value is higher than the highest valued neighbor
				if (currentMax <= currentPointValue)
				{
					//add point to the indexesOfMaxima
					indexesOfMaxima.push_back(input[i]);
				}	
			}
		}	
		//	cout << "\n Max Finder Get output with treshold ends" << endl;
	}

	//--------------------------[ Find maxima for discrete sphere data]--------------------------\\

	void MaximumFinder::getOutputDS(double* pDarraySH, int shOrder, std::vector<double*> anglesArray)
	{
		//cout << " Max Finder Get output without treshold starts" << endl;
		//get radii
		//this->radii =  bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &anglesArray, shOrder);

		for (int i=0; i< shOrder; i++) {

			this->radii.push_back(pDarraySH[i]);
			//cout << pDarraySH[i] << " " ;
		}

		//list with neighborhood
		std::vector<int> neighborslist;

		//find maximum and minimum radii
		double min = *min_element((this->radii).begin(), (this->radii).end());
		double max = *max_element((this->radii).begin(), (this->radii).end());
		//min-max normalization
		//for all points on the sphere
		(this->radii_norm).clear();
		for (unsigned int i = 0; i < (this->radii).size(); ++i)
		{
			//dont divide by 0
			if (min != max)
			{
				//add normalized radius
				this->radii_norm.push_back(((this->radii[i])-min)/(max-min));
			}
			//in case of sphere (SH-order = 0)
			else
			{	//add 1.0 (sphere)    
				this->radii_norm.push_back(1.0);
			}
		}
	}




	//-----------------------------[ Get neighbors ]------------------------------\\

	void MaximumFinder::getNeighbors(int i, int depth, std::vector<int> &neighborlist_final)
	{
		//clear output vector
		neighborlist_final.clear();
		//set point i as input and find all direct neighbors of point i
		std::vector<int> input(1, i);
		this->getDirectNeighbors(input, neighborlist_final);

		if (depth > 1) //depth=2 or depth=3
		{
			std::vector<int> neighborslist2;
			//find direct neighbors
			this->getDirectNeighbors(neighborlist_final,  neighborslist2);
			for (unsigned int index21 = 0; index21<neighborslist2.size(); index21++)
			{
				//add them to the output list
				neighborlist_final.push_back(neighborslist2[index21]);
			}
		}
		//remove duplicates from the list
		std::sort(neighborlist_final.begin(), neighborlist_final.end());
		neighborlist_final.erase(std::unique(neighborlist_final.begin(), neighborlist_final.end()), neighborlist_final.end());
		//remove seedpoint i from the list
		neighborlist_final.erase(std::remove(neighborlist_final.begin(), neighborlist_final.end(), i), neighborlist_final.end());
	}

	//-----------------------------[ Get direct neighbors ]------------------------------\\

	void MaximumFinder::getDirectNeighbors(std::vector<int> seedpoints, std::vector<int> &temp_neighbors)
	{ 
		// get direct neighbour points in the triangles array to the seedpoints (keept as indexes actually) given
		// namely get the other corners of the triangle if the input seed is a vertex of a trianble

		//clear output vector
		temp_neighbors.clear();
		//for every seed point
		for (unsigned int k =0; k<seedpoints.size(); k++)
		{
			//for every triangle
			for (int j=0;j<(this->trianglesArray->GetNumberOfTuples());j++)
			{
				//set index
				vtkIdType index = j;
				//get triangle
				double * triangle=this->trianglesArray->GetTuple(index);
				//if it contains the seedpoint index
				if ((triangle[0]==seedpoints[k]) || (triangle[1]==seedpoints[k]) || (triangle[2]==seedpoints[k]))
				{
					//add all triangle members to the list
					temp_neighbors.push_back(triangle[0]);
					temp_neighbors.push_back(triangle[1]);
					temp_neighbors.push_back(triangle[2]);
				}
			}
		}
		//remove duplicates from the list
		std::sort(temp_neighbors.begin(), temp_neighbors.end());
		temp_neighbors.erase(std::unique(temp_neighbors.begin(), temp_neighbors.end()), temp_neighbors.end());
	}

	//-----------------------------[ get GFA ]------------------------------\\

	void MaximumFinder::getGFA(double * GFA)
	{


		//get number of tesselation points
		int n = (this->radii).size();
		//set output to zero
		(*GFA) = 0.0;

		//calculate average radius
		double average = 0.0;
		for (int i = 0; i < n; ++i)
		{
			average += (this->radii)[i];
		}
		average = average/n;

		//calculate the numerator
		double numerator = 0.0;
		for (int i = 0; i < n; ++i)
		{
			numerator += pow(((this->radii)[i]-average),2);
		}
		numerator = numerator*n;

		//calculate the denominator
		double denominator = 0.0;
		for (int i = 0; i < n; ++i)
		{
			denominator += pow((this->radii)[i],2);
		}
		denominator = denominator*(n-1);

		//calculate the GFA
		(*GFA) = sqrt(numerator/denominator);

	}

	//-----------------------------[ Clean the list of initial local maxima. ]------------------------------\\

	void MaximumFinder::cleanOutput(std::vector<int> &indexOfMax, std::vector<double *>& outputlistwithunitvectors, double* pDarraySH, std::vector<double> &ODFlist, double** unitVectors, std::vector<double*> &anglesArray )
	{
		//list with single maxima
		std::vector<int> goodlist;

		//list with double maxima (2 neighboring points)
		std::vector<int> doubtlist;

		//list with registered maxima
		std::vector<int> donelist;

		//temporary output vector
		std::vector<double *> outputlistwithunitvectors1;

		//list with radii of calculated points
		std::vector<double> radius;
		//temporary angle list
		std::vector<double*> tempAngleArray;
		//temporary ODF list
		std::vector<double> tempODFlist;

		//lists with neighbors
		std::vector<int> neighborhood1;
		std::vector<int> neighborhood2;

		//for every maximum, count the number of neighboring maxima with depth 1
		//and add maximum to correct list
		for (unsigned int i = 0; i < indexOfMax.size(); ++i)
		{	
			//get neighbors
			this->getNeighbors(indexOfMax[i],2,neighborhood1); //2
			int neighborcounter = 0;
			for (unsigned int j = 0; j < neighborhood1.size(); ++j)
			{
				//count neighbors
				if (std::find(indexOfMax.begin(), indexOfMax.end(), neighborhood1[j]) != indexOfMax.end()) // not same indexed increase
				{
					neighborcounter += 1;	
				}
			}
			//single- and multi-point maxima. why this are on good list?
			if ((neighborcounter == 0) || (neighborcounter > 2))
			{
				goodlist.push_back(indexOfMax[i]);
			}
			//double and tiple maxima
			if ((neighborcounter == 1) || (neighborcounter == 2))
			{
				doubtlist.push_back(indexOfMax[i]);
			}
		}

		//for all double and tirple maxima
		for (unsigned int i = 0; i < doubtlist.size(); ++i)
		{
			//check for triangle
			for (int j=0;j<(this->trianglesArray->GetNumberOfTuples());j++)
			{
				vtkIdType index = j;
				double * henk=this->trianglesArray->GetTuple(index);
				//typecasting
				int index0 = henk[0];
				int index1 = henk[1];
				int index2 = henk[2];

				//check for triple maximum
				// 3 MAX
				if ((index0==doubtlist[i]) || (index1==doubtlist[i]) || (index2==doubtlist[i]))
				{
					if ( // all 3 are in the doubtlist
						(std::find(doubtlist.begin(), doubtlist.end(), index0) != doubtlist.end()) &&
						(std::find(doubtlist.begin(), doubtlist.end(), index1) != doubtlist.end())	&&
						(std::find(doubtlist.begin(), doubtlist.end(), index2) != doubtlist.end())
						)
					{
						//get angles of original and calculated average directions
						double * angles = new double[2];
						angles[0] = (1.0/3.0)*((anglesArray[index0][0])+ (anglesArray[index1][0])+ (anglesArray[index2][0]));
						angles[1] = (1.0/3.0)*(anglesArray[index0][1])+ (anglesArray[index1][2])+ (anglesArray[index2][3]);

						double * originals1 = new double[2];
						originals1[0] = (anglesArray[index0][0]);
						originals1[1] = (anglesArray[index0][1]);

						double * originals2 = new double[2];
						originals2[0] = (anglesArray[index1][0]);
						originals2[1] = (anglesArray[index1][1]);

						double * originals3 = new double[2];
						originals2[0] = (anglesArray[index2][0]);
						originals2[1] = (anglesArray[index2][1]);

						tempAngleArray.clear();
						tempAngleArray.push_back(angles);
						tempAngleArray.push_back(originals1);
						tempAngleArray.push_back(originals2);
						tempAngleArray.push_back(originals3);

						//get the corresponding radii
						radius.clear();
						radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);

						//delete pointers
						delete angles;
						delete originals1;
						delete originals2;
						delete originals3;

						//if the averagedirection has a higher ODF value than the original(corners of triangle) directions take the average value!!!
						if ((fabs(radius[1]) < fabs(radius[0])) && (fabs(radius[2]) < fabs(radius[0])) && (fabs(radius[3]) < fabs(radius[0])))
						{
							double * tempout = new double[3];
							//calculate the average direction
							tempout[0] = (1.0/3.0)*(unitVectors[(index0)][0])+(1.0/3.0)*(unitVectors[(index1)][0])+(1.0/3.0)*(unitVectors[(index2)][0]);
							tempout[1] = (1.0/3.0)*(unitVectors[(index0)][1])+(1.0/3.0)*(unitVectors[(index1)][1])+(1.0/3.0)*(unitVectors[(index2)][0]);
							tempout[2] = (1.0/3.0)*(unitVectors[(index0)][2])+(1.0/3.0)*(unitVectors[(index1)][2])+(1.0/3.0)*(unitVectors[(index2)][0]);
							//add the calculated direction to the output
							outputlistwithunitvectors1.push_back(tempout);
							//add the ODF value to the output
							tempODFlist.push_back(fabs(radius[0]));
						}
						else
						{
							//add the original direction(ie the corner) and ODF value to the output list
							outputlistwithunitvectors1.push_back(unitVectors[doubtlist[i]]);
							if (doubtlist[i] == index0)
							{
								tempODFlist.push_back(fabs(radius[1]));
							}
							if (doubtlist[i] == index1)
							{
								tempODFlist.push_back(fabs(radius[2]));
							}
							if (doubtlist[i] == index2)
							{
								tempODFlist.push_back(fabs(radius[3]));
							}
						}
						donelist.push_back(i); // donelist is re-registered triple maxima from doubtlist, average or coners list
					}// all in a triangle and goublist
				}// if one triple max
			} // triangles array and still in doublt iteration
			//for all points that are not a triple maximum, ie  2 maxes!!!!, which are actually the remaining from double and triple
			// 2 MAX below
			if (!(std::find(donelist.begin(), donelist.end(), i) != donelist.end()))  // i index of doubtlist, if it is not in triple max
			{
				//check for double point
				this->getNeighbors(doubtlist[i],1,neighborhood1); // 1
				for (unsigned int j = 0; j < neighborhood1.size(); ++j)
				{  //doubtlist include double and triple maxima
					if (std::find(doubtlist.begin(), doubtlist.end(), neighborhood1[j]) != doubtlist.end()) // tabi doublist neigh da doublist icinde olabilir yine ayikla
					{
						//get angles of original -ie 2 points in the exact array- and calculated average directions
						double * angles = new double[2];
						angles[0] = 0.5*(anglesArray[(neighborhood1[j])][0])+0.5*(anglesArray[(doubtlist[i])][0]);
						angles[1] = 0.5*(anglesArray[(neighborhood1[j])][1])+0.5*(anglesArray[(doubtlist[i])][1]);

						double * originals1 = new double[2];
						originals1[0] = (anglesArray[(doubtlist[i])][0]);
						originals1[1] = (anglesArray[(doubtlist[i])][1]);

						double * originals2 = new double[2];
						originals2[0] = (anglesArray[(neighborhood1[j])][0]);
						originals2[1] = (anglesArray[(neighborhood1[j])][1]);

						tempAngleArray.clear();
						tempAngleArray.push_back(angles);
						tempAngleArray.push_back(originals1);
						tempAngleArray.push_back(originals2);

						//get the corresponding radii
						radius.clear();
						radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);

						//delete pointers
						delete angles;
						delete originals1;
						delete originals2;

						//if the averagedirection of a double max has a higher ODF value than the original directions
						if ((fabs(radius[0]) > fabs(radius[1])) && (fabs(radius[0]) > fabs(radius[2])))
						{
							double * tempout = new double[3];
							//calculate the average direction
							tempout[0] = 0.5*(unitVectors[(neighborhood1[j])][0])+0.5*(unitVectors[(doubtlist[i])][0]);
							tempout[1] = 0.5*(unitVectors[(neighborhood1[j])][1])+0.5*(unitVectors[(doubtlist[i])][1]);
							tempout[2] = 0.5*(unitVectors[(neighborhood1[j])][2])+0.5*(unitVectors[(doubtlist[i])][2]);

							//add the calculated direction to the output
							outputlistwithunitvectors1.push_back(tempout);
							//add the ODF value to the output
							tempODFlist.push_back(fabs(radius[0]));
						}
						else
						{
							//add the original direction and ODF value to the output
							outputlistwithunitvectors1.push_back(unitVectors[doubtlist[i]]);
							tempODFlist.push_back(fabs(radius[1])); //add to the original
						}
						radius.clear();
					}
					else
					{
						//for all remaining points
						this->getNeighbors(doubtlist[i],1,neighborhood2);
						int neighborcounter = 0;
						//count the neighbors
						for (unsigned int j = 0; j < neighborhood2.size(); ++j)
						{
							if (std::find(indexOfMax.begin(), indexOfMax.end(), neighborhood2[j]) != indexOfMax.end())
							{
								neighborcounter += 1;	
							}
						}
						//if there are more than 1 neighbor 
						if (neighborcounter != 1)
						{
							//add unit vector to the output list
							outputlistwithunitvectors1.push_back(unitVectors[doubtlist[i]]);
							//get angles
							double * originals1 = new double[2];
							originals1[0] = (anglesArray[(doubtlist[i])][0]);
							originals1[1] = (anglesArray[(doubtlist[i])][1]);
							tempAngleArray.clear();
							tempAngleArray.push_back(originals1);
							//get radius
							radius.clear();
							radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);
							delete originals1;
							//add radius to output list
							tempODFlist.push_back(fabs(radius[0]));
						}
					}// if
				}//for neighbourhodd of double maxes
			} // if not triple max
		}

		//add the single point maxima, goodlist includes single point maximas
		for (unsigned int i = 0; i < goodlist.size(); ++i)
		{
			//add unit vector to the output list
			outputlistwithunitvectors1.push_back(unitVectors[goodlist[i]]);

			//get angles
			double * originals1 = new double[2];
			originals1[0] = (anglesArray[(goodlist[i])][0]);
			originals1[1] = (anglesArray[(goodlist[i])][1]);
			tempAngleArray.clear();
			tempAngleArray.push_back(originals1);
			//get radius
			radius.clear();
			radius = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &tempAngleArray, 4);
			delete originals1;
			//add radius to output list
			tempODFlist.push_back(radius[0]);
		}

		//delete any duplicates
		if(outputlistwithunitvectors1.size() != 0)
			outputlistwithunitvectors.push_back(outputlistwithunitvectors1.at(0));
		if(ODFlist.size()!=0)
			ODFlist.push_back(tempODFlist.at(0));
		bool duplicate;
		//for every unit vector, ceck whether it is already in the final output list
		for (unsigned int i = 1; i < (outputlistwithunitvectors1).size(); ++i)
		{
			duplicate = false;
			for (unsigned int j = 0; j < outputlistwithunitvectors.size(); ++j)
			{
				if ((outputlistwithunitvectors1[i][0] == outputlistwithunitvectors[j][0]) && (outputlistwithunitvectors1[i][1] == outputlistwithunitvectors[j][1]) && (outputlistwithunitvectors1[i][2] == outputlistwithunitvectors[j][2]))
				{
					duplicate = true;
				}
			}
			//if it is not yet it the list
			if (!duplicate)
			{
				//add unit vector and ODF value to the list
				outputlistwithunitvectors.push_back(outputlistwithunitvectors1[i]);
				ODFlist.push_back(tempODFlist[i]);
			}
		} // ODFlist sort then take the correspoding i values!!
	}

		////--------------------------[ Find maxima for Spherical Harmonics Data. More then 1 maxima are found.]--------------------------\\

	void MaximumFinder::getOutput(double* pDarraySH, int shOrder,double treshold, std::vector<double*> anglesArray,  std::vector<int>& indexesOfMaxima, std::vector<int> &indexes)
	{
		//input is indexes
		//clear the indexesOfMaxima
		indexesOfMaxima.clear();
		//get radii
		this->radii = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &anglesArray, shOrder);  //harmonics and angles as input

		//list with neighborhood
		std::vector<int> neighborslist;

		//find maximum and minimum radii
		double min = *min_element((this->radii).begin(), (this->radii).end());
		double max = *max_element((this->radii).begin(), (this->radii).end());

		//min-max normalization
		//for all points on the sphere
		(this->radii_norm).clear();
		for (unsigned int i = 0; i < (this->radii).size(); ++i)
		{
			//dont divide by 0
			if (min != max)
			{
				//add normalized radius
				this->radii_norm.push_back(((this->radii[i])-min)/(max-min)); //normalize
			}
			//in case of sphere (SH-order = 0)
			else
			{	//add 1.0 (sphere)    
				this->radii_norm.push_back(1.0);
			}
		}

		//for all points on the sphere
		for (unsigned int i = 0; i < (indexes.size()); ++i)
		{
			//get current radius
			double currentPointValue = this->radii_norm[indexes[i]];

			//if the value is high enough
			if (currentPointValue > (treshold))
			{
				//get the neighbors 
				getNeighbors(indexes[i], 1, neighborslist);
				double currentMax = 0.0;

				//find the highest valued neighbor
				for (unsigned int j = 0; j < neighborslist.size(); ++j)
				{
					if ((this->radii_norm)[(neighborslist[j])]>currentMax)
					{
						currentMax = this->radii_norm[(neighborslist[j])];
					}	
				}
				//if the current point value is higher than the highest valued neighbor
				if (currentMax <= currentPointValue)
				{
					//add point to the indexesOfMaxima
					indexesOfMaxima.push_back(indexes[i]); // choose it if it has a radius larger then all its neighbours
				}	
			}
		}	
	}


	////--------------------------[ Find maxima for Spherical Harmonics Data. More then 1 maxima are found.]--------------------------\\

	void MaximumFinder::getUniqueOutput(double* pDarraySH, int shOrder,double treshold, std::vector<double*> anglesArray,   std::vector<int> &input,int &indexOfMax)
	{
		//input is indexes
		//clear the indexesOfMaxima
	 
		//get radii
		this->radii = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &anglesArray, shOrder);  //harmonics and angles as input
		  std::vector<double>::iterator result;
 
	 

		//find maximum and minimum radii
	 
		 result = std::max_element((this->radii).begin(), (this->radii).end());
		 indexOfMax = std::distance((this->radii).begin(), result);
		 	
		
	}



	//-----------------------------[ Constructor ]-----------------------------\\

	MaximumFinder::MaximumFinder(vtkIntArray* trianglesArray)
	{

		this->trianglesArray	= trianglesArray;
		this->radii_norm.clear();
		this->radii.clear();
	}

	//-----------------------------[ Destructor ]------------------------------\\

	MaximumFinder::~MaximumFinder()
	{

		this->trianglesArray	= NULL;
		this->radii_norm.clear();
		this->radii.clear();
	}
	//--------------------------[ Find maxima ]--------------------------\\

	void MaximumFinder::getOutput(double* pDarraySH, int shOrder, std::vector<double*> anglesArray)
	{

		//get radii
		this->radii = bmia::HARDITransformationManager::CalculateDeformator(pDarraySH, &anglesArray, shOrder);


		//list with neighborhood
		std::vector<int> neighborslist;

		//find maximum and minimum radii
		double min = *min_element((this->radii).begin(), (this->radii).end());
		double max = *max_element((this->radii).begin(), (this->radii).end());

		//min-max normalization
		//for all points on the sphere
		(this->radii_norm).clear();
		for (unsigned int i = 0; i < (this->radii).size(); ++i)
		{
			//dont divide by 0
			if (min != max)
			{
				//add normalized radius
				this->radii_norm.push_back(((this->radii[i])-min)/(max-min));
			}
			//in case of sphere (SH-order = 0)
			else
			{	//add 1.0 (sphere)    
				this->radii_norm.push_back(1.0); 
			}
		}
	}

	}//namespace