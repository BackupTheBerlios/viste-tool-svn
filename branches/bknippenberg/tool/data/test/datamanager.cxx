/*
 * datamanager.cxx
 *
 * 2009-11-04	Tim Peeters
 * - First version
 */

#include "data/Manager.h"
#include "data/Consumer.h"
#include "data/DataSet.h"

#include <QTextStream>

using namespace bmia;
using namespace data;

QTextStream out(stdout);

class HungryPerson: public Consumer
{
public:
    HungryPerson() {};
    ~HungryPerson() {};
    void dataSetAdded(DataSet* ds)
	{
	out<<"Eat "<<ds->getName()<<", nom nom!"<<endl;
        };
    void dataSetChanged(DataSet* ds)
        {
        out<<"What happend to my "<<ds->getName()<<"???"<<endl;
        };
    void dataSetRemoved(DataSet* ds)
        {
        out<<"ahhh"<<endl;
        };
}; // class HungryPerson

/**
 * Test data manager.
 */
int main(int argc, char ** argv)
{
  Manager* McDonalds= new Manager();
  HungryPerson* Piet = new HungryPerson();
  HungryPerson* Frits = new HungryPerson();

  DataSet* hamburger = new DataSet("Big Mac", "fat food");
  DataSet* fries = new DataSet("French Fries", "fat food");
  DataSet* dessert = new DataSet("McFlurry", "sweet food");

  McDonalds->addConsumer(Piet);
  McDonalds->addDataSet(hamburger);
  McDonalds->addDataSet(hamburger); 	// another one?! Piet must be so hungry!
  McDonalds->printAllDataSets(); 		// McDonalds did not allow it. Only one hamburger at a time!

  McDonalds->addConsumer(Frits);
  McDonalds->addDataSet(fries);		// Piet and Frits share the fries
  McDonalds->removeConsumer(Frits);	// bye bye!

  McDonalds->addDataSet(dessert);
  McDonalds->dataSetChanged(dessert);
  McDonalds->printAllDataSets();

  McDonalds->removeDataSet(hamburger);	// now visit the toilet ;)
  McDonalds->removeDataSet(hamburger);	// fails. hamburger was already removed (no output).
  McDonalds->removeDataSet(fries);
  McDonalds->removeConsumer(Piet);

  return (McDonalds->listDataSets("sweet food").size() != 1); // pass test if there is 1 sweet food in the list.
}
