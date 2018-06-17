//std includes
#include <iostream>
#include <utility>
//boost includes
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
//hype includes
#include <query_processing/benchmark.hpp>
//CoGaDB includes
#include <core/table.hpp>
#include <query_processing/query_processor.hpp>

using namespace std;
using namespace CoGaDB;
using namespace query_processing;

	class Selection_Benchmark : public hype::queryprocessing::Operation_Benchmark<TablePtr>{
		public:
		Selection_Benchmark() : hype::queryprocessing::Operation_Benchmark<TablePtr>("SELECTION","CPU_Selection_Algorithm","GPU_Selection_Algorithm"){}

		virtual TypedNodePtr generate_logical_operator(TablePtr dataset){

			//cout << "Create Sort Operation for Table " << dataset->getName() << endl;

			int selection_value;
			ValueComparator selection_comparison_value; //0 EQUAL, 1 LESSER, 2 LARGER

			boost::mt19937& rng = this->getRandomNumberGenerator();
			boost::uniform_int<> selection_values(0,1000);
			boost::uniform_int<> filter_condition(0,2);

			selection_value = selection_values(rng);
			selection_comparison_value = (ValueComparator)filter_condition(rng); //rand()%3;

			boost::shared_ptr<logical_operator::Logical_Scan>  scan(new logical_operator::Logical_Scan(dataset->getName()));

			boost::shared_ptr<logical_operator::Logical_Selection>  selection(new logical_operator::Logical_Selection("values",selection_value,selection_comparison_value,LOOKUP));	

			selection->setLeft(scan);

			return selection;
		}


		 //virtual vector<TypedNodePtr> createOperatorQueries() = 0;


		virtual TablePtr generate_dataset(unsigned int size_in_number_of_bytes){
			static unsigned int dataset_counter=0;

			cout << "Create Dataset of Size " <<  size_in_number_of_bytes << " Byte" << endl;

			unsigned int size_in_number_of_elements = size_in_number_of_bytes/sizeof(int);

			TableSchema schema;
			schema.push_back(Attribut(INT,"values"));

			std::string table_name="Table_";
			table_name+=boost::lexical_cast<std::string>(dataset_counter++);
			TablePtr tab1(new Table(table_name,schema));

			boost::mt19937& rng = this->getRandomNumberGenerator();
			boost::uniform_int<> six(0,1000);

			for(unsigned int i=0;i<size_in_number_of_elements;++i){
				int e =  six(rng); //rand();
				//int grouping_value=i/1000; //each group has 1000 keys that needs to be aggregated
				{Tuple t; t.push_back(e); tab1->insert(t);}
			}

			//tab1->print();
			getGlobalTableList().push_back(tab1);


			return tab1;
		}
	};





int main(int argc, char* argv[]){
	Selection_Benchmark s;

	s.setup(argc, argv);

	s.run();

 return 0;
}

