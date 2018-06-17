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


//#include <util/time_measurement.hpp>
//#include <persistence/storage_manager.hpp>

//Unittests

//#include <gpu/gpu_algorithms.hpp>

//#include <lookup_table/lookup_table.hpp>

using namespace std;
using namespace CoGaDB;
using namespace query_processing;

	class Aggregation_Benchmark : public hype::queryprocessing::Operation_Benchmark<TablePtr>{
		public:
		Aggregation_Benchmark() : hype::queryprocessing::Operation_Benchmark<TablePtr>("GROUPBY","CPU_Groupby_Algorithm","GPU_Groupby_Algorithm"){}

		virtual TypedNodePtr generate_logical_operator(TablePtr dataset){

			//cout << "Create Sort Operation for Table " << dataset->getName() << endl;

			boost::shared_ptr<logical_operator::Logical_Scan>  scan(new logical_operator::Logical_Scan(dataset->getName()));

                        std::list<std::string> sorting_columns;
                        sorting_columns.push_back("Shop_ID"); 
                   
                        std::list<std::pair<string,Aggregate> > aggregation_functions;
                        aggregation_functions.push_back(make_pair("Sales",Aggregate(SUM,"Sales")));

			boost::shared_ptr<logical_operator::Logical_Groupby>  groupby(new logical_operator::Logical_Groupby(sorting_columns,aggregation_functions,LOOKUP));	

			groupby->setLeft(scan);

			return groupby;
		}

		virtual TablePtr generate_dataset(unsigned int size_in_number_of_bytes){
			static unsigned int dataset_counter=0;

			cout << "Create Dataset of Size " <<  size_in_number_of_bytes << " Byte" << endl;

			unsigned int size_in_number_of_elements = size_in_number_of_bytes/(sizeof(int)*2); //devide by 2, because we have two columns of integer values!

			TableSchema schema;
			schema.push_back(Attribut(INT,"Shop_ID"));
			schema.push_back(Attribut(INT,"Sales"));

			std::string table_name="Table_";
			table_name+=boost::lexical_cast<std::string>(dataset_counter++);
			TablePtr tab1(new Table(table_name,schema));

			boost::mt19937& rng = this->getRandomNumberGenerator();
			boost::uniform_int<> six(0,1000);

			for(unsigned int i=0;i<size_in_number_of_elements;++i){
				int e =  six(rng); //rand();
				int grouping_value=i/1000; //each group has 1000 keys that needs to be aggregated
				{Tuple t; t.push_back(grouping_value); t.push_back(e); tab1->insert(t);}
			}

			//tab1->print();
			getGlobalTableList().push_back(tab1);


			return tab1;
		}
	};




int main(int argc, char* argv[]){

	Aggregation_Benchmark s;

	s.setup(argc, argv);

	s.run();

 return 0;
}

