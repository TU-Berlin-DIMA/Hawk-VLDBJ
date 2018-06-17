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

        struct PrimaryForeignTablePair{
            PrimaryForeignTablePair(TablePtr prim_key_table, TablePtr for_key_table) : primary_key_table(prim_key_table),foreign_key_table(for_key_table){}
            unsigned int getSizeinBytes() const{ return primary_key_table->getSizeinBytes()+foreign_key_table->getSizeinBytes();}
            unsigned int getNumberofRows() const{ return primary_key_table->getNumberofRows()+foreign_key_table->getNumberofRows();}
            TablePtr primary_key_table;
            TablePtr foreign_key_table;
        };

        //typedef std::pair<TablePtr,TablePtr> PrimaryForeignTablePair;
        typedef boost::shared_ptr<PrimaryForeignTablePair> PrimaryForeignTablePairPtr;

	class Join_Benchmark : public hype::queryprocessing::Operation_Benchmark<TablePtr,PrimaryForeignTablePairPtr>{
		public:
		Join_Benchmark() : hype::queryprocessing::Operation_Benchmark<TablePtr,PrimaryForeignTablePairPtr>("JOIN","CPU_HashJoin_Algorithm","GPU_SortMergeJoin_Algorithm"){this->stemod_statistical_method_="Least Squares 2D"; }

		virtual TypedNodePtr generate_logical_operator(PrimaryForeignTablePairPtr dataset){

			//cout << "Create Sort Operation for Table " << dataset->getName() << endl;

                        boost::shared_ptr<logical_operator::Logical_Scan>  scan_primary_key_table(new logical_operator::Logical_Scan(dataset->primary_key_table));
                        boost::shared_ptr<logical_operator::Logical_Scan>  scan_foreign_key_table(new logical_operator::Logical_Scan(dataset->foreign_key_table));

                        boost::shared_ptr<logical_operator::Logical_Join>  join(new logical_operator::Logical_Join("PK_ID","FK_ID"));

                        join->setLeft(scan_primary_key_table);
                        join->setRight(scan_foreign_key_table);
			LogicalQueryPlan log_plan(join);
                        log_plan.print();
			return join;
		}


		 //virtual vector<TypedNodePtr> createOperatorQueries() = 0;


		virtual PrimaryForeignTablePairPtr generate_dataset(unsigned int size_in_number_of_bytes){
			static unsigned int dataset_counter=0;

			cout << "Create Dataset of Size " <<  size_in_number_of_bytes << " Byte" << endl;

			unsigned int size_in_number_of_elements = size_in_number_of_bytes/sizeof(int);

                        unsigned int number_of_primary_keys=size_in_number_of_elements*0.1;
                        unsigned int number_of_foreign_keys=size_in_number_of_elements*0.9;
                        
			TableSchema schema;
			schema.push_back(Attribut(INT,"PK_ID"));

			std::string table_name="Table_Primary_";
			table_name+=boost::lexical_cast<std::string>(dataset_counter);
			TablePtr table_primary(new Table(table_name,schema));
                        
			TableSchema schema_foreign;
			schema_foreign.push_back(Attribut(INT,"FK_ID"));

			std::string table_foreign_name="Table_Foreign_";
			table_foreign_name+=boost::lexical_cast<std::string>(dataset_counter++);
			TablePtr table_foreign(new Table(table_foreign_name,schema_foreign));                      

			boost::mt19937& rng = this->getRandomNumberGenerator();
                        //generate foreign keys that are uniformly distributed and index the primary key column
			boost::uniform_int<> foreign_key_uniform_distribution(0,number_of_primary_keys-1);
                        
                        //fill primary key table
                        int primary_key=0;
			for(unsigned int i=0;i<number_of_primary_keys;++i){
				{Tuple t; t.push_back(primary_key++); table_primary->insert(t);}
			}
                        
                        //fill foreign key table
			for(unsigned int i=0;i<number_of_foreign_keys;++i){
				int e =  foreign_key_uniform_distribution(rng);
				{Tuple t; t.push_back(e); table_foreign->insert(t);}
			}

			//tab1->print();
			getGlobalTableList().push_back(table_primary);
			getGlobalTableList().push_back(table_foreign);

			cout << table_primary->getName() << endl;
			cout << table_foreign->getName() << endl;
                        
			return PrimaryForeignTablePairPtr(new PrimaryForeignTablePair(table_primary,table_foreign));
		}
	};





int main(int argc, char* argv[]){
	Join_Benchmark s;

	s.setup(argc, argv);

	s.run();

 return 0;
}

