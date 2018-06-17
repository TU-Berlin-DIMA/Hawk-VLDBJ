/*
 * File:   create_table_operator.hpp
 * Author: sebastian
 *
 * Created on 22. Dezember 2015, 18:01
 */

#ifndef CREATE_TABLE_OPERATOR_HPP
#define CREATE_TABLE_OPERATOR_HPP

#include <query_processing/definitions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {

    Physical_Operator_Map_Ptr map_init_function_create_table_operator();

    namespace logical_operator {
      class Logical_CreateTable
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                map_init_function_create_table_operator>  // init_function_Join_operator>
      {
       public:
        Logical_CreateTable(
            const std::string& table_name, const TableSchema& schema,
            const CompressionSpecifications& compression_specifications,
            const std::vector<Tuple>& tuples_to_insert);

        Logical_CreateTable(
            const std::string& table_name, const TableSchema& schema,
            const CompressionSpecifications& compression_specifications,
            const std::string& path_to_file, const std::string& delimiter);

        Logical_CreateTable(
            const std::string& table_name, const TableSchema& schema,
            const CompressionSpecifications& compression_specifications,
            const std::vector<Tuple>& tuples_to_insert,
            const std::string& path_to_file, const std::string& delimiter);

        //                : TypedNode_Impl<TablePtr,
        //                map_init_function_create_table_operator>(false,
        //                hype::ANY_DEVICE),
        //                table_name_(table_name), schema_(schema),
        //                compression_specifications_(compression_specifications){
        //                }

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual const hype::Tuple getFeatureVector() const;

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

       private:
        std::string table_name_;
        TableSchema schema_;
        CompressionSpecifications compression_specifications_;
        std::vector<Tuple> tuples_to_insert_;
        std::string path_to_file_;
        std::string delimiter_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB

#endif /* CREATE_TABLE_OPERATOR_HPP */
