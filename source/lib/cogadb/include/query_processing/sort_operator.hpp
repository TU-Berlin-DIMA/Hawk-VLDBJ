#pragma once

#include <core/runtime_configuration.hpp>
#include <query_processing/definitions.hpp>

namespace CoGaDB {

  namespace query_processing {

    namespace physical_operator {

      class CPU_Sort_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_Sort_Operator(const hype::SchedulingDecision& sched_dec,
                          TypedOperatorPtr child,
                          const std::list<SortAttribute>& sort_attributes,
                          MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              sort_attributes_(sort_attributes),
              mat_stat_(mat_stat) {}

        virtual bool execute() {
          // std::cout << "Execute Sort CPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          this->result_ = BaseTable::sort(this->getInputData(),
                                          sort_attributes_, mat_stat_, CPU);

          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_Sort_Operator() {}

       private:
        std::list<SortAttribute> sort_attributes_;
        MaterializationStatus mat_stat_;
      };

      class GPU_Sort_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        GPU_Sort_Operator(const hype::SchedulingDecision& sched_dec,
                          TypedOperatorPtr child,
                          const std::list<SortAttribute>& sort_attributes,
                          MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              sort_attributes_(sort_attributes),
              mat_stat_(mat_stat) {}

        virtual bool execute() {
          this->result_ = BaseTable::sort(this->getInputData(),
                                          sort_attributes_, mat_stat_, GPU);

          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~GPU_Sort_Operator() {}

       private:
        std::list<SortAttribute> sort_attributes_;
        MaterializationStatus mat_stat_;
      };

      Physical_Operator_Map_Ptr map_init_function_sort_operator();
      TypedOperatorPtr create_CPU_SORT_Operator(TypedLogicalNode& logical_node,
                                                const hype::SchedulingDecision&,
                                                TypedOperatorPtr left_child,
                                                TypedOperatorPtr right_child);
      TypedOperatorPtr create_GPU_SORT_Operator(TypedLogicalNode& logical_node,
                                                const hype::SchedulingDecision&,
                                                TypedOperatorPtr left_child,
                                                TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_Sort
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_sort_operator>  // init_function_sort_operator>
      {
       public:
        Logical_Sort(const SortAttributeList& sort_attributes,
                     MaterializationStatus mat_stat = LOOKUP,
                     hype::DeviceConstraint dev_constr =
                         CoGaDB::RuntimeConfiguration::instance()
                             .getGlobalDeviceConstraint());

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const std::list<SortAttribute>& getSortAttributes();

        MaterializationStatus getMaterializationStatus();

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        const hype::Tuple getFeatureVector() const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

        std::list<SortAttribute> sort_attributes_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
