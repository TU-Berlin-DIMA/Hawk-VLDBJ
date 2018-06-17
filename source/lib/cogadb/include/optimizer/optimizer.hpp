
#include <query_processing/query_processor.hpp>

namespace CoGaDB {

  namespace optimizer {
    enum { verbose_optimizer = 0 };

    class Logical_Optimizer {
     public:
      static Logical_Optimizer& instance();
      bool optimize(query_processing::LogicalQueryPlanPtr);

     private:
      typedef boost::function<bool(query_processing::LogicalQueryPlanPtr)>
          OptimizerRule;
      typedef std::vector<OptimizerRule> OptimizerRules;
      struct OptimizerPipeline {
        OptimizerPipeline(std::string optimizer_name,
                          OptimizerRules optimizer_rules);
        std::string optimizer_name;
        OptimizerRules optimizer_rules;
      };
      typedef boost::shared_ptr<OptimizerPipeline> OptimizerPipelinePtr;
      typedef std::map<std::string, OptimizerPipelinePtr> OptimizerPipelines;
      Logical_Optimizer();
      Logical_Optimizer(const Logical_Optimizer&);
      Logical_Optimizer& operator()(const Logical_Optimizer&);
      // OptimizerRules optimizer_rules_;
      OptimizerPipelines optimizers_;
    };

    namespace optimizer_rules {
      bool decompose_complex_selections(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool push_down_selections(query_processing::LogicalQueryPlanPtr log_plan);
      bool compose_complex_selections(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool cross_product_to_join(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool add_artificial_pipeline_breakers(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool move_fact_table_scan_to_right_side_of_join(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool remove_cross_joins_and_keep_join_order(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool join_order_optimization(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool rewrite_join_to_pk_fk_join(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool rewrite_join_to_gather_join(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool rewrite_join_to_fetch_join(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool rewrite_join_tree_to_invisible_join(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool rewrite_join_tree_to_chain_join(
          query_processing::LogicalQueryPlanPtr log_plan);
      bool set_device_constaints_for_unsupported_operations(
          query_processing::LogicalQueryPlanPtr log_plan);

    } /* namespace optimizer_rules */

    bool checkQueryPlan(query_processing::NodePtr node);

    std::list<Attribut> getListOfAvailableAttributesChildrenAndSelf(
        query_processing::NodePtr node);
    std::list<Attribut> getListOfAvailableAttributesChildrenOnly(
        query_processing::NodePtr node);

    //        Logical_Optimizer::Logical_Optimizer() : optimizer_rules_(){
    //
    //                optimizer_rules_.push_back(
    //                OptimizerRule(optimizer_rules::push_down_selections) );
    //        }
    //
    //        Logical_Optimizer& Logical_Optimizer::instance(){
    //            static Logical_Optimizer optimizer;
    //            return optimizer;
    //        }
    //
    //        bool
    //        Logical_Optimizer::optimize(query_processing::LogicalQueryPlanPtr
    //        log_plan){
    //            OptimizerRules::iterator it;
    //            cout << "Input Plan:" < endl;
    //            log_plan->print();
    //            for(it=optimizer_rules_.begin();it!=optimizer_rules_.end();++it){
    //                if(!it->empty()){
    //                    ;
    //                    if(!(*it)(log_plan)){
    //                        cout << "Logical Optimization Failed!" << endl;
    //                        return false;
    //                    }
    //                }
    //                cout << "Optimized Plan:" < endl;
    //                log_plan->print();
    //
    //            }
    //            return true;
    //
    //        }
    //
    //
    //
    //
    //        namespace optimizer_rules{
    //            bool
    //            push_down_selections(query_processing::LogicalQueryPlanPtr
    //            log_plan){
    //
    //                struct Push_Down_Selection_Functor{
    //
    //                    bool operator()(query_processing::LogicalQueryPlanPtr
    //                    log_plan){
    //
    //
    //                    }
    //
    //                };
    //
    //                log_plan->traverse();
    //                return true;
    //            }
    //
    //            bool
    //            cross_product_to_join(query_processing::LogicalQueryPlanPtr
    //            log_plan){
    //                return false;
    //            }
    //
    //            bool
    //            join_order_optimization(query_processing::LogicalQueryPlanPtr
    //            log_plan){
    //                return false;
    //            }
    //        } /* namespace optimizer_rules */
    //

  } /* namespace optimizer */
} /* namespace CoGaDB */
