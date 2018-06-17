#pragma once

//#include <core/scheduling_decision.hpp>

//#include <boost/shared_ptr.hpp>
#include <config/configuration.hpp>
#include <config/global_definitions.hpp>
#include <query_processing/typed_operator.hpp>

namespace hype {
  namespace queryprocessing {

    template <typename OperatorInputType, typename OperatorOutputType>
    class N_AryOperator
        : public TypedOperator<OperatorOutputType> {  // uses 2D Learning Method
     public:
      // typedef typename
      // OperatorMapper_Helper_Template<OperatorOutputType>::TypedOperator
      // TypedOperator;
      typedef typename OperatorMapper_Helper_Template<
          OperatorOutputType>::TypedOperatorPtr TypedOperatorPtr;
      typedef typename OperatorMapper_Helper_Template<
          OperatorOutputType>::TypedNodePtr TypedNodePtr;

      N_AryOperator(const hype::SchedulingDecision& sched_dec,
                    const std::list<OperatorInputType>& childs)
          : TypedOperator<OperatorOutputType>(sched_dec), childs_(childs) {}

      virtual ~N_AryOperator() {}

      //			OperatorInputTypeLeftChild
      // getInputDataLeftChild(){
      //				return this->left_child_->getResult();
      //			}

      //			OperatorInputTypeRightChild
      // getInputDataRightChild(){
      //				return this->right_child_->getResult();
      //			}

      const std::list<OperatorInputType>& getOutputDataOfChilds() {
        return childs_;
      }

      virtual bool run() {
        std::list<OperatorInputType>::iterator it;
        if (!hype::core::Runtime_Configuration::instance()
                 .isQueryChoppingEnabled()) {
          for (it = childs_.begin(); it != childs_.end(); ++it) {
            if (*it) it->run();
          }
        }
        // execute this operator
        double begin = double(hype::core::getTimestamp());
        timeEstimated =
            double(this->getEstimatedExecutionTime().getTimeinNanoseconds());
        bool retVal = (*this)();
        timeNeeded = double(hype::core::getTimestamp()) - begin;
        return retVal;
      }

      virtual void print(unsigned int tree_level) const {
        for (unsigned int i = 0; i < tree_level; ++i) {
          std::cout << "\t";
        }
        assert(!this->getFeatureValues().empty());
        std::cout << this->getAlgorithmName() << " ET: "
                  << this->getEstimatedExecutionTime().getTimeinNanoseconds()
                  << "\tEC: " << this->getFeatureValues()[0]
                  << std::endl;  // << " Features: " << this->getFeatureValues()
                                 // << std::endl;

        for (it = childs_.begin(); it != childs_.end(); + cit) {
          if (*it) it->print(tree_level + 1);
        }
      }

      virtual void printResult(unsigned int tree_level) const {
        for (unsigned int i = 0; i < tree_level; ++i) {
          std::cout << "\t";
        }
        assert(!this->getFeatureValues().empty());
        std::cout << this->getAlgorithmName() << " ET: " << timeEstimated
                  << "\tMT: " << timeNeeded
                  << "\tEE: " << 1 - (timeEstimated / timeNeeded)
                  << "\tEC: " << this->getFeatureValues()[0];
        if (this->getResultSize() != -1) {
          std::cout << "\tRR: " << this->getResultSize();
        }
        std::cout << std::endl;
        for (it = childs_.begin(); it != childs_.end(); + cit) {
          if (*it) it->printResult(tree_level + 1);
        }
      }

      virtual double getRecursiveExecutionTimeInNanoseconds() {
        double executiontime = this->getTimeNeeded();
        if (executiontime < 0) {
          executiontime = 0;
        }
        if (!hype::core::Runtime_Configuration::instance()
                 .isQueryChoppingEnabled()) {
          for (it = childs_.begin(); it != childs_.end(); + cit) {
            if (*it) {
              executiontime += it->getRecursiveExecutionTimeInNanoseconds();
            }
          }
        } else {
          double maxChildExecution = 0;
          for (it = childs_.begin(); it != childs_.end(); + cit) {
            if (*it &&
                it->getRecursiveExecutionTimeInNanoseconds() >
                    maxChildExecution) {
              maxChildExecution = it->getRecursiveExecutionTimeInNanoseconds();
            }
          }
          executiontime += maxChildExecution;
        }
        return executiontime;
      }
      virtual double getRecursiveEstimationTimeInNanoseconds() {
        double estimationTime = this->getTimeEstimated();
        if (estimationTime < 0) {
          estimationTime = 0;
        }
        if (!hype::core::Runtime_Configuration::instance()
                 .isQueryChoppingEnabled()) {
          for (it = childs_.begin(); it != childs_.end(); + cit) {
            if (*it) {
              estimationTime += it->getRecursiveEstimationTimeInNanoseconds();
            }
          }
        } else {
          double maxChildExecution = 0;
          for (it = childs_.begin(); it != childs_.end(); + cit) {
            if (*it &&
                it->getRecursiveExecutionTimeInNanoseconds() >
                    maxChildExecution) {
              maxChildExecution = it->getRecursiveEstimationTimeInNanoseconds();
            }
          }
          estimationTime += maxChildExecution;
        }
        return estimationTime;
      }
      void addChild(OperatorInputType child) { childs_.push_back(child); }

      /*
      virtual std::list<TypedOperator<OperatorOutputType>& > getOperatorQueue(){

              std::list<TypedOperator<OperatorOutputType>& > left_list =
      left_child_->getOperatorQueue();
              left_list.push_front(*this);

              std::list<TypedOperator<OperatorOutputType>& > right_list =
      right_child_->getOperatorQueue();

              left_list.insert(left_list.end(),right_list.begin(),right_list.end());
              return left_list;
              //this->left_child_->
      }*/

     private:
      std::list<OperatorInputType> childs_;
      //			boost::shared_ptr<TypedOperator<OperatorInputTypeLeftChild>
      //> left_child_;
      //			boost::shared_ptr<TypedOperator<OperatorInputTypeRightChild>
      //> right_child_;
      // boost::shared_ptr<TypedOperator<OperatorOutputType> > succsessor_;
    };

  }  // end namespace queryprocessing
}  // end namespace hype
