
#include <lookup_table/join_index.hpp>
#include <unittests/unittests.hpp>
#include <util/reduce_by_keys.hpp>
#include <util/tpch_benchmark.hpp>

#include <query_processing/query_processor.hpp>

#include <boost/lexical_cast.hpp>

//#include <boost/math/distributions/beta.hpp>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

#include <sys/mman.h>
#include <fstream>
#include <iostream>

#include <utility>

#include <stdint.h>
#include <core/column.hpp>
#include <statistics/histogram.hpp>

namespace CoGaDB {

using namespace query_processing;
using namespace std;

namespace unit_tests {

bool equi_width_histogram_test() {
  Column<int> v("Test_Col", INT);
  //        for(unsigned int i = 0; i<1000000; ++i){
  for (unsigned int i = 0; i < 100; ++i) {
    v.insert(i);
    //            v.insert(501);
  }

  SelectivityEstimatorPtr sel_est_(
      new EquiHeightHistogram<int>(v.data(), v.size()));
  cout << sel_est_->toString() << endl;

  std::vector<double> measured_rows;
  std::vector<double> estimated_rows;
  std::vector<double> estimation_error;

  for (int i = 0; i < 1000; i++) {
    int value = rand() % v.size();
    ValueComparator comp = ValueComparator(rand() % 5);
    Predicate p("Test_Col", boost::any(int(value)), ValueConstantPredicate,
                comp);
    hype::ProcessingDeviceID id = hype::PD0;
    ProcessorSpecification proc_spec(id);
    SelectionParam param(proc_spec, ValueConstantPredicate,
                         boost::any(int(value)), comp);
    PositionListPtr tids = v.selection(param);
    //            PositionListPtr tids = v.selection( boost::any(int(value)),
    //            comp);
    cout << "Predicate: " << p.toString() << endl;
    if (tids) {
      cout << "Real Selectivity: " << double(tids->size()) / v.size() << endl;
      measured_rows.push_back(tids->size());
    }
    double est_sel = sel_est_->getEstimatedSelectivity(p);
    double est_card = est_sel * v.size();
    cout << "Est Sel: " << est_sel << endl;
    estimated_rows.push_back(est_card);
    if (tids->size())
      estimation_error.push_back(abs((tids->size() - est_card) / tids->size()));
    else
      estimation_error.push_back(0);

    cout << "Real Cardinality: " << measured_rows[i] << endl;
    cout << "Estimated Cardinality: " << estimated_rows[i] << endl;
    cout << "Estimation Error: " << estimation_error[i] << endl;
    cout << "=========================================================="
         << endl;
  }

  double sum =
      std::accumulate(estimation_error.begin(), estimation_error.end(), 0);
  double avg_est_error = sum / estimation_error.size();

  cout << "Min Estimation Error: "
       << *std::min_element(estimation_error.begin(), estimation_error.end())
       << endl;
  cout << "Max Estimation Error: "
       << *std::max_element(estimation_error.begin(), estimation_error.end())
       << endl;
  cout << "Average Estimation Error: " << avg_est_error << endl;

  return true;
}

bool equi_width_histogram_range_test() {
  Column<int> v("Test_Col", INT);
  //        for(unsigned int i = 0; i<1000000; ++i){
  for (unsigned int i = 0; i < 100; ++i) {
    v.insert(i);
    //            v.insert(501);
  }

  SelectivityEstimatorPtr sel_est_(
      new EquiHeightHistogram<int>(v.data(), v.size()));
  cout << sel_est_->toString() << endl;

  std::vector<double> measured_rows;
  std::vector<double> estimated_rows;
  std::vector<double> estimation_error;

  for (int i = 0; i < 1000; i++) {
    int lower_value = rand() % v.size();
    int upper_value = rand() % v.size();
    if (upper_value < lower_value) std::swap(lower_value, upper_value);
    // ValueComparator comp=ValueComparator(rand()%2); //only GREATER, LESSER
    // predicates
    Predicate lower_pred("Test_Col", boost::any(int(lower_value)),
                         ValueConstantPredicate, GREATER);
    Predicate upper_pred("Test_Col", boost::any(int(upper_value)),
                         ValueConstantPredicate, LESSER);

    hype::ProcessingDeviceID id = hype::PD0;
    ProcessorSpecification proc_spec(id);
    SelectionParam param_lower(proc_spec, ValueConstantPredicate,
                               boost::any(int(lower_value)), GREATER);
    SelectionParam param_upper(proc_spec, ValueConstantPredicate,
                               boost::any(int(upper_value)), LESSER);

    PositionListPtr tids_lower = v.selection(param_lower);
    PositionListPtr tids_upper = v.selection(param_upper);
    PositionListPtr tids =
        computePositionListIntersection(tids_lower, tids_upper);

    cout << "Range Query: " << lower_pred.toString() << " AND "
         << upper_pred.toString() << endl;
    if (tids) {
      cout << "Real Selectivity: " << double(tids->size()) / v.size() << endl;
      measured_rows.push_back(tids->size());
    }
    double est_sel = sel_est_->getEstimatedSelectivity(lower_pred, upper_pred);
    double est_card = est_sel * v.size();
    cout << "Est Sel: " << est_sel << endl;
    estimated_rows.push_back(est_card);
    if (tids->size())
      estimation_error.push_back(abs((tids->size() - est_card) / tids->size()));
    else
      estimation_error.push_back(0);

    cout << "Real Cardinality: " << measured_rows[i] << endl;
    cout << "Estimated Cardinality: " << estimated_rows[i] << endl;
    cout << "Estimation Error: " << estimation_error[i] << endl;
    cout << "=========================================================="
         << endl;
  }

  double sum =
      std::accumulate(estimation_error.begin(), estimation_error.end(), 0);
  double avg_est_error = sum / estimation_error.size();

  cout << "Min Estimation Error: "
       << *std::min_element(estimation_error.begin(), estimation_error.end())
       << endl;
  cout << "Max Estimation Error: "
       << *std::max_element(estimation_error.begin(), estimation_error.end())
       << endl;
  cout << "Average Estimation Error: " << avg_est_error << endl;

  return true;
}
}

}  // end namespace CogaDB
