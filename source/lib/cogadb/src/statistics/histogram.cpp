
#include <cmath>
#include <core/column.hpp>
#include <core/global_definitions.hpp>
#include <statistics/histogram.hpp>

#include <core/selection_expression.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
using namespace std;

template <typename T>
EquiHeightHistogram<T>::EquiHeightHistogram(ColumnBaseTyped<T>* col,
                                            unsigned int number_of_buckets)
    : buckets_(number_of_buckets), number_of_values_per_bucket_() {
  //            if(!col) return;
  //            typename ColumnBaseTyped<T>::TypedVectorPtr plain_value_column =
  //            col->copyIntoPlainVector();
  //            std::sort(plain_value_column->begin(),
  //            plain_value_column->end());
  ////            ColumnPtr copy = col->copy();
  ////            PositionListPtr tids = copy->sort();
  ////            ColumnPtr sorted = copy->gather(tids);
  ////            shared_pointer_namespace::shared_ptr<Column<T> >
  /// plain_value_column =
  /// shared_pointer_namespace::dynamic_pointer_cast<Column<T> >(sorted);
  //            if(plain_value_column){
  //                T* array = plain_value_column->data();
  //                size_t array_size = plain_value_column->size();
  //                size_t chunk_size = array_size/number_of_buckets;
  //                size_t bucket_id = 0;
  //                size_t value_counter = 0;
  //
  //                for (unsigned int i = 0; i < array_size; ++i,
  //                ++value_counter) {
  //                    if(value_counter==chunk_size){
  //                        bucket_id++;
  //                    }
  //                    if(bucket_id>0)
  //                        buckets_[bucket_id].first=buckets_[bucket_id-1].second;
  //                    buckets_[bucket_id].second=array[i];
  //                }
  //            }
}

template <typename T>
EquiHeightHistogram<T>::EquiHeightHistogram(const T* array, size_t array_size,
                                            unsigned int number_of_buckets)
    : buckets_(number_of_buckets),
      number_of_distinct_keys_per_bucket_(number_of_buckets),
      number_of_values_per_bucket_(0),
      number_of_values_in_column_(0) {
  if (array && array_size > 0) {
    if (array_size <= 1000) buckets_.resize(100);
    if (array_size <= 100) buckets_.resize(10);
    if (array_size <= 10) buckets_.resize(2);
    if (array_size <= 2) buckets_.resize(1);

    size_t chunk_size = array_size / buckets_.size();
    size_t bucket_id = 0;
    size_t value_counter = 0;
    size_t distinct_value_counter = 1;

    number_of_values_per_bucket_ = chunk_size;
    number_of_values_in_column_ = array_size;

    // init first bucket
    buckets_[0].first = array[0];
    if (chunk_size < array_size)
      buckets_[0].second = array[chunk_size];
    else
      buckets_[0].second = array[1];
    for (unsigned int i = 0; i < array_size; ++i, ++value_counter) {
      if (i < array_size - 1 && array[i] < array[i + 1])
        distinct_value_counter++;
      // are we at end of a bucket, or are we at the end of the last bucket?
      if (value_counter == chunk_size || i + value_counter == array_size) {
        this->number_of_distinct_keys_per_bucket_[bucket_id] =
            distinct_value_counter;
        distinct_value_counter = 1;
        buckets_[bucket_id].second = array[i];
        if (bucket_id > 0)
          buckets_[bucket_id].first = buckets_[bucket_id - 1].second;
        bucket_id++;
        value_counter = 0;
        //                        if(bucket_id==buckets_.size()){
        //                            //init last value in histogram
        //                            buckets_.back().second=array[array_size-1];
        //                            break;
        //                        }

        //                       buckets_[bucket_id].second=array[i];
        //                        if(bucket_id>0)
        //                            buckets_[bucket_id].first=buckets_[bucket_id-1].second;
      }
      if (bucket_id == buckets_.size()) {
        // init last value in histogram
        buckets_.back().second = array[array_size - 1];
        break;
      }
    }
  }
}

template <typename T>
size_t EquiHeightHistogram<T>::countNumberOfExpectedMatches(
    T value, ValueComparator comp) const {
  //                    T value = boost::any_cast<T>(pred.getConstant());
  //            ValueComparator comp = pred.getValueComparator();

  // find first and last bucket containing the value
  size_t start_bucket_id = 0;
  size_t end_bucket_id = buckets_.size() - 1;

  // for (size_t i = buckets_.size(); i >0; --i) {
  for (size_t i = 0; i < buckets_.size(); ++i) {
    if (buckets_[i].first < value) {  // && buckets_[i].second >= value){
      start_bucket_id = i;
      // break;
    }
  }
  for (size_t i = 0; i < buckets_.size(); ++i) {
    if (buckets_[i].second > value) {  // && buckets_[i].second >= value){
      end_bucket_id = i;
      break;
    }
  }

  //            cout << "Apply Predicate " << pred.toString() << " to
  //            histogram..." << endl;
  cout << "Apply Predicate 'Column" << util::getName(comp) << value
       << "' to histogram..." << endl;
  cout << "start bucket id: " << start_bucket_id << " ("
       << buckets_[start_bucket_id].first << ", "
       << buckets_[start_bucket_id].second << ")" << endl;

  cout << "end bucket id: " << end_bucket_id << " ("
       << buckets_[end_bucket_id].first << ", "
       << buckets_[end_bucket_id].second << ")" << endl;

  double estimated_number_of_matches = 0;

  if (comp == EQUAL) {
    for (size_t i = start_bucket_id; i < end_bucket_id; ++i) {
      double number_of_distinct_values_in_bucket =
          number_of_distinct_keys_per_bucket_[i];

      if (i == start_bucket_id) {
        // bucket with first occurence of value
        // assume equal distribution of values in a single bucket
        estimated_number_of_matches +=
            ceil(double(number_of_values_per_bucket_) /
                 number_of_distinct_values_in_bucket);
        // consider last bucket, but check whether we have only
        // a single bucket, so we do not count values twice
      } else if (i == end_bucket_id - 1 && i > start_bucket_id) {
        // bucket with last occurence of value
        estimated_number_of_matches +=
            ceil(double(number_of_values_per_bucket_) /
                 number_of_distinct_values_in_bucket);
      } else {
        // bucket is completely filled with single value
        estimated_number_of_matches += number_of_values_per_bucket_;
      }
    }
    // sum up number of values per bucket for all buckets in range
    // [bucket0,end_bucked_id]
    // this is for predicates such as x<10 or y<=15
  } else if (comp == LESSER || comp == LESSER_EQUAL) {
    for (size_t i = 0; i < end_bucket_id; ++i) {
      estimated_number_of_matches += number_of_values_per_bucket_;
    }
    // sum up number of values per bucket for all buckets in range
    // [start_bucket_id,last_bucket]
    // this is for predicates such as x>10 or y>=15
  } else if (comp == GREATER || comp == GREATER_EQUAL) {
    for (size_t i = start_bucket_id; i < buckets_.size(); ++i) {
      estimated_number_of_matches += number_of_values_per_bucket_;
    }
  }

  return estimated_number_of_matches;
}

template <typename T>
double EquiHeightHistogram<T>::getEstimatedSelectivity(
    const Predicate& pred) const {
  if (pred.getPredicateType() == ValueValuePredicate) {
    COGADB_ERROR("Cannot estimate selectivity of predicate '"
                     << pred.toString() << "' using a single histogram!",
                 "");
    return 0.1;
  }

  // now we can be sure we have a simple predicate
  assert(pred.getPredicateType() == ValueConstantPredicate);
  if (pred.getConstant().type() != typeid(T)) {
    COGADB_FATAL_ERROR("Typemismatch for column", "");
  }

  T value = boost::any_cast<T>(pred.getConstant());
  ValueComparator comp = pred.getValueComparator();

  size_t estimated_matches = countNumberOfExpectedMatches(value, comp);
  cout << "Estimated matches: " << estimated_matches << endl;
  if (number_of_values_in_column_ > 0)
    return double(estimated_matches) / number_of_values_in_column_;
  else
    return double(0);

  /*
  //find first and last bucket containing the value
  size_t start_bucket_id=0;
  size_t end_bucket_id=buckets_.size()-1;

  for (size_t i = 0; i < buckets_.size(); ++i) {
      if(buckets_[i].first<=value){ // && buckets_[i].second >= value){
          start_bucket_id=i;
      }
      if(buckets_[i].second>value){ // && buckets_[i].second >= value){
          end_bucket_id=i;
      }
  }

  cout << "Apply Predicate 'Column" << util::getName(comp) << value << "' to
histogram..." << endl;
  cout << "start bucket id: " << start_bucket_id
       << "(" << buckets_[start_bucket_id].first << ", " <<
buckets_[end_bucket_id].second << ")" << endl;

  double estimated_number_of_matches=0;

  if(comp==EQUAL){
      for (size_t i = start_bucket_id; i < end_bucket_id; ++i) {
          double number_of_distinct_values_in_bucket=1;

          if(i==0){
              //bucket with first occurence of value
// = buckets_[i].second-buckets_[i].first;
               estimated_number_of_matches+=number_of_values_per_bucket_/number_of_distinct_values_in_bucket;
          }else if(i==end_bucket_id-1 && i>0){
              //bucket with last occurence of value
               estimated_number_of_matches+=number_of_values_per_bucket_/number_of_distinct_values_in_bucket;
          }else{
              //bucket is completely filled with single value
              estimated_number_of_matches+=number_of_values_per_bucket_;
          }

      }
  }else if(comp==LESSER || comp==LESSER_EQUAL){

  }else if(comp==GREATER || comp==GREATER_EQUAL){

  }       */

  // return 0;
}

template <typename T>
double EquiHeightHistogram<T>::getEstimatedSelectivity(
    const Predicate& lower_pred, const Predicate& upper_pred) const {
  assert(lower_pred.getPredicateType() == ValueConstantPredicate);
  assert(upper_pred.getPredicateType() == ValueConstantPredicate);

  T lower_value = boost::any_cast<T>(lower_pred.getConstant());
  T upper_value = boost::any_cast<T>(upper_pred.getConstant());
  if (lower_value > upper_value) {
    return getEstimatedSelectivity(upper_pred, lower_pred);
  }

  size_t estimated_matches_in_interval = 0;

  if ((lower_pred.getValueComparator() == LESSER ||
       lower_pred.getValueComparator() == LESSER_EQUAL) &&
      (upper_pred.getValueComparator() == LESSER ||
       upper_pred.getValueComparator() == LESSER_EQUAL)) {
    estimated_matches_in_interval = countNumberOfExpectedMatches(
        lower_value, lower_pred.getValueComparator());

  } else if ((lower_pred.getValueComparator() == GREATER ||
              lower_pred.getValueComparator() == GREATER_EQUAL) &&
             (upper_pred.getValueComparator() == GREATER ||
              upper_pred.getValueComparator() == GREATER_EQUAL)) {
    estimated_matches_in_interval = countNumberOfExpectedMatches(
        upper_value, upper_pred.getValueComparator());

  } else if ((lower_pred.getValueComparator() == GREATER ||
              lower_pred.getValueComparator() == GREATER_EQUAL) &&
             (upper_pred.getValueComparator() == LESSER ||
              upper_pred.getValueComparator() == LESSER_EQUAL)) {
    // case: 10 < x < 20
    // rewrite lower predicate 10 < x to: x <=10
    // then we can estimate the cardinality with two histogram lookups

    ValueComparator comp = lower_pred.getValueComparator();

    if (comp == LESSER) {
      comp = GREATER;
    } else if (comp == GREATER) {
      comp = LESSER;
    } else if (comp == LESSER_EQUAL) {
      comp = GREATER_EQUAL;
    } else if (comp == GREATER_EQUAL) {
      comp = LESSER_EQUAL;
    }

    size_t estimated_matches_lower_pred =
        countNumberOfExpectedMatches(lower_value, comp);
    size_t estimated_matches_upper_pred = countNumberOfExpectedMatches(
        upper_value, upper_pred.getValueComparator());
    assert(estimated_matches_upper_pred >= estimated_matches_lower_pred);
    estimated_matches_in_interval =
        estimated_matches_upper_pred - estimated_matches_lower_pred;

  } else {
    COGADB_FATAL_ERROR("Invalid value range: " << lower_pred.toString()
                                               << " AND "
                                               << upper_pred.toString(),
                       "");
  }

  cout << "Estimated matches for range query '" << lower_pred.toString()
       << " AND " << upper_pred.toString()
       << "': " << estimated_matches_in_interval << endl;
  if (number_of_values_in_column_ > 0)
    return double(estimated_matches_in_interval) / number_of_values_in_column_;
  else
    return double(0);

  return 0;
}

template <typename T>
std::string EquiHeightHistogram<T>::getName() const {
  return "EquiHeightHistogram";
}

template <typename T>
SelectivityEstimationStrategy
EquiHeightHistogram<T>::getSelectivityEstimationStrategy() const {
  return EQUI_HEIGHT_HISTOGRAM;
}

template <typename T>
std::string EquiHeightHistogram<T>::toString() const {
  std::stringstream ss;
  ss << "Selectivity Estimator: " << getName() << endl;
  ss << "Number of Values in Column: " << number_of_values_in_column_ << endl;
  ss << "Number of Buckets: " << buckets_.size() << endl;
  for (size_t i = 0; i < buckets_.size(); i++) {
    ss << "\tBucket " << i << ": Begin " << buckets_[i].first
       << ", End: " << buckets_[i].second
       << " Distinct Values: " << number_of_distinct_keys_per_bucket_[i]
       << endl;
  }
  return ss.str();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(EquiHeightHistogram)

}  // end namespace CoGaDB
