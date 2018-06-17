#pragma once

#include <compression/dictionary_compressed_column.hpp>
#include <core/column.hpp>
#include <iostream>

#include <math.h>
#include <algorithm>
#include <functional>
#include <string>
#include <utility>

#include <typeinfo>
#include <util/column_grouping_keys.hpp>
#include <util/getname.hpp>

#include <boost/mpl/comparison.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/if.hpp>
#include <boost/unordered_map.hpp>

#include <backends/cpu/aggregation.hpp>
#include <core/variable_manager.hpp>
#include <util/profiling.hpp>
#include <util/types.hpp>

namespace CoGaDB {

  template <class T>
  struct min_obj : std::binary_function<T, T, T> {
    T operator()(const T& x, const T& y) const { return x + 1; }
  };

  template <class T>
  struct max_obj : std::binary_function<T, T, T> {
    T operator()(const T& x, const T& y) const { return std::max(x, y); }
  };

  template <class T>
  struct count_obj : std::binary_function<T, T, T> {
    T operator()(const T& x, const T& y) const { return x + 1; }
  };

  template <class T>
  struct generic_min {
    generic_min() : minimal_value(std::numeric_limits<T>::max()) {}

    void aggregate(const T& val) {
      minimal_value = std::min(minimal_value, val);
    }

    T getValue() const { return minimal_value; }

    void reset() { minimal_value = std::numeric_limits<T>::max(); }
    T minimal_value;
  };

  template <class T>
  struct generic_max {
    generic_max() : maximal_value(std::numeric_limits<T>::min()) {}

    void aggregate(const T& val) {
      maximal_value = std::max(maximal_value, val);
    }

    T getValue() const { return maximal_value; }

    void reset() { maximal_value = std::numeric_limits<T>::min(); }
    T maximal_value;
  };

  template <class T>
  struct generic_sum {
    generic_sum() : sum(0) {}

    void aggregate(const T& val) { sum += val; }

    T getValue() const { return sum; }

    void reset() { sum = 0; }
    T sum;
  };

  template <class T>
  struct generic_count {
    generic_count() : count(0) {}

    template <class U>
    void aggregate(const U& val) {
      count++;
    }

    T getValue() const { return count; }

    void reset() { count = 0; }
    T count;
  };

  template <class T>
  struct generic_average {
    generic_average() : count(0), sum(0) {}

    void aggregate(const T& val) {
      sum += val;
      count++;
    }

    T getValue() const {
      if (count)
        return sum / count;
      else
        return 0;
    }

    void reset() {
      count = 0;
      sum = 0;
    }
    T count;
    T sum;
  };

  struct simple_genotype {
    simple_genotype()
        : count_adenine(0),
          count_cytosine(0),
          count_guanine(0),
          count_thymine(0),
          count_deletions(0),
          count(0) {
      threshold_min = VariableManager::instance().getVariableValueFloat(
          "genotype_frequency_min");
      threshold_max = VariableManager::instance().getVariableValueFloat(
          "genotype_frequency_max");
    }

    void aggregate(const std::string& val) {
      count_adenine += ("A" == val);
      count_cytosine += ("C" == val);
      count_thymine += ("T" == val);
      count_guanine += ("G" == val);
      count_deletions += ("X" == val);
      ++count;
    }

    std::string getValue() const {
      // threshold = VariableManager::instance().
      //      getVariableValueString("genotype_frequency");
      // maximum
      if (count_adenine >= threshold_max * count) return "A";
      if (count_cytosine >= threshold_max * count) return "C";
      if (count_guanine >= threshold_max * count) return "G";
      if (count_thymine >= threshold_max * count) return "T";
      if (count_deletions >= threshold_max * count) return "X";
      // between
      std::string ret_val;
      if (count_adenine >= threshold_min * count) ret_val += "A";
      if (count_cytosine >= threshold_min * count) ret_val += "C";
      if (count_guanine >= threshold_min * count) ret_val += "G";
      if (count_thymine >= threshold_min * count) ret_val += "T";
      if (count_deletions >= threshold_max * count) ret_val += "X";
      if (ret_val.empty()) ret_val += "N";
      // no call possible
      return ret_val;
    }

    void reset() {
      count_adenine = 0;
      count_cytosine = 0;
      count_guanine = 0;
      count_thymine = 0;
      count_deletions = 0;
      count = 0;
    }

    uint32_t count_adenine;
    uint32_t count_cytosine;
    uint32_t count_guanine;
    uint32_t count_thymine;
    uint32_t count_deletions;
    uint32_t count;
    float threshold_min;
    float threshold_max;
  };

  struct concat_bases {
    concat_bases() : ret_val("") {}

    void aggregate(const std::string& val) { ret_val += val; }

    std::string getValue() const { return ret_val; }

    void reset() { ret_val = ""; }
    std::string ret_val;
  };

  /**
   * Checks whether a ordered list of base characters is a homopolymer region.
   * For example, "AAAA" is a homopolymer region, where a insertion or deletion
   * is problematic to detect.
   * TODO Parameter that specifies how many consecutive bases of same kind must
   *appear.
   * Default is 4!
   * TODO Parameter that specifies the position of insertions/ deletion and a
   *second
   * parameter the number of surrounding bases.
   * Currently, we use a range query to fit determine the region to check.
   *
   * More info at
   *http://www.clcsupport.com/clcgenomicsworkbench/650/Filtering_variants_in_homopolymeric_regions.html
   *
   * Returns 1 if it IS a homopolymer region, and 0 if NOT.
   */
  struct is_homopolymer {
    is_homopolymer()
        : count_adenine(0),
          count_cytosine(0),
          count_guanine(0),
          count_thymine(0),
          last_val("") {}

    void aggregate(const std::string& val) {
      uint32_t check_a = "A" == val;
      uint32_t check_c = "C" == val;
      uint32_t check_t = "T" == val;
      uint32_t check_g = "G" == val;

      count_adenine += check_a;
      count_cytosine += check_c;
      count_thymine += check_t;
      count_guanine += check_g;

      count_adenine -= (1 - check_a) * count_adenine;
      count_cytosine -= (1 - check_c) * count_cytosine;
      count_thymine -= (1 - check_t) * count_thymine;
      count_guanine -= (1 - check_g) * count_guanine;
    }

    int getValue() const {
      return ((count_adenine + count_cytosine + count_thymine + count_guanine) >
              3);
    }

    void reset() {
      count_adenine = 0;
      count_cytosine = 0;
      count_thymine = 0;
      count_guanine = 0;
      last_val = "";
    }
    uint32_t count_adenine;
    uint32_t count_cytosine;
    uint32_t count_guanine;
    uint32_t count_thymine;
    std::string last_val;
  };

  struct genotype_statistics {
    genotype_statistics()
        : count_adenine(0),
          count_cytosine(0),
          count_guanine(0),
          count_thymine(0),
          count_deletions(0),
          count(0) {}

    void aggregate(const std::string& val) {
      count_adenine += ("A" == val);
      count_cytosine += ("C" == val);
      count_thymine += ("T" == val);
      count_guanine += ("G" == val);
      count_deletions += ("X" == val);
      ++count;
    }

    std::string getValue() const {
      std::string ret_val =
          "[A:" + boost::lexical_cast<std::string>(count_adenine) + ", C:" +
          boost::lexical_cast<std::string>(count_cytosine) + ", G:" +
          boost::lexical_cast<std::string>(count_guanine) + ", T:" +
          boost::lexical_cast<std::string>(count_thymine) + ", X:" +
          boost::lexical_cast<std::string>(count_deletions) + ", SUM:" +
          boost::lexical_cast<std::string>(count) + "]";
      return ret_val;
    }

    void reset() {
      count_adenine = 0;
      count_cytosine = 0;
      count_guanine = 0;
      count_thymine = 0;
      count_deletions = 0;
      count = 0;
    }

    uint32_t count_adenine;
    uint32_t count_cytosine;
    uint32_t count_guanine;
    uint32_t count_thymine;
    uint32_t count_deletions;
    uint32_t count;
    float threshold;
  };

  /* This redution assumes that the input data is sorted! */
  template <typename U, typename T, typename AggregationOperator,
            typename AggregationType>
  std::pair<ColumnPtr, ColumnPtr> generic_reduce_by_keys(
      ColumnGroupingKeysPtr grouping_keys, const T* values, size_t num_elements,
      AggregationOperator current_aggregate,
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<AggregationType> >
          aggregated_values = shared_pointer_namespace::shared_ptr<
              ColumnBaseTyped<AggregationType> >(new Column<AggregationType>(
              "", getAttributeType(typeid(AggregationType))))) {
    assert(grouping_keys->keys->size() == num_elements);
    GroupingKeys::value_type* keys = grouping_keys->keys->data();
    PositionListPtr new_keys = createPositionList(0, hype::PD_Memory_0);
    assert(new_keys != NULL);

    // if there is nothing to do, return valid but empty colums
    if (grouping_keys->keys->size() == 0) {
      return std::pair<ColumnPtr, ColumnPtr>(
          ColumnPtr(new_keys),
          aggregated_values);  // ColumnPtr(aggregated_values));
    }

    current_aggregate.reset();
    for (size_t i = 0; i < num_elements; i++) {
      if (i == 0) {
        current_aggregate.aggregate(values[i]);
        continue;
      }
      if (keys[i - 1] == keys[i]) {
        // std::cout << i <<  "SUM up values: " << val << "+" << (*values)[i] <<
        // "=" << binary_op(val,(*values)[i]) << std::endl;
        current_aggregate.aggregate(values[i]);
      } else {
        // std::cout <<  "found new key: " << (*keys)[i]  << " old: " <<
        // (*keys)[i-1] << " computed value: " << val << std::endl;
        TID tmp_key = i - 1;  // keys[i - 1];
        new_keys->insert(tmp_key);
        aggregated_values->insert(current_aggregate.getValue());
        current_aggregate.reset();
        current_aggregate.aggregate(values[i]);
      }
    }
    // write result of last tuple
    TID tmp_key = grouping_keys->keys->size() - 1;
    new_keys->insert(tmp_key);
    aggregated_values->insert(current_aggregate.getValue());
    return std::pair<ColumnPtr, ColumnPtr>(new_keys, aggregated_values);
  }

  template <typename U, typename T, typename AggregationOperator,
            typename AggregationType>
  std::pair<ColumnPtr, ColumnPtr>
  special_minimal_grouping_bits_hash_aggregation(
      ColumnGroupingKeysPtr grouping_keys, const T* values, size_t num_elements,
      AggregationOperator current_aggregate, bool write_group_tid_array,
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<AggregationType> >
          aggregated_values) {
    size_t max_group_key = pow(2, grouping_keys->required_number_of_bits);
    typedef GroupingKeys::value_type GroupingKeysType;
    GroupingKeysType* keys = grouping_keys->keys->data();
    PositionListPtr new_keys;

    // if there is nothing to do, return valid but empty colums
    if (grouping_keys->keys->size() == 0) {
      return std::pair<ColumnPtr, ColumnPtr>(
          boost::make_shared<Column<U> >(grouping_keys->keys->getName(),
                                         grouping_keys->keys->getType()),
          aggregated_values);
    }
    AggregationOperator* hashtable = new AggregationOperator[max_group_key];
    char* flag_array = (char*)malloc(max_group_key * sizeof(char));
    std::memset(flag_array, 0, max_group_key * sizeof(char));
    if (write_group_tid_array) {
      TID* tid_array = (TID*)malloc(max_group_key * sizeof(TID));
      //#pragma omp parallel for
      for (size_t i = 0; i < num_elements; ++i) {
        GroupingKeysType id = keys[i];
        hashtable[id].aggregate(values[i]);
        tid_array[id] = i;
        flag_array[id] = 1;
      }
      new_keys = createPositionList(0, hype::PD_Memory_0);
      assert(new_keys != NULL);
      for (size_t i = 0; i < max_group_key; ++i) {
        if (flag_array[i]) {
          new_keys->insert(tid_array[i]);
          aggregated_values->insert(hashtable[i].getValue());
        }
      }
      free(tid_array);
    } else {
      for (size_t i = 0; i < num_elements; ++i) {
        GroupingKeysType id = keys[i];
        hashtable[id].aggregate(values[i]);  // +=values[i];
        flag_array[id] = 1;
      }

      for (size_t i = 0; i < max_group_key; ++i) {
        if (flag_array[i]) {
          aggregated_values->insert(hashtable[i].getValue());
        }
      }
    }
    delete[] hashtable;
    free(flag_array);
    return std::pair<ColumnPtr, ColumnPtr>(new_keys, aggregated_values);
  }

  template <typename U, typename T, typename AggregationOperator,
            typename AggregationType>
  std::pair<ColumnPtr, ColumnPtr> generic_hash_aggregation(
      ColumnGroupingKeysPtr grouping_keys, const T* values, size_t num_elements,
      AggregationOperator current_aggregate, bool write_group_tid_array,
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<AggregationType> >
          aggregated_values = shared_pointer_namespace::shared_ptr<
              ColumnBaseTyped<AggregationType> >(new Column<AggregationType>(
              "", getAttributeType(typeid(AggregationType))))) {
    assert(grouping_keys->keys->size() == num_elements);
    /* Use optimized hash aggregation for columns with small elements only.*/
    if (grouping_keys->required_number_of_bits <= 14) {
      return special_minimal_grouping_bits_hash_aggregation<
          U, T, AggregationOperator, AggregationType>(
          grouping_keys, values, num_elements, current_aggregate,
          write_group_tid_array, aggregated_values);
    }
    typedef GroupingKeys::value_type GroupingKeysType;
    GroupingKeysType* keys = grouping_keys->keys->data();
    typedef std::pair<TID, AggregationOperator> Payload;
    typedef boost::unordered_map<GroupingKeysType, Payload,
                                 boost::hash<GroupingKeysType>,
                                 std::equal_to<GroupingKeysType> >
        HashTable;
    // create hash table
    HashTable hash_table;
    std::pair<typename HashTable::iterator, typename HashTable::iterator> range;
    typename HashTable::iterator it;
    size_t number_of_rows = num_elements;
    for (size_t i = 0; i < number_of_rows; ++i) {
      it = hash_table.find(keys[i]);
      if (it != hash_table.end()) {
        it->second.second.aggregate(values[i]);
      } else {
        std::pair<typename HashTable::iterator, bool> ret = hash_table.insert(
            std::make_pair(keys[i], Payload(i, AggregationOperator())));
        if (ret.second) {
          ret.first->second.second.reset();
          ret.first->second.second.aggregate(values[i]);
        }
      }
    }
    PositionListPtr new_keys = createPositionList(0, hype::PD_Memory_0);
    assert(new_keys != NULL);
    // if there is nothing to do, return valid but empty colums
    if (grouping_keys->keys->size() == 0) {
      return std::pair<ColumnPtr, ColumnPtr>(
          new_keys, aggregated_values);  // ColumnPtr(aggregated_values));
    }
    // write result
    for (it = hash_table.begin(); it != hash_table.end(); ++it) {
      new_keys->insert(it->second.first);
      aggregated_values->insert(it->second.second.getValue());
    }
    return std::pair<ColumnPtr, ColumnPtr>(new_keys, aggregated_values);
  }

  template <typename T>
  std::pair<ColumnPtr, ColumnPtr> hash_aggregation(ColumnGroupingKeysPtr keys,
                                                   const T* values,
                                                   size_t num_elements,
                                                   AggregationMethod agg_meth,
                                                   bool write_group_tid_array) {
    std::pair<ColumnPtr, ColumnPtr> result;
    typedef ColumnGroupingKeys::GroupingKeysType U;

    //        std::cout << "Required Bits for Grouping Key: " <<
    //        keys->required_number_of_bits << std::endl;

    COGADB_PCM_START_PROFILING("Hash-Based Aggregation", std::cout);

    if (agg_meth == SUM) {
      result = generic_hash_aggregation<U, T, generic_sum<T>, T>(
          keys, values, num_elements, generic_sum<T>(), write_group_tid_array);
    } else if (agg_meth == MIN) {
      result = generic_hash_aggregation<U, T, generic_min<T>, T>(
          keys, values, num_elements, generic_min<T>(), write_group_tid_array);
    } else if (agg_meth == MAX) {
      result = generic_hash_aggregation<U, T, generic_max<T>, T>(
          keys, values, num_elements, generic_max<T>(), write_group_tid_array);
    } else if (agg_meth == COUNT) {
      result = generic_hash_aggregation<U, T, generic_count<T>, T>(
          keys, values, num_elements, generic_count<T>(),
          write_group_tid_array);
    } else if (agg_meth == AVERAGE) {
      result = generic_hash_aggregation<U, T, generic_average<double>, double>(
          keys, values, num_elements, generic_average<double>(),
          write_group_tid_array);
    } else {
      std::cerr << "FATAL Error! Unknown Aggregation Method!" << agg_meth
                << std::endl;
    }

    bool show_core_output = true;
    bool show_socket_output = false;
    bool show_system_output = true;
    COGADB_PCM_STOP_PROFILING("Hash-Based Aggregation", std::cout, num_elements,
                              sizeof(T), show_core_output, show_socket_output,
                              show_system_output);

    return result;
  }

  template <>
  inline std::pair<ColumnPtr, ColumnPtr> hash_aggregation(
      ColumnGroupingKeysPtr keys, const std::string* values,
      size_t num_elements, AggregationMethod agg_meth,
      bool write_group_tid_array) {
    std::pair<ColumnPtr, ColumnPtr> result;
    typedef ColumnGroupingKeys::GroupingKeysType U;
    if (agg_meth == COUNT) {
      result =
          generic_hash_aggregation<U, std::string, generic_count<int>, int>(
              keys, values, num_elements, generic_count<int>(),
              write_group_tid_array);
    } else if (agg_meth == AGG_GENOTYPE) {
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<std::string> >
          aggregated_values(
              new DictionaryCompressedColumn<std::string>("", VARCHAR));
      result = generic_hash_aggregation<U, std::string, simple_genotype,
                                        std::string>(
          keys, values, num_elements, simple_genotype(), write_group_tid_array,
          aggregated_values);
    } else if (agg_meth == AGG_CONCAT_BASES) {
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<std::string> >
          aggregated_values(new Column<std::string>("", VARCHAR));
      result =
          generic_hash_aggregation<U, std::string, concat_bases, std::string>(
              keys, values, num_elements, concat_bases(), write_group_tid_array,
              aggregated_values);
    } else if (agg_meth == AGG_IS_HOMOPOLYMER) {
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<int> >
      aggregated_values(new Column<int>("", INT));
      result = generic_hash_aggregation<U, std::string, is_homopolymer, int>(
          keys, values, num_elements, is_homopolymer(), write_group_tid_array,
          aggregated_values);
    } else if (agg_meth == AGG_GENOTYPE_STATISTICS) {
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<std::string> >
          aggregated_values(new Column<std::string>("", VARCHAR));
      result = generic_hash_aggregation<U, std::string, genotype_statistics,
                                        std::string>(
          keys, values, num_elements, genotype_statistics(),
          write_group_tid_array, aggregated_values);
    } else {
      std::cerr << "FATAL Error! Aggregation Method not supported on data type "
                   "'VARCHAR'!"
                << util::getName(agg_meth) << std::endl;
    }
    return result;
  }

  template <typename T>
  std::pair<ColumnPtr, ColumnPtr> reduce_by_keys(ColumnGroupingKeysPtr keys,
                                                 const T* values,
                                                 size_t num_elements,
                                                 AggregationMethod agg_meth) {
    std::pair<ColumnPtr, ColumnPtr> result;
    typedef ColumnGroupingKeys::GroupingKeysType U;
    if (agg_meth == SUM) {
      result = generic_reduce_by_keys<U, T, generic_sum<double>, double>(
          keys, values, num_elements, generic_sum<double>());
    } else if (agg_meth == MIN) {
      result = generic_reduce_by_keys<U, T, generic_min<T>, T>(
          keys, values, num_elements, generic_min<T>());
    } else if (agg_meth == MAX) {
      result = generic_reduce_by_keys<U, T, generic_max<T>, T>(
          keys, values, num_elements, generic_max<T>());
    } else if (agg_meth == COUNT) {
      result = generic_reduce_by_keys<U, T, generic_count<T>, T>(
          keys, values, num_elements, generic_count<T>());
    } else if (agg_meth == AVERAGE) {
      result = generic_reduce_by_keys<U, T, generic_average<double>, double>(
          keys, values, num_elements, generic_average<double>());
    } else {
      std::cerr << "FATAL Error! Unknown Aggregation Method!" << agg_meth
                << std::endl;
    }
    return result;
  }

  template <>
  inline std::pair<ColumnPtr, ColumnPtr> reduce_by_keys(
      ColumnGroupingKeysPtr keys, const std::string* values,
      size_t num_elements, AggregationMethod agg_meth) {
    std::pair<ColumnPtr, ColumnPtr> result;
    typedef ColumnGroupingKeys::GroupingKeysType U;
    std::string name_of_aggregated_column = "";
    if (agg_meth == COUNT) {
      result = generic_reduce_by_keys<U, std::string, generic_count<int>, int>(
          keys, values, num_elements, generic_count<int>());
    } else if (agg_meth == AGG_GENOTYPE) {
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<std::string> >
          aggregated_values(new DictionaryCompressedColumn<std::string>(
              name_of_aggregated_column, VARCHAR));
      result =
          generic_reduce_by_keys<U, std::string, simple_genotype, std::string>(
              keys, values, num_elements, simple_genotype(), aggregated_values);
    } else if (agg_meth == AGG_CONCAT_BASES) {
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<std::string> >
          aggregated_values(new DictionaryCompressedColumn<std::string>(
              name_of_aggregated_column, VARCHAR));
      result =
          generic_reduce_by_keys<U, std::string, concat_bases, std::string>(
              keys, values, num_elements, concat_bases(), aggregated_values);
    } else if (agg_meth == AGG_IS_HOMOPOLYMER) {
      shared_pointer_namespace::shared_ptr<ColumnBaseTyped<int> >
      aggregated_values(new Column<int>(name_of_aggregated_column, INT));
      result = generic_reduce_by_keys<U, std::string, is_homopolymer, int>(
          keys, values, num_elements, is_homopolymer(), aggregated_values);
    } else {
      std::cerr << "FATAL Error! Aggregation Method not supported on data type "
                   "'VARCHAR'!"
                << util::getName(agg_meth) << std::endl;
    }
    return result;
  }

  //    template <>
  //    inline std::pair<ColumnPtr, ColumnPtr>
  //    reduce_by_keys<char*>(ColumnGroupingKeysPtr keys, const char** values,
  //    size_t num_elements, AggregationMethod agg_meth) {
  //        std::pair<ColumnPtr, ColumnPtr> result;
  ////        typedef ColumnGroupingKeys::GroupingKeysType U;
  ////        std::string name_of_aggregated_column = "";
  ////        if (agg_meth == COUNT) {
  ////            result = generic_reduce_by_keys<U, std::string,
  /// generic_count<int>, int>(keys, values, num_elements,
  /// generic_count<int>());
  ////        } else if (agg_meth == AGG_GENOTYPE) {
  ////
  /// shared_pointer_namespace::shared_ptr<ColumnBaseTyped<std::string> >
  /// aggregated_values(new
  /// DictionaryCompressedColumn<std::string>(name_of_aggregated_column,
  /// VARCHAR));
  ////            result = generic_reduce_by_keys<U, std::string,
  /// simple_genotype, std::string>(keys, values, num_elements,
  /// simple_genotype(), aggregated_values);
  ////        } else if (agg_meth == AGG_CONCAT_BASES) {
  ////
  /// shared_pointer_namespace::shared_ptr<ColumnBaseTyped<std::string> >
  /// aggregated_values(new
  /// DictionaryCompressedColumn<std::string>(name_of_aggregated_column,
  /// VARCHAR));
  ////            result = generic_reduce_by_keys<U, std::string, concat_bases,
  /// std::string>(keys, values, num_elements, concat_bases(),
  /// aggregated_values);
  ////        } else if (agg_meth == AGG_IS_HOMOPOLYMER) {
  ////            shared_pointer_namespace::shared_ptr<ColumnBaseTyped<int> >
  /// aggregated_values(new Column<int>(name_of_aggregated_column, INT));
  ////            result = generic_reduce_by_keys<U, std::string,
  /// is_homopolymer, int>(keys, values, num_elements, is_homopolymer(),
  /// aggregated_values);
  ////        } else {
  ////            std::cerr << "FATAL Error! Aggregation Method not supported on
  /// data type 'VARCHAR'!" << util::getName(agg_meth) << std::endl;
  ////        }
  //        COGADB_FATAL_ERROR("Called unimplemented method!","");
  //        return result;
  //    }

}  // end namespace CogaDB
