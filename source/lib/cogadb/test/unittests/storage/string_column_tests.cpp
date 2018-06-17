/*
 * File:   string_column_tests.cpp
 * Author: sebastian
 *
 * Created on 14. November 2015, 18:37
 */

#include <cstdlib>

using namespace std;

#include <boost/filesystem/operations.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <core/column.hpp>
#include <core/cstring_column.hpp>

namespace CoGaDB {

const std::string generateRandomString(size_t min_length, size_t max_length);

const std::string generateRandomString(size_t min_length, size_t max_length) {
  std::string characterfield = "abcdefghijklmnopqrstuvwxyz";

  std::string s;
  size_t offset = max_length - min_length;
  size_t length = (rand() % offset) + min_length;
  // last character is a zero byte
  if (length == max_length) length--;
  assert(length >= min_length);
  assert(length < max_length);
  for (size_t i = 0; i < length; i++) {
    s.push_back(characterfield[rand() % characterfield.size()]);
  }
  return s;
}

bool insertTest() {
  size_t min_string_length = 10;
  size_t max_string_length = 100;
  size_t num_elements = 80000;
  bool call_reserve_before_insert = false;
  bool unittest_print_checkpoints = true;
  bool validate_each_insert_c_string_col = false;

  std::vector<std::string> reference_data;
  boost::shared_ptr<Column<std::string> > string_col(
      new Column<std::string>("std::string", VARCHAR));
  boost::shared_ptr<CStringColumn> c_string_col(
      new CStringColumn("c_string", max_string_length));
  boost::shared_ptr<DictionaryCompressedColumn<std::string> > dict_comp_col(
      new DictionaryCompressedColumn<std::string>("std::string", VARCHAR));

  if (call_reserve_before_insert) {
    try {
      string_col->reserve(num_elements);
      c_string_col->reserve(num_elements);
    } catch (std::bad_alloc&) {
      return false;
    }
  }

  for (size_t i = 0; i < num_elements; ++i) {
    std::string str =
        generateRandomString(min_string_length, max_string_length);
    reference_data.push_back(str);
  }

  Timestamp begin_string_col = getTimestamp();
  for (size_t i = 0; i < num_elements; ++i) {
    string_col->push_back(reference_data[i]);
  }
  Timestamp end_string_col = getTimestamp();

  Timestamp begin_c_string_col = getTimestamp();
  if (validate_each_insert_c_string_col) {
    for (size_t i = 0; i < num_elements; ++i) {
      const char* c_str = reference_data[i].c_str();
      c_string_col->push_back(c_str);
      char** array = c_string_col->data();
      if (strcmp(array[i], c_str) != 0) {
        COGADB_FATAL_ERROR("" << i, "");
      }
    }
  } else {
    for (size_t i = 0; i < num_elements; ++i) {
      auto c_str = reference_data[i].c_str();
      c_string_col->push_back(c_str);
    }
  }
  Timestamp end_c_string_col = getTimestamp();

  Timestamp begin_dict_comp_col = getTimestamp();
  for (size_t i = 0; i < num_elements; ++i) {
    dict_comp_col->insert(reference_data[i]);
  }
  Timestamp end_dict_comp_col = getTimestamp();

  if (unittest_print_checkpoints) {
    std::cout << "Inserted Data..." << std::endl;
    double string_col_time_in_sec =
        double(end_string_col - begin_string_col) / (1024 * 1024 * 1024);
    double c_string_col_time_in_sec =
        double(end_c_string_col - begin_c_string_col) / (1024 * 1024 * 1024);
    double dict_comp_col_time_in_sec =
        double(end_dict_comp_col - begin_dict_comp_col) / (1024 * 1024 * 1024);
    std::cout << "Column<std::string>: " << string_col_time_in_sec << "s"
              << std::endl;
    std::cout << "CStringColumn: " << c_string_col_time_in_sec << "s"
              << std::endl;
    std::cout << "DictionaryCompressedColumn<std::string>: "
              << dict_comp_col_time_in_sec << "s" << std::endl;
  }

  assert(string_col->size() == num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    std::string* string_col_data = string_col->data();
    if (reference_data[i] != string_col_data[i]) {
      COGADB_FATAL_ERROR(
          "Unittest Failed!: Result of Column<std::string> is wrong!", "");
    }
  }

  if (unittest_print_checkpoints)
    std::cout << "Insert Test Column<std::string> passed..." << std::endl;

  assert(c_string_col->size() == num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    char** c_string_col_data = c_string_col->data();
    if (reference_data[i] != std::string(c_string_col_data[i])) {
      COGADB_ERROR(
          "Unittest Failed!: Result of CStringColumn is wrong!"
              << std::endl
              << "At position " << i << std::endl
              << " reference: '" << reference_data[i] << "'" << std::endl
              << " c_string_col: '" << std::string(c_string_col_data[i]) << "'",
          "");
      return false;
    }
  }
  if (unittest_print_checkpoints)
    std::cout << "Insert Test CStringColumn passed..." << std::endl;

  assert(dict_comp_col->size() == num_elements);
  ProcessorSpecification proc_spec(hype::PD0);
  boost::shared_ptr<Column<std::string> > dict_string_col =
      dict_comp_col->copyIntoDenseValueColumn(proc_spec);
  for (size_t i = 0; i < num_elements; ++i) {
    std::string* dict_string_col_data = dict_string_col->data();
    if (reference_data[i] != dict_string_col_data[i]) {
      COGADB_ERROR(
          "Unittest Failed!: Result of DictionaryCompressedColumn<std::string> "
          "is wrong!",
          "");
      return false;
    }
  }
  if (unittest_print_checkpoints)
    std::cout << "Insert Test DictionaryCompressedColumn<std::string> passed..."
              << std::endl;

  return true;
}

bool persistenceTest() {
  size_t min_string_length = 10;
  size_t max_string_length = 100;
  size_t num_elements = 80;
  std::string path_to_database = "./temp_database_unittest";
  std::vector<std::string> reference_data;
  {
    boost::shared_ptr<CStringColumn> c_string_col(
        new CStringColumn("c_string", max_string_length));

    for (size_t i = 0; i < num_elements; ++i) {
      std::string str =
          generateRandomString(min_string_length, max_string_length).c_str();
      reference_data.push_back(str);
      const char* val = str.c_str();
      c_string_col->push_back(val);
      char** array = c_string_col->data();
      if (strcmp(array[i], str.c_str()) != 0) {
        COGADB_FATAL_ERROR("" << i, "");
      }
    }

    if (boost::filesystem::exists(path_to_database)) {
      std::cerr << "Error: directory './temp_database_unittest' exists "
                   "already, delete it and restart."
                << std::endl;
      return false;
    }
    boost::filesystem::create_directory(path_to_database);
    if (!boost::filesystem::exists(path_to_database)) {
      std::cerr << "Error: Failed to create directory "
                   "'./temp_database_unittest', ensure you have write access "
                   "to the working directory."
                << std::endl;
      return false;
    }

    if (!c_string_col->store(path_to_database)) {
      COGADB_ERROR("Failed to store column!", "");
      boost::filesystem::remove_all(path_to_database);
      return false;
    }

    boost::shared_ptr<CStringColumn> loaded_c_string_col(
        new CStringColumn(c_string_col->getName(), 0));
    if (!loaded_c_string_col->load(path_to_database, LOAD_ALL_DATA)) {
      COGADB_ERROR("Failed to load column!", "");
      boost::filesystem::remove_all(path_to_database);
      return false;
    }

    if (c_string_col->size() != loaded_c_string_col->size()) {
      COGADB_ERROR(
          "Loaded column unequal to reference column! Something went wrong "
          "other with load or store.",
          "");
      boost::filesystem::remove_all(path_to_database);
      return false;
    }
    char** reference_array = c_string_col->data();
    char** value_array = loaded_c_string_col->data();
    for (size_t i = 0; i < c_string_col->size(); ++i) {
      if (strcmp(reference_array[i], value_array[i]) != 0) {
        COGADB_FATAL_ERROR("" << i, "");
      }
    }
  }

  /* now check whether load still works when the original column does no longer
   * exist */
  boost::shared_ptr<CStringColumn> loaded_c_string_col(
      new CStringColumn("c_string", 0));
  if (!loaded_c_string_col->load(path_to_database, LOAD_ALL_DATA)) {
    COGADB_ERROR("Failed to load column!", "");
    boost::filesystem::remove_all(path_to_database);
    return false;
  }

  char** array = loaded_c_string_col->data();
  for (size_t i = 0; i < num_elements; ++i) {
    std::string str = reference_data[i];
    reference_data.push_back(str);
    if (strcmp(array[i], str.c_str()) != 0) {
      COGADB_FATAL_ERROR("" << i, "");
    }
  }

  boost::filesystem::remove_all(path_to_database);
  return true;
}
}

int main(int argc, char** argv) {
  if (!CoGaDB::insertTest()) {
    COGADB_ERROR("Unittest " << __FILE__ << " Failed!", "");
    return -1;
  }
  if (!CoGaDB::persistenceTest()) {
    COGADB_ERROR("Unittest " << __FILE__ << " Failed!", "");
    return -1;
  }

  std::cout << "Unittest " << __FILE__ << " Passed!" << std::endl;

  return 0;
}
