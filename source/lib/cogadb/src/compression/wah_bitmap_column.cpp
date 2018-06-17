//
// Created by Sebastian Dorok on 27.07.15.
//

#include "compression/wah_bitmap_column.hpp"
#include <backends/gpu/util.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <util/functions.hpp>

namespace CoGaDB {

// WAHBitmapColumn::WAHBitmapColumn() : bitmap_(), number_of_rows_(0) { };
WAHBitmapColumn::WAHBitmapColumn(const std::string column_name)
    : words_(),
      counts_(),
      tailing_word_(0),
      number_of_rows_(0),
      column_name_(column_name),
      _last_index_position(0),
      _last_lookup_index(0) {}

WAHBitmapColumn::~WAHBitmapColumn() {}

void WAHBitmapColumn::setNextRow() { this->appendRow(1); }

void WAHBitmapColumn::unsetNextRow() { this->appendRow(0); }

bool WAHBitmapColumn::isRowSet(TID index) {
  return ((this->getRowValue(index) == 1) ? true : false);
}

bool WAHBitmapColumn::isFillWord(uint32_t word) {
  return ((word & word_type_bitmask) >> 31) == 1;
}

uint32_t WAHBitmapColumn::getFillWordValue(uint32_t word) {
  return (word & fill_word_value_bitmask) >> 30;
}

uint32_t WAHBitmapColumn::getFillWordCount(uint32_t word) {
  return word & fill_word_count_bitmask;
}

uint32_t WAHBitmapColumn::getLiteralWordValues(uint32_t word) {
  return word & literal_word_values_bitmask;
}

TID WAHBitmapColumn::getPrefixSum(const TID index) {
  assert(index < number_of_rows_);
  TID result = 0;
  TID index_in_word = index % number_of_values_per_word_;
  // check how many vector words we have to look at
  // TID containing_vector_index = index / number_of_values_per_word_;
  // TID containing_vector_index =
  // binary_search_find_nearest_greater(counts_.data(), counts_.size(), index);
  // index that indicates tailing word
  // TID tailing_word_index = number_of_rows_ / number_of_values_per_word_;
  // assert(containing_vector_index <= tailing_word_index);
  // count all ones from words before word our index is in
  // int compressed_index = 0;
  // int uncompressed_index = 0;
  // int fill_index = 0;
  /*for (; compressed_index < words_.size() && uncompressed_index <
  containing_vector_index;) {
      if (isFillWord(words_[compressed_index])) {
          // it is a fill word
          uint32_t fill_word_pop_count =
  getFillWordValue(words_[compressed_index]) * number_of_values_per_word_;
          result += fill_word_pop_count;
          fill_index++;
          uncompressed_index++;
          compressed_index += (getFillWordCount(words_[compressed_index]) ==
  fill_index);
      } else {
          // it is a literal word
          uint32_t literal_word_pop_count =
  pop_count(getLiteralWordValues(words_[compressed_index]));
          result += literal_word_pop_count;
          compressed_index++;
          uncompressed_index++;
          fill_index = 0;
      }
  }*/
  if (counts_.size() == 0) {
    // we have only a tailing word yet
    uint32_t bitmask = 0xFFFFFFFF
                       << ((number_of_rows_ % number_of_values_per_word_) -
                           index_in_word);
    uint32_t tailing_word_pop_count = pop_count(tailing_word_ & bitmask);
    return tailing_word_pop_count;
  }
  int i = 0;
  while (counts_[i] <= index) {
    if (isFillWord(words_[i])) {
      // it is a fill word
      uint32_t fill_word_pop_count = getFillWordValue(words_[i]) *
                                     getFillWordCount(words_[i]) *
                                     number_of_values_per_word_;
      result += fill_word_pop_count;
    } else {
      // it is a literal word
      uint32_t literal_word_pop_count =
          pop_count(getLiteralWordValues(words_[i]));
      result += literal_word_pop_count;
    }
    i++;
  }
  if (index >= counts_.back()) {
    // the index is in the tailing word, this also means that the previous while
    // loop already visited all words_
    // then we also have to look into the tailing word
    // last but not least: the tailing word
    uint32_t bitmask = 0xFFFFFFFF
                       << ((number_of_rows_ % number_of_values_per_word_) -
                           index_in_word);
    uint32_t tailing_word_pop_count = pop_count(tailing_word_ & bitmask);
    result += tailing_word_pop_count;
  } else {
    // index is not in the tailing word but in a word_
    if (isFillWord(words_[i])) {
      // it is a fill word
      if (i == 0) {
        uint32_t fill_word_pop_count =
            getFillWordValue(words_[i]) * index * number_of_values_per_word_;
        result += fill_word_pop_count;
      } else {
        uint32_t fill_word_pop_count = getFillWordValue(words_[i]) *
                                       (index - counts_[i - 1]) *
                                       number_of_values_per_word_;
        result += fill_word_pop_count;
      }
    } else {
      // it is a literal word
      uint32_t bitmask = 0xFFFFFFFF
                         << (number_of_values_per_word_ - index_in_word);
      uint32_t tailing_word_pop_count = pop_count(words_[i] & bitmask);
      result += tailing_word_pop_count;
    }
  }
  /*
  if (containing_vector_index < tailing_word_index) {
      uint32_t containing_word_pop_count = 0;
      if (isFillWord(words_[compressed_index])) {
          containing_word_pop_count = getFillWordValue(words_[compressed_index])
  * index_in_word;
      } else {
          uint32_t bitmask = 0xFFFFFFFF << (number_of_values_per_word_ -
  index_in_word);
          containing_word_pop_count = pop_count(words_[compressed_index] &
  bitmask);
      }
      result += containing_word_pop_count;
  }
  if (containing_vector_index == tailing_word_index) {
      // then we also have to look into the tailing word
      // last but not least: the tailing word
      uint32_t bitmask = 0xFFFFFFFF << ((number_of_rows_ %
  number_of_values_per_word_) - index_in_word);
      uint32_t tailing_word_pop_count = pop_count(tailing_word_ & bitmask);
      result += tailing_word_pop_count;
  }
   */
  return result;
}

uint32_t WAHBitmapColumn::getRowValue(TID index) {
  // index is zero based -> index 0 yields row 1
  assert(index < number_of_rows_);
  TID count = number_of_rows_;
  uint32_t containing_word = 0;
  // is the index stored in the tailing word
  count -= number_of_rows_ % number_of_values_per_word_;
  if (index >= count) {
    // yes it is in the tailing word
    containing_word = tailing_word_;
    // which entry is it in the word
    uint32_t offset = (uint32_t)index % number_of_values_per_word_;
    // create bit mask to retrieve compressed value
    // determine highest offset within word
    int highest_offset_in_tailing_word =
        (int)number_of_rows_ % number_of_values_per_word_;
    // extract compressed value
    uint32_t y =
        containing_word >> ((highest_offset_in_tailing_word - 1) - offset) & 1;
    // return uncompressed value
    return y;
  } else {
    if (counts_.size() == 0) {
      COGADB_FATAL_ERROR(
          "Count_ vector has no values. Column seems not to be in-memory.", "");
    }
    if (index >= number_of_rows_) {
      COGADB_FATAL_ERROR("Index out of bounds.", "");
    }
    TID word_index = 0;
    if (_last_lookup_index == 0 || index - 1 != _last_lookup_index) {
      word_index = binary_search_find_nearest_greater(counts_.data(),
                                                      counts_.size(), index);
      _last_index_position = word_index;
      _last_lookup_index = index;
    } else {
      word_index = fast_sequential_lookup(index);
    }
    // iterate through stored words
    containing_word = words_[word_index];
  }
  if (!isFillWord(containing_word)) {
    // which entry is it in the word
    uint32_t offset = (uint32_t)index % number_of_values_per_word_;
    // create bit mask to retrieve compressed value
    // extract compressed value
    uint32_t y =
        containing_word >> ((number_of_values_per_word_ - 1) - offset) & 1;
    // return uncompressed value
    return y;
  } else {
    return getFillWordValue(containing_word);
  }
}

TID WAHBitmapColumn::fast_sequential_lookup(TID index) {
  while (index >= counts_[_last_index_position]) {
    _last_index_position++;
  }
  return _last_index_position;
}

void WAHBitmapColumn::appendRow(uint32_t set) {
  // check whether the index requires a new word
  if (this->number_of_rows_ % number_of_values_per_word_ == 0) {
    // we require a new word, e.g., every 31st value requires a new 32 bit word
    // when we have 31 values per word
    if (number_of_rows_ > 0) {
      // we already have a tailing word, thus, we have to check whether it fits
      // to the previous fill word,
      // becomes part of a new fill word or becomes a literal word
      if (tailing_word_ == 0) {
        if (words_.size() > 0 && isFillWord(words_.back()) &&
            getFillWordValue(words_.back()) == 0) {
          // check whether number of words is less than maximum, otherwise
          // create a new fill
          uint32_t number_of_words_in_fill =
              words_.back() & fill_word_count_bitmask;
          if (number_of_words_in_fill <= fill_word_count_bitmask) {
            // add to previous fill
            words_.back()++;
            counts_.back() = counts_.back() + number_of_values_per_word_;
          } else {
            // create a new fill
            words_.push_back(uint32_t(initial_zero_fill_word));
            counts_.push_back(counts_.back() + number_of_values_per_word_);
          }
        } else {
          // it is the first fill word ever or after a literal word or fill one
          // word
          // create a new fill
          words_.push_back(uint32_t(initial_zero_fill_word));
          if (counts_.empty()) {
            counts_.push_back(number_of_values_per_word_);
          } else {
            counts_.push_back(counts_.back() + number_of_values_per_word_);
          }
        }
      } else if (tailing_word_ == literal_word_values_bitmask) {
        // if all possible bits in the tailing word are one this is equal to the
        // literal_word_values_bitmask
        if (words_.size() > 0 && isFillWord(words_.back()) &&
            getFillWordValue(words_.back()) == 1) {
          // check whether number of words is less than maximum, otherwise
          // create a new fill
          uint32_t number_of_words_in_fill =
              words_.back() & fill_word_count_bitmask;
          if (number_of_words_in_fill <= fill_word_count_bitmask) {
            // add to previous fill
            words_.back()++;
            counts_.back() = counts_.back() + number_of_values_per_word_;
          } else {
            // create a new fill
            words_.push_back(uint32_t(initial_one_fill_word));
            counts_.push_back(counts_.back() + number_of_values_per_word_);
          }
        } else {
          // it is the first fill word ever or after a literal word or fill one
          // word
          // create a new fill
          words_.push_back(uint32_t(initial_one_fill_word));
          if (counts_.empty()) {
            counts_.push_back(number_of_values_per_word_);
          } else {
            counts_.push_back(counts_.back() + number_of_values_per_word_);
          }
        }
      } else {
        // this tailing word is not suitable to use as a fill word, just add it
        // as literal word to
        // the words vector by setting most significant bit zero
        words_.push_back(tailing_word_ & literal_word_values_bitmask);
        if (counts_.empty()) {
          counts_.push_back(number_of_values_per_word_);
        } else {
          counts_.push_back(counts_.back() + number_of_values_per_word_);
        }
      }
    }
    // the new parameter value is used as tailing word
    tailing_word_ = set;
  } else {
    // just add this value to the tailing word
    tailing_word_ = (tailing_word_ << 1) | set;
  }
  // we just inserted a new word
  number_of_rows_++;
}

size_t WAHBitmapColumn::getSizeInBytes() {
  return 8 * sizeof(uint32_t) + sizeof(TID) +
         (words_.capacity() * sizeof(uint32_t)) +
         (counts_.capacity() * sizeof(TID));
}

TID WAHBitmapColumn::getNumberOfRows() { return number_of_rows_; }

bool WAHBitmapColumn::store(const std::string &path_to_database) {
  std::string path_to_column(path_to_database);
  path_to_column += "/";
  path_to_column += this->column_name_;
  if (!quiet && verbose && debug)
    std::cout << "Writing Column " << this->column_name_ << " to File "
              << path_to_column << std::endl;
  std::ofstream outfile(path_to_column.c_str(),
                        std::ios_base::binary | std::ios_base::out);
  if (!outfile.good()) {
    COGADB_ERROR("Could not store column '"
                     << path_to_column << "'!" << std::endl
                     << "Check whether you have write access to the database "
                     << "directory: '" << path_to_database << "'",
                 "");
  }
  boost::archive::binary_oarchive oa(outfile);
  oa << this->number_of_rows_;
  oa << this->tailing_word_;
  oa << this->words_;
  oa << this->counts_;

  outfile.flush();
  outfile.close();
  return true;
}

bool WAHBitmapColumn::load(const std::string &path_to_database) {
  std::string path_to_column(path_to_database);
  if (!quiet && verbose && debug)
    std::cout << "Loading column '" << this->column_name_ << "' from path '"
              << path_to_column << "'..." << std::endl;
  path_to_column += "/";
  path_to_column += this->column_name_;

  std::ifstream infile(path_to_column.c_str(),
                       std::ios_base::binary | std::ios_base::in);
  boost::archive::binary_iarchive ia(infile);
  ia >> this->number_of_rows_;
  ia >> this->tailing_word_;
  ia >> this->words_;
  ia >> this->counts_;

  infile.close();
  return true;
}
}
