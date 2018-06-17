//
// Created by basti on 27.07.15.
//

#ifndef GPUDBMS_BITMAP_COLUMN_HPP
#define GPUDBMS_BITMAP_COLUMN_HPP

#pragma once

#include <boost/shared_ptr.hpp>
#include <core/global_definitions.hpp>

namespace CoGaDB {

  class WAHBitmapColumn {
   public:
    typedef boost::shared_ptr<WAHBitmapColumn> WAHBitmapColumnPtr;

    WAHBitmapColumn(const std::string column_name);

    ~WAHBitmapColumn();

    void setNextRow();

    void unsetNextRow();

    bool isRowSet(TID index);

    TID getPrefixSum(const TID index);

    TID getNumberOfRows();

    size_t getSizeInBytes();

    bool store(const std::string &path_to_database);

    bool load(const std::string &path_to_database);

   private:
    // the number of values that can be stored per word
    static const uint32_t number_of_values_per_word_ = 31;
    // contains all words be it fill or literal ones
    std::vector<uint32_t> words_;

    // stores maximum id + 1 that is stored in a word -> can be used for faster
    // index lookup
    std::vector<TID> counts_;

    // the last word within the data structure - here new values go
    uint32_t tailing_word_;
    // indicates the current number of rows stored in this column
    TID number_of_rows_;

    static const uint32_t word_type_bitmask = 0x80000000;
    static const uint32_t fill_word_value_bitmask = 0x40000000;
    static const uint32_t fill_word_count_bitmask = 0x3FFFFFFF;
    static const uint32_t initial_zero_fill_word = 0x80000001;
    static const uint32_t initial_one_fill_word = 0xC0000001;

    static const uint32_t literal_word_values_bitmask = 0x7FFFFFFF;

    uint32_t getRowValue(TID index);

    void appendRow(uint32_t set);

    bool isFillWord(uint32_t word);

    // assumes that word is a fill word (check before with isFillWord())
    uint32_t getFillWordValue(uint32_t word);

    // assumes that word is a fill word (check before with isFillWord())
    uint32_t getFillWordCount(uint32_t word);

    // assumes that word is no fill word (check before with isFillWord())
    uint32_t getLiteralWordValues(uint32_t word);

    // unique identifier for column, mainly used for persisting data
    std::string column_name_;

    /** FOR UNIT TESTING: ALLOWS ACCESS TO PRIVATE MEMBERS **/
    /**
     * https://code.google.com/p/googletest/wiki/AdvancedGuide#Private_Class_Members
     * **/
    friend class WAHBitmapColumnTest;

    TID _last_index_position;
    TID _last_lookup_index;

    TID fast_sequential_lookup(TID index);
  };

  typedef WAHBitmapColumn::WAHBitmapColumnPtr WAHBitmapColumnPtr;

}  // end namespace CoGaDB

#endif  // GPUDBMS_BITMAP_COLUMN_HPP
