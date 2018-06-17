#pragma once

namespace CoGaDB {
  class DictionaryCompressedCol {
   public:
    virtual uint32_t* getIdData() = 0;
    virtual unsigned int getNumberofDistinctValues() const = 0;
    virtual uint32_t getLargestID() const = 0;
    virtual size_t getNumberOfRows() const = 0;
  };
}