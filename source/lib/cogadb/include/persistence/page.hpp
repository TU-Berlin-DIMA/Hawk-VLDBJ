#pragma once
#include <core/global_definitions.hpp>
#include <fstream>
#include <iostream>

#define VARCHAR_LENGTH_BYTES \
  2;  // Number of Bytes for decoding the length of a string
#define BIT_COMPRESSED 0;  // C -> Is page content compressed?
#define BIT_TYPE_LOW 1;    // T -> datatype of page content
#define BIT_TYPE_HIGH 2;   // T
#define BIT_DIRTY 3;       // D -> Page modified?
#define BIT_USAGE 4;       // U -> Is Page in usage?
#define BIT_LENGTH 5;      // L -> Count of size-byte's
#define DATA_OFFSET 1;     // Number of bytes where data starts in file

namespace CoGaDB {
  /*! \n
   * | ? | ? | L | U | D | T | T | C | -> Status Byte (First Byte in FileHeader)
   *\n
   * | S | S | S | S | S | S | S | S | -> Size of inserted values in bytes \n
   * | S | S | S | S | S | S | S | S | -> nur bei L = 1 \n
   * | O | O | O | O | O | O | O | O | -> Offest in File \n
   * | O | O | O | O | O | O | O | O | \n
   * | O | O | O | O | O | O | O | O | \n
   * | O | O | O | O | O | O | O | O | \n
   * | N | N | N | N | N | N | N | N | -> Value count \n
   * | N | N | N | N | N | N | N | N | \n
   *
   * = 7 Byte PageHeader (Not FileHeader)
   * because of max 2^16 Bytes (2*S) => 64 kb data per page *
   */

  struct Header {
    /*! Status of this page */
    char Status;
    /*! Current data size in bytes */
    unsigned short Size;
    /*! file offset -> maybe exlude to buffer manager */
    int Offset;
    /*! Number of values stored in this page */
    unsigned short Count;
  };
  /*! \brief Representation of a Page (content from file). Each page holds data
   * from one column. However, all data at one page will be of same type. */
  class Page {
   private:
    /*! Information of this page (7 bytes). */
    Header _header;
    /*! The data of the page -> all of the same type. */
    char* _data;
    bool _isFull;
    std::vector<unsigned int> _value_offsets;

   public:
    /* Use for new columns */
    Page(bool, AttributeType, bool);
    Page(const Page&);
    Page& operator=(const Page&);
    /* Use for loading columns from hdd */
    Page(std::ifstream&, char);
    virtual ~Page();
    bool isCompressed() const;
    AttributeType getType() const;
    bool isDirty() const;
    bool isInUsage() const;
    /*! Gets size of the current holding data (in bytes). */
    unsigned short getDataSize() const;
    int getFileOffset() const;
    /*! Gets the max. size of the data this page can hold (in bytes). */
    unsigned short getMaxDataSize() const;
    void printStatus() const;
    int getSizeOfOneValue() const;
    void print() const;
    bool isFull() const;
    char* getData();
    unsigned short count() const;
    char* getStringByIndex(unsigned int index, unsigned short& valueSize);
    char* getValueByIndex(unsigned int index, unsigned short& valueSize);

   protected:
    bool appendData(std::ifstream&);
    char getStatusBitAt(char) const;
    bool append(std::ifstream&);
    bool appendVarchar(std::ifstream&);
    void setStatusBit(char, bool);
    void printInt(unsigned short) const;
    void printFloat(unsigned short) const;
    void printBool(unsigned short) const;
    int printVarchar(unsigned short) const;
  };

  /*! SharedPointer to a Page */
  typedef shared_pointer_namespace::shared_ptr<Page> PagePtr;
}