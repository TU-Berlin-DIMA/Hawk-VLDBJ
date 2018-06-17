#pragma once
#include <core/global_definitions.hpp>
#include <persistence/buffer_object.hpp>

namespace CoGaDB {
  /*! \brief Representation of a set of rows stored as bytes */
  class RowPage {
   private:
    /*! \brief Represents the rwos as bytes */
    char* _rows;
    /*! \brief Holds the offsets for each row */
    std::vector<unsigned short> _row_offsets;
    /*! \brief Number of columns for this rows */
    unsigned short _col_count;
    /*! \brief Number of rows on this page */
    unsigned short _row_count;
    /*! \brief true, if the page can't hold a row  anymore */
    bool _is_full;
    /*! \brief The current data size of all rows (_rows) in bytes */
    unsigned short _current_data_size;
    /*! \brief Gets from a buffered page the value at 'index'. Also results the
     * size of the value in bytes */
    char* getValueFromPage(BufferObject*, int, unsigned short&);
    /*! \brief Gets from a buffered page the string at 'index'. Also results the
     * size of the string in bytes */
    char* getStringFromPage(BufferObject*, int, unsigned short&);

   public:
    RowPage(unsigned short);
    RowPage(const RowPage&);
    RowPage& operator=(const RowPage&);
    bool isFull() const;
    bool fillPage(std::vector<BufferObject*>, int);
    void print(const TableSchema&) const;
    unsigned short getMaxDataSize() const;
    unsigned short getDataSize() const;
    unsigned short count() const;
    char* createRow(std::vector<BufferObject*>, int, unsigned short&,
                    unsigned short&);
    char* getValue(unsigned int, unsigned int, unsigned short&) const;
    char* getData() const;
  };

  typedef shared_pointer_namespace::shared_ptr<RowPage> RowPagePtr;
}