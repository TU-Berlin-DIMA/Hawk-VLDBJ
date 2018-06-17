#pragma once
#include <core/base_column.hpp>
#include <core/base_table.hpp>
#include <core/global_definitions.hpp>
#include <persistence/buffer_manager.hpp>
#include <persistence/row_page.hpp>
#include <util/time_measurement.hpp>

using namespace std;
namespace CoGaDB {

  template <typename T>
  class RowValueColumn;

  class RowTable : public BaseTable {
   public:
    typedef shared_pointer_namespace::shared_ptr<RowTable> RowTablePtr;
    RowTable(const string& name, const TableSchema& schema);
    virtual ~RowTable();

    bool update(PositionListPtr tids, const std::string& attribute_name,
                const boost::any& value);
    bool remove(PositionListPtr tids);

    /* *** Loading, Storing and Output *** */
    /*! \brief  For printing the table */
    virtual void print();
    /*! \brief  Stores the column in filesystem */
    virtual bool store(const std::string& path_to_table_dir);
    /*! \brief  Load Table from DB (implemented as loading from file at specific
     * path) */
    virtual bool load(TableLoaderMode loader_mode);
    /*! \brief  Load table from file (not implemented). Use load() */
    virtual bool loadDatafromFile(string filepath);

    virtual const TablePtr materialize() const; /*! Not implemented */

    /* *** Status *** */
    virtual size_t getNumberofRows() const throw();
    /*! \brief  Always true, becuase of early materialization at loading time */
    bool isMaterialized() const throw();

    /* *** Operations *** */
    /*! \brief  - Not implemented - Returns a tuple based on TID */
    virtual const Tuple fetchTuple(const TID& id) const;
    /*! \brief  - Not implemented - */
    virtual bool insert(const Tuple& t);
    /*! \brief  - Not implemented - */
    virtual bool update(const string& attribute_name, const boost::any& value);
    /*! \brief  - Not implemented - */
    virtual bool remove(const string& attribute_name, const boost::any& value);
    /*! \brief  - Not implemented - */
    virtual const ColumnPtr getColumnbyName(
        const std::string& column_name) const throw();
    /*! \brief Gets the RowPages for this table */
    const std::vector<RowPagePtr>& getRowPages() const;

    RowPagePtr getPageByIndex(unsigned int index, unsigned int& prefix);

   protected:
    virtual const std::vector<ColumnPtr>& getColumns() const;

   private:
    RowTablePtr _rtp;
    std::vector<RowPagePtr> _row_pages;
    std::vector<unsigned int> _row_pages_prefix;
    std::vector<unsigned int> _row_page_mapping;
    std::vector<ColumnPtr> _cols;
    unsigned int _row_count;
    /*! \brief  Always true, becuase of early materialization at loading time */
    bool _isMaterialized;
    void updateMapping(unsigned int count, unsigned int value);
  };
  typedef RowTable::RowTablePtr RowTablePtr;
}
