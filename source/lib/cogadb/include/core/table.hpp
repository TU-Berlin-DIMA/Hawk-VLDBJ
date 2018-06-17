#pragma once
#include <core/base_table.hpp>
#include <core/global_definitions.hpp>

#include <boost/thread/mutex.hpp>

namespace CoGaDB {

  struct AccessStatistics {
    AccessStatistics() : number_of_accesses(0), last_access_timestamp(0) {}
    uint64_t number_of_accesses;
    uint64_t last_access_timestamp;
  };

  struct ColumnAccessStatistics {
    ColumnAccessStatistics(const std::string& _table_name,
                           const std::string& _column_name,
                           const AccessStatistics& _statistics)
        : table_name(_table_name),
          column_name(_column_name),
          statistics(_statistics) {}
    std::string table_name;
    std::string column_name;
    AccessStatistics statistics;
  };

  typedef std::vector<ColumnAccessStatistics> ColumnAccessStatisticsVector;

  class Table : public BaseTable {
   public:
    // typedef boost::shared_ptr<Table> TablePtr;
    /***************** constructors and destructor *****************/
    Table(const std::string& name, const TableSchema& schema);

    Table(const std::string& name, const TableSchema& schema,
          const CompressionSpecifications& compression_specifications);

    Table(const std::string& name, const std::string& path_to_database,
          bool& error_occured);

    Table(const std::string& name,
          const std::vector<ColumnPtr>& columns);  // if we already have
                                                   // columns, we can use this
                                                   // constructor to pass them
                                                   // in the table without any
                                                   // copy effort
    virtual ~Table();
    /***************** utility functions *****************/
    //	const std::string& getName() const throw();

    //	const TableSchema getSchema() const throw();

    static const TableSchema expandToQualifiedColumnNamesIfRequired(
        const TableSchema& schema, const std::string& table_name);

    virtual void print();
    /*! tries to store table in database*/
    bool store(const std::string& path_to_table_dir);
    /*! tries to load table form database*/
    bool load(TableLoaderMode loader_mode);
    bool loadDatafromFile(std::string filepath, bool quiet = false);

    virtual const TablePtr materialize() const;

    virtual bool addColumn(ColumnPtr);

    /***************** status report *****************/
    size_t getNumberofRows() const throw();

    // FIXME when we directly insert data into columns of tables,
    // we bypass the mechanism to count inserted rows, so we need the
    // ability to do so
    void setNumberofRows(size_t _number_of_rows);
    //	unsigned int getSizeinBytes() const throw(){
    //		const std::vector<ColumnPtr>& columns = this->getColumns();
    //		unsigned int size_in_bytes=0;
    //		std::vector<ColumnPtr>::const_iterator cit;
    //		for(cit=columns.begin();cit!=columns.end();++cit){
    //			size_in_bytes+=(*cit)->size();
    //		}
    //		return size_in_bytes;
    //	}
    bool isMaterialized() const throw();

    /***************** relational operations *****************/
    //	virtual const TablePtr selection(const std::string& column_name, const
    // boost::any& value_for_comparison, const ValueComparator& comp, const
    // ComputeDevice& comp_dev) const;// = 0;

    //	virtual const TablePtr projection(const std::list<std::string>&
    // columns_to_select, const ComputeDevice comp_dev) const;// = 0;

    //	virtual const TablePtr join(TablePtr table, const std::string&
    // join_column_table1, const std::string& join_column_table2, const
    // ComputeDevice comp_dev) const;// = 0;

    //	virtual const TablePtr sort(const std::string& column_name, SortOrder
    // order, ComputeDevice comp_dev) const;// = 0;

    //	virtual const TablePtr groupby(const std::string& grouping_column, const
    // std::string& aggregation_column,  AggregationMethod agg_meth=SUM,
    // ComputeDevice comp_dev=CPU) const;// = 0;
    /***************** read and write operations at table level
     * *****************/
    const Tuple fetchTuple(const TID& id) const;

    bool insert(const Tuple& t);

    bool update(const std::string& attribute_name, const boost::any& value);

    bool remove(const std::string& attribute_name, const boost::any& value);

    virtual bool append(TablePtr table);

    bool replaceColumn(const std::string& column_name,
                       const ColumnPtr new_column);

    const ColumnPtr getColumnbyName(const std::string& column_name) const
        throw();
    const ColumnPtr getColumnbyId(const unsigned int id) const throw();
    unsigned int getColumnIdbyColumnName(const std::string& column_name) const
        throw();

    const std::vector<ColumnProperties> getPropertiesOfColumns() const;

    bool copyColumnsInMainMemory();

    ColumnAccessStatisticsVector getColumnAccessStatistics() const;

   protected:
    virtual const std::vector<ColumnPtr>& getColumns() const;

    /***************** relational operations that return lookup tables
     * *****************/
    //	const std::vector<TID> lookup_selection(const std::string& column_name,
    // const boost::any& value_for_comparison, const ValueComparator& comp,
    // const
    // ComputeDevice& comp_dev) const;

    //	//const TablePtr projection(const std::list<std::string>&
    // columns_to_select, const ComputeDevice comp_dev) const;

    //	const std::vector<TID_Pair> lookup_join(TablePtr table, const
    // std::string& join_column_table1, const std::string& join_column_table2,
    // const ComputeDevice comp_dev) const;

    //	const TablePtr lookup_sort(const std::string& column_name, SortOrder
    // order, ComputeDevice comp_dev) const;

    //	//const TablePtr groupby(const std::string& grouping_column, const
    // std::string& aggregation_column,  AggregationMethod agg_meth=SUM,
    // ComputeDevice comp_dev=CPU) const;

    std::vector<ColumnPtr> columns_;
    std::string name_of_primary_key_column_;
    std::vector<ForeignKeyConstraint> foreign_key_constraints_;
    size_t number_of_rows_;
    mutable boost::mutex table_mutex_;
    typedef boost::shared_ptr<boost::mutex> MutexPtr;
    /* Protects column to be loaded from or stored to disk concurrently
     * from different threads. */
    mutable std::vector<MutexPtr> column_mutexes_;
    typedef std::map<std::string, AccessStatistics> AccessStatisticsMap;
    mutable AccessStatisticsMap access_statistics_map_;
    mutable boost::mutex access_statistics_map_mutex_;
    std::string path_to_database_;
    //	std::string name_;
    //	TableSchema schema_;
  };

  typedef shared_pointer_namespace::shared_ptr<Table> MaterializedTablePtr;

  const TableSchema mergeTableSchemas(
      const TableSchema& schema1, const std::string& join_attributname_table1,
      const TableSchema& schema2, const std::string& join_attributname_table2);
  // typedef Table::TablePtr TablePtr;

}  // end namespace CogaDB
