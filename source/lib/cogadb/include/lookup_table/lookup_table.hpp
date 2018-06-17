#pragma once
#include <core/base_table.hpp>
#include <lookup_table/lookup_column.hpp>
//#include <core/lookup_array.hpp>

namespace CoGaDB {

  typedef std::vector<LookupColumnPtr> LookupColumnVector;
  typedef shared_pointer_namespace::shared_ptr<LookupColumnVector>
      LookupColumnVectorPtr;

  // typedef std::pair<TableSchema,TablePtr> LookupTableSchema;

  class LookupTable : public BaseTable {
   public:
    typedef shared_pointer_namespace::shared_ptr<LookupTable> LookupTablePtr;
    /***************** constructors and destructor *****************/
    //	LookupTable(const std::string& name, const TableSchema& schema);

    LookupTable(const std::string& name, const TableSchema& schema,
                const std::vector<LookupColumnPtr>& lookup_columns,
                const std::vector<ColumnPtr> lookup_arrays_,
                const std::vector<ColumnPtr> dense_value_arrays =
                    std::vector<ColumnPtr>());

    //	LookupTable(const std::string& name, const std::vector<ColumnPtr>&
    // columns); //if we already have columns, we can use this constructor to
    // pass them in the table without any copy effort
    virtual ~LookupTable();
    /***************** utility functions *****************/
    //	const std::string& getName() const throw();

    //	const TableSchema getSchema() const throw();

    virtual void print();
    /*! tries to store table in database*/
    virtual bool store(const std::string& path_to_table_dir);
    /*! tries to load table form database*/
    virtual bool load(TableLoaderMode loader_mode);
    virtual bool loadDatafromFile(std::string filepath, bool quiet = false);

    virtual const TablePtr materialize() const;

    virtual bool addColumn(ColumnPtr);
    /************** Lookup Table Algebra *************/

    static const LookupTablePtr aggregate(
        const std::string& result_lookup_table_name,
        const LookupTable& lookup_table, const LookupColumn& lookup_col,
        const ProcessorSpecification& proc_spec);
    static const LookupTablePtr concatenate(
        const std::string& result_lookup_table_name,
        const LookupTable& lookup_table1, const LookupTable& lookup_table2);

    /***************** status report *****************/
    //	const unsigned int getNumberofRows() const throw();

    virtual bool isMaterialized() const throw();

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
    virtual const Tuple fetchTuple(const TID& id) const;

    virtual bool insert(const Tuple& t);

    virtual bool update(const std::string& attribute_name,
                        const boost::any& value);

    virtual bool remove(const std::string& attribute_name,
                        const boost::any& value);

    virtual bool append(TablePtr table);

    virtual bool replaceColumn(const std::string& column_name,
                               const ColumnPtr new_column);

    virtual const ColumnPtr getColumnbyName(
        const std::string& column_name) const throw();
    virtual const ColumnPtr getColumnbyId(const unsigned int id) const throw();
    virtual unsigned int getColumnIdbyColumnName(
        const std::string& column_name) const throw();

    const std::vector<LookupColumnPtr>& getLookupColumns() const;
    const ColumnVector& getDenseValueColumns();

    bool copyColumnsInMainMemory();

   protected:
    virtual const std::vector<ColumnPtr>& getColumns() const;
    const ColumnVectorPtr getLookupArrays();

   private:
    std::vector<LookupColumnPtr> lookup_columns_;           // Lookup Colums
    std::vector<ColumnPtr> lookup_arrays_to_real_columns_;  // each column from
                                                            // this list
                                                            // corresponds to
                                                            // the same entry in
                                                            // schema_
    std::vector<ColumnPtr> appended_dense_value_columns_;
    std::vector<ColumnPtr> all_columns_;
  };

  typedef LookupTable::LookupTablePtr LookupTablePtr;

  /* Utility functions*/
  const LookupTablePtr createLookupTableforUnaryOperation(
      const std::string& lookup_table_name, const TablePtr table,
      PositionListPtr ids, const ProcessorSpecification& proc_spec);

}  // end namespace CogaDB
