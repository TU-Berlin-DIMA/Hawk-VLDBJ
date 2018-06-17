
#pragma once
#include <core/base_column.hpp>
#include <core/global_definitions.hpp>
#include <core/hash_table.hpp>
#include <core/selection_expression.hpp>
#include <hype.hpp>

#include "attribute_reference.hpp"

namespace CoGaDB {

  class LookupColumn;

  class BlockIterator;

  struct ColumnProperties {
    std::string name;
    AttributeType attribute_type;
    ColumnType column_type;
    bool is_in_main_memory;
    size_t size_in_main_memory;
    size_t number_of_rows;
    uint64_t number_of_accesses;
    uint64_t last_access_timestamp;
    bool statistics_up_to_date;
    bool is_sorted_ascending;
    bool is_sorted_descending;
    bool is_dense_value_array_starting_with_zero;
  };

  class BaseTable {
    friend class LookupColumn;

   public:
    typedef shared_pointer_namespace::shared_ptr<BaseTable> TablePtr;
    /***************** constructors and destructor *****************/
    BaseTable(const std::string& name, const TableSchema& schema);

    // BaseTable(const std::string& name, const std::vector<ColumnPtr>&
    // columns); //if we already have columns, we can use this constructor to
    // pass them in the BaseTable without any copy effort
    virtual ~BaseTable();
    /***************** utility functions *****************/
    const std::string& getName() const throw();
    void setName(const std::string&) throw();

    const TableSchema getSchema() const throw();
    static const TablePtr getNext(TablePtr table, BlockIteratorPtr it,
                                  const ProcessorSpecification& proc_spec);
    std::string toString();
    std::string toString(const std::string& result_output_format,
                         bool include_header = true);

    virtual void print() = 0;
    /*! tries to store BaseTable in database*/
    virtual bool store(const std::string& path_to_table_dir) = 0;
    /*! tries to load BaseTable form database*/
    virtual bool load(TableLoaderMode loader_mode) = 0;
    virtual bool loadDatafromFile(std::string filepath, bool quiet = false) = 0;

    virtual const TablePtr materialize() const = 0;

    virtual bool addColumn(ColumnPtr) = 0;
    bool addHashTable(const std::string& column_name, HashTablePtr hash_table);
    // create result table for unary operation
    static const TablePtr createResultTable(
        TablePtr table, PositionListPtr tids, MaterializationStatus mat_stat,
        const std::string& operation_name,
        const ProcessorSpecification& proc_spec);
    // create result table for binary operation
    static const TablePtr createResultTable(
        TablePtr table1, const std::string& join_column_table1, TablePtr table2,
        const std::string& join_column_table2, PositionListPairPtr join_tids,
        TableSchema result_schema, MaterializationStatus mat_stat,
        const std::string& operation_name,
        const ProcessorSpecification& proc_spec);
    /***************** status report *****************/
    virtual size_t getNumberofRows() const throw();

    // FIXME when we directly insert data into columns of tables,
    // we bypass the mechanism to count inserted rows, so we need the
    // ability to do so
    virtual void setNumberofRows(size_t _number_of_rows);

    size_t getSizeinBytes() const throw();

    void printSchema() const {
      TableSchema schema = getSchema();
      TableSchema::iterator it;
      for (it = schema.begin(); it != schema.end(); ++it) {
        if (it != schema.begin()) std::cout << ", ";
        std::cout << it->second;
      }
      std::cout << std::endl;
    }

    virtual bool isMaterialized() const throw() = 0;

    static bool approximatelyEquals(TablePtr reference, TablePtr candidate);

    static bool equals(TablePtr, TablePtr);

    /***************** relational operations *****************/
    static const TablePtr selection(TablePtr table,
                                    const std::string& column_name,
                                    const SelectionParam& param);

    static const TablePtr selection(
        TablePtr table, const KNF_Selection_Expression&,
        MaterializationStatus mat_stat = MATERIALIZE,
        ParallelizationMode comp_mode = SERIAL);

    static const TablePtr selection(TablePtr table,
                                    const Disjunction& disjunction,
                                    const ProcessorSpecification& proc_spec,
                                    const hype::DeviceConstraint&);

    static const TablePtr projection(
        TablePtr table, const std::list<std::string>& columns_to_select,
        MaterializationStatus mat_stat = MATERIALIZE,
        const ComputeDevice comp_dev = CPU);

    static const TablePtr join(TablePtr table1,
                               const std::string& join_column_table1,
                               TablePtr table2,
                               const std::string& join_column_table2,
                               const JoinParam&);
    //											 JoinAlgorithm
    // join_alg=SORT_MERGE_JOIN,
    // MaterializationStatus mat_stat=LOOKUP, const ComputeDevice comp_dev=CPU);

    static const TablePtr semi_join(TablePtr table1,
                                    const std::string& join_column_table1,
                                    TablePtr table2,
                                    const std::string& join_column_table2,
                                    const JoinParam&);

    static const TablePtr pk_fk_join(TablePtr table1,
                                     const std::string& join_column_table1,
                                     TablePtr table2,
                                     const std::string& join_column_table2,
                                     JoinAlgorithm join_alg = SORT_MERGE_JOIN,
                                     MaterializationStatus mat_stat = LOOKUP,
                                     const ComputeDevice comp_dev = CPU);

    static const TablePtr fetch_join(TablePtr table1,
                                     const std::string& join_column_table1,
                                     TablePtr table2,
                                     const std::string& join_column_table2,
                                     const FetchJoinParam& param);

    static const TablePtr crossjoin(
        TablePtr table1, TablePtr table2,
        MaterializationStatus mat_stat = MATERIALIZE);

    // crossjoin(this->getInputDataLeftChild(), this->getInputDataRightChild(),
    // mat_stat_);

    static const TablePtr sort(TablePtr table, const std::string& column_name,
                               SortOrder order = ASCENDING,
                               MaterializationStatus mat_stat = MATERIALIZE,
                               ComputeDevice comp_dev = CPU);
    // assumes stable sort!
    static const TablePtr sort(TablePtr table,
                               const std::list<std::string>& column_names,
                               SortOrder order = ASCENDING,
                               MaterializationStatus mat_stat = MATERIALIZE,
                               ComputeDevice comp_dev = CPU);

    static const TablePtr sort(TablePtr table,
                               const std::list<SortAttribute>& sort_attributes,
                               MaterializationStatus mat_stat = MATERIALIZE,
                               ComputeDevice comp_dev = CPU);

    static const TablePtr groupby(TablePtr table,
                                  const std::string& grouping_column,
                                  const std::string& aggregation_column,
                                  const std::string& result_column_name,
                                  AggregationMethod agg_meth = SUM,
                                  ComputeDevice comp_dev = CPU);

    //	static const TablePtr groupby(TablePtr table, const
    // std::list<std::string>& grouping_columns,
    // std::list<std::pair<std::string,AggregationMethod> >
    // aggregation_functions, ComputeDevice comp_dev=CPU);
    static const TablePtr groupby(TablePtr table, const GroupbyParam& param);
    /***************** Aggregation Functions *****************/
    /*! \brief adds a new column named result_col_name to the table, which is
     * the result of col_name <operation> value*/
    static TablePtr ColumnConstantOperation(TablePtr tab,
                                            const std::string& col_name,
                                            const boost::any& value,
                                            const std::string& result_col_name,
                                            const AlgebraOperationParam& param);
    /*! \brief adds a new column named result_col_name to the table, which is
     * the result of col1_name <operation> col2_name*/
    static TablePtr ColumnAlgebraOperation(TablePtr tab,
                                           const std::string& col1_name,
                                           const std::string& col2_name,
                                           const std::string& result_col_name,
                                           const AlgebraOperationParam& param);
    /*! \brief fills a Column #rows times with value and append to table */
    static TablePtr AddConstantValueColumnOperation(
        TablePtr tab, const std::string& col_name, AttributeType type,
        const boost::any& value, const ProcessorSpecification& proc_spec);

    static const TablePtr user_defined_function(
        TablePtr table, const std::string& function_name,
        const std::vector<boost::any>& function_parameters,
        const ProcessorSpecification& proc_spec);

    /***************** read and write operations at BaseTable level
     * *****************/
    virtual const Tuple fetchTuple(const TID& id) const = 0;

    virtual bool insert(const Tuple& t) = 0;

    virtual bool update(const std::string& attribute_name,
                        const boost::any& value) = 0;

    virtual bool remove(const std::string& attribute_name,
                        const boost::any& value) = 0;

    virtual bool append(TablePtr table) = 0;

    bool hasColumn(const std::string& column_name);
    bool hasColumn(const AttributeReference& column_name);

    virtual bool replaceColumn(const std::string& column_name,
                               const ColumnPtr new_column) = 0;

    bool setPrimaryKeyConstraint(const std::string& column_name);
    bool hasPrimaryKeyConstraint(const std::string& column_name) const throw();

    bool hasForeignKeyConstraint(const std::string& column_name) const throw();
    //        bool setForeignKeyConstraint(const std::string& column_name, const
    //        ForeignKeyConstraint& prim_foreign_key_reference);
    bool setForeignKeyConstraint(const std::string& foreign_key_column_name,
                                 const std::string& primary_key_column_name,
                                 const std::string& primary_key_table_name);
    const ForeignKeyConstraint* getForeignKeyConstraint(
        const std::string& column_name);
    std::vector<const ForeignKeyConstraint*> getForeignKeyConstraints();

    virtual const ColumnPtr getColumnbyName(
        const std::string& column_name) const throw() = 0;
    virtual const ColumnPtr getColumnbyId(const unsigned int id) const
        throw() = 0;
    virtual unsigned int getColumnIdbyColumnName(
        const std::string& column_name) const throw() = 0;
    const HashTablePtr getHashTablebyName(const std::string& column_name) const
        throw();

    CompressionSpecifications getCompressionSpecifications();
    virtual const std::vector<ColumnProperties> getPropertiesOfColumns() const;
    bool renameColumns(const RenameList& rename_list);

    virtual bool copyColumnsInMainMemory() = 0;

   protected:
    virtual const std::vector<ColumnPtr>& getColumns() const = 0;
    // const TablePtr groupby(const std::string& grouping_column, const
    // std::string& aggregation_column,  AggregationMethod agg_meth=SUM,
    // ComputeDevice comp_dev=CPU) const;

    // std::vector<ColumnPtr> columns_;
    std::string name_;
    TableSchema schema_;
    typedef std::map<std::string, HashTablePtr> HashTables;
    HashTables hash_tables_;
  };

  typedef BaseTable::TablePtr TablePtr;

  bool isSameTableSchema(const TableSchema& schema,
                         const TableSchema& candidate);

  void renameFullyQualifiedNamesToUnqualifiedNames(TablePtr table);
  void expandUnqualifiedColumnNamesToQualifiedColumnNames(TablePtr table);

}  // end namespace CogaDB
