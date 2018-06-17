
#ifndef __SQL_PARSETREE_HPP
#define __SQL_PARSETREE_HPP

#include <list>
#include <ostream>
#include <string>
#include <utility>

#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>

#include <query_processing/query_processor.hpp>

namespace CoGaDB {
  namespace SQL {
    namespace ParseTree {

      /*
       * Scalar types
       */
      typedef std::string String;
      typedef boost::any Integer;
      typedef CoGaDB::AggregationMethod AggregationFunction;

      /* base class for all Statements */
      class Statement {
       public:
        virtual ~Statement() {}

        virtual TablePtr execute(ClientPtr client) = 0;
        virtual void explain(std::ostream &os) {}
        virtual CoGaDB::query_processing::LogicalQueryPlanPtr
        getLogicalQueryPlan() {
          return CoGaDB::query_processing::LogicalQueryPlanPtr();
        }
      };
      typedef boost::shared_ptr<Statement> StatementPtr;

      typedef CoGaDB::AttributeType AttributeType;
      typedef CoGaDB::Attribut Attribute;
      typedef CoGaDB::TableSchema TableSchema;
      typedef CoGaDB::Table Table;
      typedef CoGaDB::TablePtr TablePtr;

      typedef query_processing::LogicalQueryPlan::TypedNodePtr TypedNodePtr;

      class CreateTable : public Statement {
        TablePtr table;

       public:
        CreateTable(TablePtr _table) : table(_table) {}
        CreateTable(Table *_table) : table(_table) {}

        TablePtr execute(ClientPtr client);
      };

      struct ScalarExpression {
        virtual ~ScalarExpression() {}

        virtual void setColumnName(const String &name) = 0;
        virtual String getColumnName() const = 0;
        virtual String getQueryPlan(TypedNodePtr &io_node) = 0;

       protected:
        typedef std::map<std::string, std::string> RenameMap;
        RenameMap rename_map_;
      };
      typedef boost::shared_ptr<ScalarExpression> ScalarExpressionPtr;
      typedef std::list<ScalarExpressionPtr> ScalarExpressionList;
      typedef boost::shared_ptr<ScalarExpressionList> ScalarExpressionListPtr;

      struct AtomExpression : public ScalarExpression {
        boost::any atom;
        AttributeType type;

        String explicit_result_colname;

        AtomExpression(const boost::any &_atom, AttributeType _type)
            : ScalarExpression(), atom(_atom), type(_type) {}

        void setColumnName(const String &name);
        String toString();
        String getColumnName() const;
        String getQueryPlan(TypedNodePtr &io_node);
      };
      typedef boost::shared_ptr<AtomExpression> AtomExpressionPtr;

      struct ColumnExpression : public ScalarExpression {
        String column;

        ColumnExpression(String &_column);
        //        : ScalarExpression(), column(_column) {
        //
        //            std::string column_identifier=_column;
        ////            std::string qualified_column_name = _column;
        //            if(!isFullyQualifiedColumnIdentifier(column_identifier)){
        //                if(isPlainAttributeName(column_identifier)){
        //                    std::string table_name;
        //                    if(DataDictionary::instance().findUniqueTableNameForSimpleAttributeName(column_identifier,
        //                    table_name)){
        //                        column_identifier=table_name+std::string(".")+_column;
        ////                        column=column_identifier;
        //                    }
        //                }
        //                column = column_identifier;
        //                if(!convertColumnNameToFullyQualifiedName(column_identifier,
        //                    column)){
        //
        //                }
        //            }
        //            std::cout << "Column Expression: " << column << std::endl;
        ////             column=qualified_column_name;
        //        }
        void setColumnName(const String &name);
        String getColumnName() const;
        String getQueryPlan(TypedNodePtr &io_node);
      };
      typedef boost::shared_ptr<ColumnExpression> ColumnExpressionPtr;

      struct FunctionExpression : public ScalarExpression {
        AggregationFunction function;
        ScalarExpressionPtr param;

        FunctionExpression(AggregationFunction _function,
                           ScalarExpressionPtr _param)
            : ScalarExpression(),
              function(_function),
              param(_param),
              new_name(_param->getColumnName()) {}
        FunctionExpression(AggregationFunction _function,
                           ScalarExpressionPtr _param, std::string _new_name)
            : ScalarExpression(),
              function(_function),
              param(_param),
              new_name(_new_name) {}

        void setColumnName(const String &name);
        String getRenameColumnName();
        String getColumnName() const;
        String getQueryPlan(TypedNodePtr &io_node);
        std::string new_name;
      };
      typedef boost::shared_ptr<FunctionExpression> FunctionExpressionPtr;

      typedef CoGaDB::ColumnAlgebraOperation ColumnAlgebraOperation;

      struct AlgebraExpression : public ScalarExpression {
        ScalarExpressionPtr lvalue;
        ColumnAlgebraOperation op;
        ScalarExpressionPtr rvalue;

        String explicit_result_colname;

        AlgebraExpression(ScalarExpressionPtr _lvalue,
                          ColumnAlgebraOperation _op,
                          ScalarExpressionPtr _rvalue)
            : ScalarExpression(), lvalue(_lvalue), op(_op), rvalue(_rvalue) {}

        void setColumnName(const String &name);
        String getColumnName() const;
        String getQueryPlan(TypedNodePtr &io_node);
      };

      struct SearchCondition {
        virtual ~SearchCondition() {}

        virtual KNF_Selection_Expression getCNF(TypedNodePtr &io_node) = 0;
      };
      typedef boost::shared_ptr<SearchCondition> SearchConditionPtr;

      struct AndCondition : public SearchCondition {
        SearchConditionPtr lvalue;
        SearchConditionPtr rvalue;

        AndCondition(SearchConditionPtr _lvalue, SearchConditionPtr _rvalue)
            : lvalue(_lvalue), rvalue(_rvalue) {}

        KNF_Selection_Expression getCNF(TypedNodePtr &io_node);
      };
      typedef boost::shared_ptr<AndCondition> AndConditionPtr;

      struct OrCondition : public SearchCondition {
        SearchConditionPtr lvalue;
        SearchConditionPtr rvalue;

        OrCondition(SearchConditionPtr _lvalue, SearchConditionPtr _rvalue)
            : lvalue(_lvalue), rvalue(_rvalue) {}

        KNF_Selection_Expression getCNF(TypedNodePtr &io_node);
      };
      typedef boost::shared_ptr<OrCondition> OrConditionPtr;

      /*!
       * \brief Operators supported by ComparisonPredicate
       *
       * It is not merely a typedef to CoGaDB::ValueComparator,
       * so we can implement a broader set of operators on top of CoGaDB's
       * very limited set.
       */
      enum ValueComparator {
        LESSER,
        LESSER_EQUAL,
        GREATER,
        GREATER_EQUAL,
        EQUAL,
        UNEQUAL
      };

      struct Predicate : public SearchCondition {};

      struct ComparisonPredicate : public Predicate {
        ScalarExpressionPtr lvalue;
        ValueComparator op;
        ScalarExpressionPtr rvalue;

        ComparisonPredicate(ScalarExpressionPtr _lvalue, ValueComparator _op,
                            ScalarExpressionPtr _rvalue)
            : lvalue(_lvalue), op(_op), rvalue(_rvalue) {}

        KNF_Selection_Expression getCNF(TypedNodePtr &io_node);
      };
      typedef boost::shared_ptr<ComparisonPredicate> ComparisonPredicatePtr;

      struct BetweenPredicate : public Predicate {
        AndConditionPtr and_cond;

        BetweenPredicate(ScalarExpressionPtr _exp, ScalarExpressionPtr _lvalue,
                         ScalarExpressionPtr _rvalue);

        KNF_Selection_Expression getCNF(TypedNodePtr &io_node);
      };

      struct NotBetweenPredicate : public Predicate {
        OrConditionPtr or_cond;

        NotBetweenPredicate(ScalarExpressionPtr _exp,
                            ScalarExpressionPtr _lvalue,
                            ScalarExpressionPtr _rvalue);

        KNF_Selection_Expression getCNF(TypedNodePtr &io_node);
      };

      struct LikePredicate : public Predicate {
        ScalarExpressionPtr value;
        CoGaDB::ValueComparator op;
        AtomExpressionPtr atom;

        LikePredicate(ScalarExpressionPtr _lvalue, CoGaDB::ValueComparator _op,
                      AtomExpressionPtr _atom)
            : value(_lvalue), op(_op), atom(_atom) {}

        KNF_Selection_Expression getCNF(TypedNodePtr &io_node);
      };

      struct TableReference {
        virtual ~TableReference() {}

        virtual TypedNodePtr getQueryPlan() = 0;
      };
      typedef boost::shared_ptr<TableReference> TableReferencePtr;
      typedef std::list<TableReferencePtr> TableReferenceList;

      struct TableName : public TableReference {
        String table;

        TableName(String &_table) : table(_table) {}

        TypedNodePtr getQueryPlan();
      };

      class SelectFrom;
      typedef boost::shared_ptr<SelectFrom> SelectFromPtr;

      struct SubQueryResult : public TableReference {
        SelectFromPtr sub_query;

        SubQueryResult(SelectFromPtr _sub_query) : sub_query(_sub_query) {}

        TypedNodePtr getQueryPlan();
      };

      typedef boost::shared_ptr<TableName> TableNamePtr;

      struct Join : public TableReference {
        TableReferencePtr lvalue;
        TableReferencePtr rvalue;

        Join(TableReferencePtr _lvalue, TableReferencePtr _rvalue)
            : lvalue(_lvalue), rvalue(_rvalue) {}
      };

      struct InnerJoin : public Join {
        SearchConditionPtr condition;

        InnerJoin(TableReferencePtr _lvalue, TableReferencePtr _rvalue,
                  SearchConditionPtr _condition)
            : Join(_lvalue, _rvalue), condition(_condition) {}

        TypedNodePtr getQueryPlan();
      };
      typedef boost::shared_ptr<InnerJoin> InnerJoinPtr;

      struct CrossJoin : public Join {
        CrossJoin(TableReferencePtr _lvalue, TableReferencePtr _rvalue)
            : Join(_lvalue, _rvalue) {}

        TypedNodePtr getQueryPlan();
      };
      typedef boost::shared_ptr<CrossJoin> CrossJoinPtr;

      typedef std::list<String> ColumnList;
      typedef boost::shared_ptr<ColumnList> ColumnListPtr;
      typedef CoGaDB::Tuple Tuple;

      typedef CoGaDB::SortOrder SortOrder;

      struct OrderBy {
        SortAttributeList order;
      };
      typedef boost::shared_ptr<OrderBy> OrderByPtr;

      struct Limit {
        size_t num_rows;
        Limit(size_t _num_rows) : num_rows(_num_rows) {}
      };
      typedef boost::shared_ptr<Limit> LimitPtr;

      class InsertInto : public Statement {
        String table;
        Tuple row;

       public:
        InsertInto(TableName &_table, Tuple &_row)
            : table(_table.table), row(_row) {}

        TablePtr execute(ClientPtr client);
      };

      struct TableExpression {
        TypedNodePtr table;
        ColumnListPtr group_by;
        SearchConditionPtr having;
        OrderByPtr order_by;
        LimitPtr limit;

        TableExpression(TableReferenceList &from, SearchConditionPtr where,
                        ColumnListPtr _group_by, OrderByPtr _order_by,
                        SearchConditionPtr having = SearchConditionPtr(),
                        LimitPtr limit = LimitPtr());
      };

      class SelectFrom : public Statement {
        TypedNodePtr node;

       public:
        SelectFrom(ScalarExpressionListPtr columns, TableExpression &table_exp);

        TablePtr execute(ClientPtr client);
        void explain(std::ostream &os);
        virtual CoGaDB::query_processing::LogicalQueryPlanPtr
        getLogicalQueryPlan();
        TypedNodePtr getQueryPlan();
      };

      class Sequence {
        std::list<StatementPtr> statements;

       public:
        inline void push_back(StatementPtr statement) {
          statements.push_back(statement);
        }
        inline void push_back(Statement *statement) {
          push_back(StatementPtr(statement));
        }

        TablePtr execute(ClientPtr client);
        void explain(std::ostream &os = std::cerr);
        std::list<CoGaDB::query_processing::LogicalQueryPlanPtr>
        getLogicalQueryPlans();
      };
      typedef boost::shared_ptr<Sequence> SequencePtr;

    } /* namespace ParseTree */
  }   /* namespace SQL */
} /* namespace CoGaDB */

#endif
