
#include <stdio.h>

#include <iostream>
#include <list>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <optimizer/optimizer.hpp>
#include <query_processing/query_processor.hpp>
#include <query_processing/query_processor.hpp>

#include <core/data_dictionary.hpp>
#include <core/global_definitions.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <sql/server/sql_driver.hpp>
#include <sql/server/sql_parsetree.hpp>
#include <util/query_processing.hpp>
#include <util/types.hpp>

using namespace CoGaDB;
using namespace query_processing;

namespace CoGaDB {
namespace SQL {
namespace ParseTree {

void checkColumnExistence(const std::string &cogadb_column) {
  if (!DataDictionary::instance().existColumn(cogadb_column)) {
    std::stringstream ss;
    ss << "Column '" << cogadb_column << "' not found!";
    throw Driver::ParseError(ss.str());
  }
}

void checkColumnExistence(const KNF_Selection_Expression &knf) {
  for (unsigned int i = 0; i < knf.disjunctions.size(); ++i) {
    for (unsigned int j = 0; j < knf.disjunctions[i].size(); ++j) {
      if (knf.disjunctions[i][j].getPredicateType() == ValueConstantPredicate) {
        std::string column_name = knf.disjunctions[i][j].getColumn1Name();
        checkColumnExistence(column_name);
      } else if (knf.disjunctions[i][j].getPredicateType() ==
                 ValueValuePredicate) {
        checkColumnExistence(knf.disjunctions[i][j].getColumn1Name());
        checkColumnExistence(knf.disjunctions[i][j].getColumn2Name());
      } else {
        COGADB_FATAL_ERROR("Invalid Predicate Type!", "");
      }
    }
  }
}

TablePtr CreateTable::execute(ClientPtr client) {
  getGlobalTableList().push_back(table);

  return table;
}

// void
// ScalarExpression::setColumnName(const String &name)
//{
//	std::cerr << "Setting name on column is not supported" << std::endl;
//	/* FIXME */
//
//}

void AtomExpression::setColumnName(const String &name) {
  /* set by AS clause */
  explicit_result_colname = name;
}

String AtomExpression::getColumnName() const { return explicit_result_colname; }

String AtomExpression::toString() {
  char buf[255];

  /*
   * Holy god of C++ idiosyncracies, forgive me!
   */
  switch (type) {
    case INT:
      snprintf(buf, sizeof(buf), "%d", boost::any_cast<int>(atom));
      return String(buf);

    case FLOAT:
      snprintf(buf, sizeof(buf), "%f", double(boost::any_cast<float>(atom)));
      return String(buf);
    case DOUBLE:
      snprintf(buf, sizeof(buf), "%f", boost::any_cast<double>(atom));
      return String(buf);

    case VARCHAR:
      return "'" + boost::any_cast<std::string>(atom) + "'";

    case BOOLEAN:
      return boost::any_cast<bool>(atom) ? String("TRUE") : String("FALSE");
    case UINT32:
      snprintf(buf, sizeof(buf), "%u", boost::any_cast<uint32_t>(atom));
      return String(buf);
    case OID: {
      std::stringstream str;
      str << boost::any_cast<uint64_t>(atom);
      return str.str();
    }
    case CHAR:
      return "'" + boost::any_cast<std::string>(atom) + "'";
    case DATE: {
      std::string result;
      CoGaDB::convertInternalDateTypeToString(boost::any_cast<uint32_t>(atom),
                                              result);
      return String(result.c_str());
    }
    default:
      COGADB_FATAL_ERROR("Not Implemented! For type " << type << "!", "");
  }

  /* not reached */
  return String("");
}

String AtomExpression::getQueryPlan(TypedNodePtr &io_node) {
  TypedNodePtr result;
  String result_col = explicit_result_colname;

  if (result_col.empty()) result_col = "(" + toString() + ")";

  result = boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
      result_col, type, atom,
      RuntimeConfiguration::instance().getGlobalDeviceConstraint());
  result->setLeft(io_node);

  io_node = result;
  return result_col;
}

std::string resolveUnqualifiedAttributeIfRequiredAndPossbile(
    const std::string &_column) {
  std::string column;
  std::string column_identifier = _column;
  if (!isFullyQualifiedColumnIdentifier(column_identifier)) {
    if (isPlainAttributeName(column_identifier)) {
      std::string table_name;
      if (DataDictionary::instance().findUniqueTableNameForSimpleAttributeName(
              column_identifier, table_name)) {
        column_identifier = table_name + std::string(".") + _column;
      }
    }
    column = column_identifier;
    if (!convertColumnNameToFullyQualifiedName(column_identifier, column)) {
    }
  }
  return column;
}

ColumnExpression::ColumnExpression(String &_column)
    : ScalarExpression(), column(_column) {
  column = resolveUnqualifiedAttributeIfRequiredAndPossbile(_column);
}

void ColumnExpression::setColumnName(const String &name) {
  this->rename_map_.insert(std::make_pair(column, name));
  column = name;
}

String ColumnExpression::getColumnName() const { return column; }

String ColumnExpression::getQueryPlan(TypedNodePtr &io_node) { return column; }

void FunctionExpression::setColumnName(const String &name) {
  return param->setColumnName(name);
}

String FunctionExpression::getColumnName() const {
  return param->getColumnName();
}

String FunctionExpression::getRenameColumnName() { return this->new_name; }

String FunctionExpression::getQueryPlan(TypedNodePtr &io_node) {
  // RenameList rename_list;
  // rename_list.push_back(RenameEntry(this->param->getColumnName(),
  // this->new_name));
  // TypedNodePtr result(new logical_operator::Logical_Rename(rename_list));

  String ret = param->getQueryPlan(io_node);
  // result->setLeft(io_node);
  // io_node=result;
  return ret;
  // return param->getQueryPlan(io_node);
}

void AlgebraExpression::setColumnName(const String &name) {
  /* set by AS clause */
  explicit_result_colname = name;
}

String AlgebraExpression::getColumnName() const {
  return explicit_result_colname;
}

String AlgebraExpression::getQueryPlan(TypedNodePtr &io_node) {
  /* FIXME: in C++ this might not be the perfect data structure:
     what if ColumnAlgebraOperation changes? */
  static const char *op2name[] = {/* [ADD] = */ "+",
                                  /* [SUB] = */ "-",
                                  /* [MUL] = */ "*",
                                  /* [DIV] = */ "/"};

  String result_col = explicit_result_colname;
  TypedNodePtr result;

  /*
   * TODO: if both operands are atomic, we do not
   * need to create two logical operators.
   * But this optimization has to be performed before
   * query plan generation (as an algebraic optimization)
   */
  if (dynamic_cast<AtomExpression *>(lvalue.get()) ||
      dynamic_cast<AtomExpression *>(rvalue.get())) {
    String col;
    AtomExpressionPtr value;

    /*
     * FIXME: fails for "5 - Col" as "-" and "/" are not commutative
     * easiest would be to treat them like a ColumnColumnOperator
     */
    if (dynamic_cast<AtomExpression *>(lvalue.get())) {
      value = boost::static_pointer_cast<AtomExpression>(lvalue);
      String l_col = lvalue->getQueryPlan(io_node);
      String r_col = rvalue->getQueryPlan(io_node);
      col = rvalue->getQueryPlan(io_node);
      if (result_col.empty())
        result_col = "(" + value->toString() + op2name[op] + col + ")";
      result =
          boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
              value->toString(), value->type, value->atom,
              RuntimeConfiguration::instance().getGlobalDeviceConstraint());
      result->setLeft(io_node);
      io_node = result;

      result =
          boost::make_shared<logical_operator::Logical_ColumnAlgebraOperator>(
              value->toString(), r_col, result_col, op);

    } else {
      value = boost::static_pointer_cast<AtomExpression>(rvalue);
      col = lvalue->getQueryPlan(io_node);
      if (result_col.empty())
        result_col = "(" + col + op2name[op] + value->toString() + ")";
      result =
          boost::make_shared<logical_operator::Logical_ColumnConstantOperator>(
              col, value->atom, result_col, op);
    }

  } else {
    String l_col = lvalue->getQueryPlan(io_node);
    String r_col = rvalue->getQueryPlan(io_node);

    if (result_col.empty())
      result_col = "(" + l_col + op2name[op] + r_col + ")";

    result =
        boost::make_shared<logical_operator::Logical_ColumnAlgebraOperator>(
            l_col, r_col, result_col, op);
  }

  result->setLeft(io_node);
  io_node = result;
  return result_col;
}

/*!
 * \brief Convert a parse tree \b AND condition into
 *        Conjunctive Normal Form
 *
 * To convert the \b AND condition of two search conditions,
 * conjunctions of their CNF simply have to be concatenated.
 *
 * \param io_node Reference to node pointer that holds the current
 *                chain of algebraic logical operators that is updated
 *                during CNF construction.
 * \return The CNF of the conjunction of the two search conditions
 */
KNF_Selection_Expression AndCondition::getCNF(TypedNodePtr &io_node) {
  KNF_Selection_Expression l_knf = lvalue->getCNF(io_node);
  KNF_Selection_Expression r_knf = rvalue->getCNF(io_node);

  l_knf.disjunctions.insert(l_knf.disjunctions.end(),
                            r_knf.disjunctions.begin(),
                            r_knf.disjunctions.end());

  return l_knf;
}

/*!
 * \brief Convert a parse tree \b OR condition into
 *        Conjunctive Normal Form
 *
 * In an \b OR condition the CNF of lvalue (<tt>lvalue->getCNF()</tt>) and
 * rvalue (<tt>rvalue->getCNF()</tt>) are of the following form:
 * \remark
 *   L1 \b AND L2 \b AND L3 ... \b AND Ln<br/>
 *   R1 \b AND R2 \b AND R3 ... \b AND Rm
 *
 * The disjunction of these terms is equivalent to:
 * \remark
 *   (L1 \b OR R1) \b AND (L1 \b OR R2) ... \b AND (L1 \b OR Rm) \b AND<br/>
 *   (L2 \b OR R1) \b AND (L2 \b OR R2) ... \b AND (L2 \b OR Rm) \b AND ...<br/>
 *   (Ln \b OR R1) \b AND (Ln \b OR R2) ... \b AND (Ln \b OR Rm)
 *
 * \param io_node Reference to node pointer that holds the current
 *                chain of algebraic logical operators that is updated
 *                during CNF construction.
 * \return The CNF of the disjunction of the two search conditions
 */
KNF_Selection_Expression OrCondition::getCNF(TypedNodePtr &io_node) {
  KNF_Selection_Expression l_knf = lvalue->getCNF(io_node);
  KNF_Selection_Expression r_knf = rvalue->getCNF(io_node);
  KNF_Selection_Expression result;

  result.disjunctions.reserve(l_knf.disjunctions.size() *
                              r_knf.disjunctions.size());

  for (auto n = 0u; n < l_knf.disjunctions.size(); n++) {
    Disjunction &l_disj = l_knf.disjunctions[n];

    for (auto m = 0u; m < r_knf.disjunctions.size(); m++) {
      Disjunction &r_disj = r_knf.disjunctions[m];
      Disjunction result_disj;

      result_disj.reserve(l_disj.size() + r_disj.size());

      result_disj.insert(result_disj.end(), l_disj.begin(), l_disj.end());
      result_disj.insert(result_disj.end(), r_disj.begin(), r_disj.end());

      result.disjunctions.push_back(result_disj);
    }
  }

  return result;
}

/*!
 * \brief Construct CoGaDB::Predicate
 *
 * It resolves boost::any to strings for ValueValuePredicates.
 *
 * \param column Name of column to compare
 * \param value Value or name of column to compare with
 * \param pred Predicate type
 * \param comp Comparison operator
 * \return a CoGaDB::Predicate
 */
static CoGaDB::Predicate make_pred(const std::string &column,
                                   const boost::any &value, PredicateType pred,
                                   CoGaDB::ValueComparator comp) {
  if (pred == ValueValuePredicate)
    return CoGaDB::Predicate(column, boost::any_cast<std::string>(value), pred,
                             comp);
  else {
    return CoGaDB::Predicate(column, value, pred, comp);
  }
}

/*!
 * \brief Get Conjunctive Normal Form of a search condition predicate
 *
 * Converting a single predicate to CNF is trivial - it is
 * simply wrapped in a disjunction that is wrapped in a conjunction.
 *
 * Since predicates in a CNF structure have their atomic type always
 * on the right side of the operator (as in Column > Atom),
 * the predicate operator might have to be flipped.
 *
 * \see CoGaDB::Predicate
 *
 * \param io_node Reference to node pointer that holds the current
 *                chain of algebraic logical operators that is updated
 *                during CNF construction.
 * \return CNF of parse tree predicate
 */
KNF_Selection_Expression ComparisonPredicate::getCNF(TypedNodePtr &io_node) {
  KNF_Selection_Expression knf;
  Disjunction disjunction;

  std::string cogadb_column;
  boost::any cogadb_value; /* may be atom or column name */
  PredicateType cogadb_type;
  ValueComparator normalized_op = op;

  if (!dynamic_cast<AtomExpression *>(lvalue.get()) &&
      !dynamic_cast<AtomExpression *>(rvalue.get())) {
    cogadb_column = lvalue->getQueryPlan(io_node);
    cogadb_value = boost::any(rvalue->getQueryPlan(io_node));
    cogadb_type = ValueValuePredicate;
  } else if (dynamic_cast<AtomExpression *>(rvalue.get())) {
    AtomExpressionPtr ratom(boost::static_pointer_cast<AtomExpression>(rvalue));

    cogadb_column = lvalue->getQueryPlan(io_node);
    cogadb_value = ratom->atom;
    cogadb_type = ValueConstantPredicate;
  } else if (dynamic_cast<AtomExpression *>(lvalue.get())) {
    AtomExpressionPtr latom(boost::static_pointer_cast<AtomExpression>(lvalue));

    cogadb_column = rvalue->getQueryPlan(io_node);
    cogadb_value = latom->atom;
    cogadb_type = ValueConstantPredicate;

    /* operands are switched, so reverse operator */
    switch (normalized_op) {
      case LESSER:
        normalized_op = GREATER;
        break;
      case LESSER_EQUAL:
        normalized_op = GREATER_EQUAL;
        break;
      case GREATER:
        normalized_op = LESSER;
        break;
      case GREATER_EQUAL:
        normalized_op = LESSER_EQUAL;
        break;
      case EQUAL:
      case UNEQUAL:
        /* nothing to do */
        break;
    }
  } else {
    COGADB_FATAL_ERROR("Got no left or right value!", "");
  }

  /*!
   * map SQL comparison operators to CoGaDB's limited set of operators.
   * \todo If we would save the data type in the AtomExpression we
   *       could optimize expressions like (x >= 3) to (x > 2), saving
   *       one predicate.
   */
  switch (normalized_op) {
    case LESSER_EQUAL:
      disjunction.push_back(make_pred(cogadb_column, cogadb_value, cogadb_type,
                                      CoGaDB::LESSER_EQUAL));
      break;
    case LESSER:
      disjunction.push_back(
          make_pred(cogadb_column, cogadb_value, cogadb_type, CoGaDB::LESSER));
      break;

    case GREATER_EQUAL:
      disjunction.push_back(make_pred(cogadb_column, cogadb_value, cogadb_type,
                                      CoGaDB::GREATER_EQUAL));
      break;
    case GREATER:
      disjunction.push_back(
          make_pred(cogadb_column, cogadb_value, cogadb_type, CoGaDB::GREATER));
      break;

    case EQUAL:
      disjunction.push_back(
          make_pred(cogadb_column, cogadb_value, cogadb_type, CoGaDB::EQUAL));
      break;

    case UNEQUAL:
      disjunction.push_back(
          make_pred(cogadb_column, cogadb_value, cogadb_type, CoGaDB::UNEQUAL));
      break;
  }

  knf.disjunctions.push_back(disjunction);
  return knf;
}

/*!
 * \brief Construct BETWEEN predicate
 *
 * Currently the BETWEEN predicate is implemented as a conjunction of
 * to predicates using parse tree nodes.
 * With CoGaDB's current set of supported predicate operators, this results in
 * a CNF like "(exp = val1 OR exp > val1) AND (exp = val2 OR exp < val2)".
 *
 * In the future, we might represent BETWEEN predicates as entirely distinct
 * nodes with custom implementations of getCNF().
 * This would allow us to minimize the amount of logical query plan operators
 * for algebraic operations in the predicate.
 *
 * \param _exp The value that should be checked
 * \param _lvalue The lower inclusive value
 * \param _rvalue The upper inclusive value
 * \returns A BETWEEN predicate node
 */
BetweenPredicate::BetweenPredicate(ScalarExpressionPtr _exp,
                                   ScalarExpressionPtr _lvalue,
                                   ScalarExpressionPtr _rvalue) {
  ComparisonPredicatePtr lpred =
      boost::make_shared<ComparisonPredicate>(_exp, GREATER_EQUAL, _lvalue);
  ComparisonPredicatePtr rpred =
      boost::make_shared<ComparisonPredicate>(_exp, LESSER_EQUAL, _rvalue);
  and_cond = boost::make_shared<AndCondition>(lpred, rpred);
}

/*!
 * \brief Calculate CNF of a BETWEEN predicate
 */
KNF_Selection_Expression BetweenPredicate::getCNF(TypedNodePtr &io_node) {
  return and_cond->getCNF(io_node);
}

/*!
 * \brief Construct NOT BETWEEN predicate
 *
 * \see BetweenPredicate
 */
NotBetweenPredicate::NotBetweenPredicate(ScalarExpressionPtr _exp,
                                         ScalarExpressionPtr _lvalue,
                                         ScalarExpressionPtr _rvalue) {
  ComparisonPredicatePtr lpred =
      boost::make_shared<ComparisonPredicate>(_exp, LESSER, _lvalue);
  ComparisonPredicatePtr rpred =
      boost::make_shared<ComparisonPredicate>(_exp, GREATER, _rvalue);
  or_cond = boost::make_shared<OrCondition>(lpred, rpred);
}

/*!
 * \brief Calculate CNF of a NOT BETWEEN predicate
 */
KNF_Selection_Expression NotBetweenPredicate::getCNF(TypedNodePtr &io_node) {
  return or_cond->getCNF(io_node);
}

KNF_Selection_Expression LikePredicate::getCNF(TypedNodePtr &io_node) {
  std::string cogadb_column;
  boost::any cogadb_value; /* may be atom or column name */
  PredicateType cogadb_type;
  //	ValueComparator normalized_op = op;

  cogadb_column = value->getQueryPlan(io_node);
  cogadb_value = this->atom->atom;
  cogadb_type = ValueRegularExpressionPredicate;

  //        assert(dynamic_cast<AtomExpression *>(this->value)!=NULL);

  KNF_Selection_Expression knf;
  Disjunction disjunction;
  disjunction.push_back(
      make_pred(cogadb_column, cogadb_value, cogadb_type, this->op));
  knf.disjunctions.push_back(disjunction);
  return knf;
}
void warn_if_database_empty(std::ostream &out) {
  const std::string path = RuntimeConfiguration::instance().getPathToDatabase();
  if (path == "" || path == "./data") {
    out << std::endl
        << "==================================================================="
           "========"
        << std::endl;
    out << "[INFO]: Your database is empty. Either you did not specify a "
        << "database or you have not loaded it." << std::endl
        << std::endl
        << "Please set variable 'path_to_database' to the absolute path of "
        << "the CoGaDB database you want to query. Then, type 'loaddatabase'."
        << std::endl
        << std::endl;
    out << "If you do not have created a cogadb database, you can do so by "
        << "performing the following steps: " << std::endl
        << "\t1. Get a dataset and create a schema. To try out CoGaDB, we "
        << "recommend the star schema benchmark "
           "(https://github.com/electrum/ssb-dbgen) "
        << "and the TPC-H benchmark (http://www.tpc.org/tpch/)." << std::endl
        << "\t2. Import the data. You can do so by using the command "
        << "'import_csv_file <TABLE_NAME> <PATH_TO_CSV_FILE>'. For "
           "convenience, "
        << "you can use 'create_ssb_database <PATH_TBL_FILES/>' and "
        << "'create_tpch_database <PATH_TBL_FILES/>' to create a schema and"
        << " import all tables of the star schema benchmark and the TPC-H "
           "benchmark, respectively."
        << std::endl
        << "\t3. Save your configuration in a config file 'startup.coga'. "
        << std::endl
        << "\tset path_to_database=<PATH_TO_DATABASE>" << std::endl
        << "\t#keep data on disk until it is accessed" << std::endl
        << "\tset table_loader_mode=disk" << std::endl
        << "\tloaddatabase" << std::endl
        << std::endl;
    out << "You can find more information in the CoGaDB user guide "
        << "(http://cogadb.cs.tu-dortmund.de/wordpress/download/). For an "
        << "overview of commands, please type 'help'." << std::endl
        << std::endl;
    out << "If you have questions, suggestions, or a bug to report, please "
        << "feel free to contact the CoGaDB developer team "
           "<cogadb@googlegroups.com>."
        << std::endl
        << "==================================================================="
           "========";
  }
}

TypedNodePtr TableName::getQueryPlan() {
  if (!getTablebyName(this->table)) {
    std::stringstream ss;
    ss << "Table '" << this->table << "' not found!";
    warn_if_database_empty(ss);
    // std::cout << "Error! Table " << this->table << " not found in database"
    // << std::endl;
    throw Driver::ParseError(ss.str());
    // return TypedNodePtr();
  }

  return boost::make_shared<logical_operator::Logical_Scan>(table);
}

TypedNodePtr SubQueryResult::getQueryPlan() {
  return this->sub_query->getQueryPlan();
}

TypedNodePtr InnerJoin::getQueryPlan() {
  TypedNodePtr join;
  std::string join_col1, join_col2;

  ComparisonPredicatePtr predicate(
      boost::dynamic_pointer_cast<ComparisonPredicate>(condition));

  if (predicate && predicate->op == EQUAL &&
      dynamic_cast<ColumnExpression *>(predicate->lvalue.get()) &&
      dynamic_cast<ColumnExpression *>(predicate->rvalue.get())) {
    ColumnExpressionPtr lvalue(
        boost::static_pointer_cast<ColumnExpression>(predicate->lvalue));
    ColumnExpressionPtr rvalue(
        boost::static_pointer_cast<ColumnExpression>(predicate->rvalue));

    join_col1 = lvalue->column;
    join_col2 = rvalue->column;
  } else {
    std::cerr << "Unsupported JOIN predicate!" << std::endl;
    /* FIXME */
  }

  //        checkColumnExistence(join_col1);
  //        checkColumnExistence(join_col2);
  join = boost::make_shared<logical_operator::Logical_Join>(
      join_col1, join_col2, INNER_JOIN,
      RuntimeConfiguration::instance().getGlobalDeviceConstraint());
  join->setLeft(lvalue->getQueryPlan());
  join->setRight(rvalue->getQueryPlan());

  return join;
}

TypedNodePtr CrossJoin::getQueryPlan() {
  TypedNodePtr join = boost::make_shared<logical_operator::Logical_CrossJoin>();
  join->setLeft(lvalue->getQueryPlan());
  join->setRight(rvalue->getQueryPlan());

  return join;
}

TablePtr InsertInto::execute(ClientPtr client) {
  TablePtr table_p = getTablebyName(table);
  if (!table_p) {
    std::stringstream ss;
    ss << "Table '" << this->table << "' not found!";
    // std::cout << "Error! Table " << this->table << " not found in database"
    // << std::endl;
    throw Driver::ParseError(ss.str());
  }
  table_p->insert(row);

  return table_p;
}

TableExpression::TableExpression(TableReferenceList &from,
                                 SearchConditionPtr where,
                                 ColumnListPtr _group_by, OrderByPtr _order_by,
                                 SearchConditionPtr _having, LimitPtr _limit)
    : group_by(_group_by), having(_having), order_by(_order_by), limit(_limit) {
  /*
   * Check whether all specified column names are valid
   */
  {
    //            ColumnList::iterator it;
    //            if(_group_by){
    //                for(it=_group_by->begin();it!=_group_by->end();++it){
    //                    checkColumnExistence(*it);
    //                }
    //            }
    //            if(_order_by){
    //                for(it=_order_by->list.begin();it!=_order_by->list.end();++it){
    //                    checkColumnExistence(*it);
    //                }
    //            }
  }

  /*
   * Compile list of table references into chained logical
   * cross joins
   */
  for (TableReferenceList::iterator it = from.begin(); it != from.end(); it++) {
    TypedNodePtr plan = (*it)->getQueryPlan();

    if (!table.get()) {
      /* first table reference */
      table = plan;
    } else {
      TypedNodePtr last_table = table;

      table = boost::make_shared<logical_operator::Logical_CrossJoin>();
      table->setLeft(last_table);
      table->setRight(plan);
    }
  }

  /*
   * FIXME: resolve selections and cross joins to
   * inner joins.
   * This can only be done generically at execution time
   * to decide which column referenced in the predicates
   * belongs to which table.
   */
  if (where.get()) {
    /*
     * FIXME: nodes added to io_node by getCNF() and in turn
     * ScalarExpression::getQueryPlan() must be projected
     * out again.
     * E.g.: SELECT * FROM table WHERE x*2 > 2;
     * This requires table introspection and keeping a list
     * of columns added by column operations to io_node.
     */
    TypedNodePtr io_node = table; /* joins */
    KNF_Selection_Expression knf = where->getCNF(io_node);
    //                checkColumnExistence(knf);

    table = boost::make_shared<logical_operator::Logical_ComplexSelection>(
        knf, LOOKUP,
        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint());
    table->setLeft(io_node);
  }
}

SelectFrom::SelectFrom(ScalarExpressionListPtr columns,
                       TableExpression &table_exp) {
  ColumnList proj_list;
  std::list<ColumnAggregation> aggr_list;

  node = table_exp.table;

  RenameList rename_list;
  // rename_list.push_back(RenameEntry(this->param->getColumnName(),
  // this->new_name));
  // TypedNodePtr result(new logical_operator::Logical_Rename(rename_list));
  size_t number_of_non_aggregation_columns = 0;
  std::set<std::string> non_computed_columns;

  if (columns) {
    /*
     * add nodes to io_node required for calculations (like algebraic
     * expressions) in the column list and aggregate the column names
     * necessary for projection (since they might be calculated as well).
     * ColumnAggregation tuples are calculated from function names
     * and the column names of their arguments.
     */

    for (ScalarExpressionList::iterator it = columns->begin();
         it != columns->end(); it++) {
      ScalarExpressionPtr cur = *it;
      String cur_colname = cur->getQueryPlan(node);

      proj_list.push_back(cur_colname);

      if (dynamic_cast<FunctionExpression *>(cur.get())) {
        FunctionExpressionPtr fnc(
            boost::static_pointer_cast<FunctionExpression>(cur));
        aggr_list.push_back(ColumnAggregation(
            cur_colname, Aggregate(fnc->function, fnc->getRenameColumnName())));
        rename_list.push_back(
            RenameEntry(cur_colname, fnc->getRenameColumnName()));
        //                                checkColumnExistence(cur_colname);
      } else {
        number_of_non_aggregation_columns++;
        //                            checkColumnExistence(cur_colname);
        non_computed_columns.insert(cur_colname);
      }
    }
  }

  if (table_exp.group_by) {
    TypedNodePtr sort;

    if (aggr_list.empty()) {
      std::stringstream ss;
      ss << "Invalid GROUP BY! At least one column_name must be specified!"
         << std::endl;
      throw Driver::ParseError(ss.str());
    }

    // error checking: in SQL queries with group by statement,
    // the groupby attributes have to appear in the selection clause and vice
    // versa
    std::list<std::string> grouping_columns;
    ColumnList::const_iterator group_col_cit;
    for (group_col_cit = table_exp.group_by->begin();
         group_col_cit != table_exp.group_by->end(); ++group_col_cit) {
      std::string name = *group_col_cit;
      name = resolveUnqualifiedAttributeIfRequiredAndPossbile(name);
      grouping_columns.push_back(name);
    }
    std::list<std::string> sorted_grouping_columns(grouping_columns);
    sorted_grouping_columns.sort();

    std::vector<std::string> intersect(
        std::max(sorted_grouping_columns.size(), non_computed_columns.size()));
    std::vector<std::string>::iterator it = std::set_intersection(
        sorted_grouping_columns.begin(), sorted_grouping_columns.end(),
        non_computed_columns.begin(), non_computed_columns.end(),
        intersect.begin());
    intersect.resize(it - intersect.begin());
    if (grouping_columns.size() != intersect.size() ||
        non_computed_columns.size() != intersect.size()) {
      // COGADB_ERROR("Grouping Columns do not match non computed columns in
      // selection clause!","");
      std::stringstream ss;
      ss << "Grouping Columns '";
      {
        std::list<std::string>::iterator it;
        for (it = grouping_columns.begin(); it != grouping_columns.end();
             ++it) {
          ss << *it;
          if (it != --grouping_columns.end()) ss << ",";
        }
      }
      ss << "' do not match non computed columns in SELECT clause '";
      {
        std::set<std::string>::iterator it;
        for (it = non_computed_columns.begin();
             it != non_computed_columns.end(); ++it) {
          ss << *it;
          if (it != --non_computed_columns.end()) ss << ",";
        }
      }
      ss << "'!";
      throw Driver::ParseError(ss.str());
    }

    TypedNodePtr group = boost::make_shared<logical_operator::Logical_Groupby>(
        grouping_columns, aggr_list);
    group->setLeft(node);
    node = group;

    /*
     * no additional projection necessary:
     * GroupBy already discared all non-grouping and non-aggregated columns
     */

  } else if (!aggr_list.empty()) {
    TypedNodePtr groupby =
        boost::make_shared<logical_operator::Logical_Groupby>(
            std::list<std::string>(), aggr_list);
    groupby->setLeft(node);
    node = groupby;
  }

  if (table_exp.having) {
    /* \todo TODO: Check that columns in having clause are computed by this
     * query*/
    TypedNodePtr io_node = node;
    KNF_Selection_Expression knf = table_exp.having->getCNF(io_node);

    std::cout << "Having clause found! Here are the computed columns:"
              << std::endl;
    RenameList::const_iterator cit;
    for (cit = rename_list.begin(); cit != rename_list.end(); ++cit) {
      std::cout << "Rename: " << cit->first << "->" << cit->second << std::endl;
    }

    boost::shared_ptr<logical_operator::Logical_ComplexSelection>
        having_selection =
            boost::make_shared<logical_operator::Logical_ComplexSelection>(
                knf, LOOKUP, CoGaDB::RuntimeConfiguration::instance()
                                 .getGlobalDeviceConstraint());
    // tell logical optimizer that it should not try to push down this selection
    // down
    having_selection->couldNotBePushedDownFurther(true);
    node = having_selection;
    node->setLeft(io_node);
  }

  /* add sort operator for order by clause*/
  if (table_exp.order_by) {
    TypedNodePtr sort;
    SortAttributeList sort_attributes;
    SortAttributeList::const_iterator cit;
    for (cit = table_exp.order_by->order.begin();
         cit != table_exp.order_by->order.end(); ++cit) {
      sort_attributes.push_back(SortAttribute(
          resolveUnqualifiedAttributeIfRequiredAndPossbile(cit->first),
          cit->second));
    }
    sort = boost::make_shared<logical_operator::Logical_Sort>(
        sort_attributes);  //, table_exp.order_by->order);
    sort->setLeft(node);
    node = sort;
  }

  /* add UDF operator to limit the number of result rows if neccessary */
  if (table_exp.limit) {
    TypedNodePtr limit;
    std::vector<boost::any> parameters;
    parameters.push_back(table_exp.limit->num_rows);
    limit = boost::make_shared<logical_operator::Logical_UDF>(
        "LIMIT", parameters,
        CoGaDB::RuntimeConfiguration::instance()
            .getGlobalDeviceConstraint());  //, table_exp.order_by->order);
    limit->setLeft(node);
    node = limit;
  }

  if (!proj_list.empty() && !table_exp.group_by &&
      number_of_non_aggregation_columns > 0) {
    TypedNodePtr source = node;

    node = boost::make_shared<logical_operator::Logical_Projection>(proj_list);
    node->setLeft(source);
  }

  //        TypedNodePtr source = node;
  //        node =
  //        boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  ////	node =
  /// boost::make_shared<logical_operator::Logical_Rename>(RenameList());
  //	node->setLeft(source);
}

TablePtr SelectFrom::execute(ClientPtr client) {
  if (!client) {
    return TablePtr();
  }

  std::ostream &out = client->getOutputStream();
  LogicalQueryPlanPtr log_plan(boost::make_shared<LogicalQueryPlan>(node));
  log_plan->setOutputStream(client->getOutputStream());

  if (!optimizer::checkQueryPlan(log_plan->getRoot())) {
    out << "Query Compilation Failed!" << std::endl;
    return TablePtr();
  }

  optimizer::Logical_Optimizer::instance().optimize(log_plan);
  return executeQueryPlan(log_plan, client);
}

void SelectFrom::explain(std::ostream &os) {
  LogicalQueryPlanPtr log_plan(boost::make_shared<LogicalQueryPlan>(node));

  if (!optimizer::checkQueryPlan(log_plan->getRoot())) {
    os << "Query Compilation Failed!" << std::endl;
    return;
  }

  optimizer::Logical_Optimizer::instance().optimize(log_plan);

  log_plan->print_graph(os);
}

CoGaDB::query_processing::LogicalQueryPlanPtr
SelectFrom::getLogicalQueryPlan() {
  return CoGaDB::query_processing::LogicalQueryPlanPtr(
      boost::make_shared<LogicalQueryPlan>(node));
}

TypedNodePtr SelectFrom::getQueryPlan() { return node; }

TablePtr Sequence::execute(ClientPtr client) {
  TablePtr result;

  for (std::list<StatementPtr>::iterator it = statements.begin();
       it != statements.end(); it++) {
    StatementPtr statement = *it;

    result = statement->execute(client);
  }

  return result;
}

void Sequence::explain(std::ostream &os) {
  for (std::list<StatementPtr>::iterator it = statements.begin();
       it != statements.end(); it++) {
    StatementPtr statement = *it;

    statement->explain(os);
  }
}

std::list<CoGaDB::query_processing::LogicalQueryPlanPtr>
Sequence::getLogicalQueryPlans() {
  std::list<CoGaDB::query_processing::LogicalQueryPlanPtr> result_plans;
  for (std::list<StatementPtr>::iterator it = statements.begin();
       it != statements.end(); it++) {
    StatementPtr statement = *it;

    // statement->explain(os);
    result_plans.push_back(statement->getLogicalQueryPlan());
  }
  return result_plans;
}

} /* namespace ParseTree */
} /* namespace SQL */
} /* namespace CoGaDB */
