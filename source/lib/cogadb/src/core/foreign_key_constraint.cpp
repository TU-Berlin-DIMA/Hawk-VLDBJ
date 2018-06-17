
#include <core/foreign_key_constraint.hpp>
#include <persistence/storage_manager.hpp>

namespace CoGaDB {

ForeignKeyConstraint::ForeignKeyConstraint()
    : name_of_referred_primary_key_col_(),
      name_of_referred_primary_key_table_() {}

ForeignKeyConstraint::ForeignKeyConstraint(
    std::string name_of_referred_primary_key_col,
    std::string name_of_referred_primary_key_table,
    std::string name_of_referring_foreign_key_col,
    std::string name_of_referring_foreign_key_table)
    : name_of_referred_primary_key_col_(name_of_referred_primary_key_col),
      name_of_referred_primary_key_table_(name_of_referred_primary_key_table),
      name_of_referring_foreign_key_col_(name_of_referring_foreign_key_col),
      name_of_referring_foreign_key_table_(
          name_of_referring_foreign_key_table) {}
ColumnPtr ForeignKeyConstraint::getPrimaryKeyColumn() const {
  TablePtr tab = getTablebyName(name_of_referred_primary_key_table_);
  if (!tab) return ColumnPtr();
  return tab->getColumnbyName(name_of_referred_primary_key_col_);
  //            std::vector<TablePtr> tables = getGlobalTableList().
}
std::string ForeignKeyConstraint::getNameOfPrimaryKeyColumn() const {
  return name_of_referred_primary_key_col_;
}
std::string ForeignKeyConstraint::getNameOfPrimaryKeyTable() const {
  return name_of_referred_primary_key_table_;
}
std::string ForeignKeyConstraint::getNameOfForeignKeyColumn() const {
  return name_of_referring_foreign_key_col_;
}
std::string ForeignKeyConstraint::getNameOfForeignKeyTable() const {
  return name_of_referring_foreign_key_table_;
}
}  // end namespace CoGaDB
