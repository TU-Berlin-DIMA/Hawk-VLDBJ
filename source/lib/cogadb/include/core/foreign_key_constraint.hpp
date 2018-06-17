/*
 * File:   foreign_key_constraint.hpp
 * Author: sebastian
 *
 * Created on 3. Januar 2014, 10:27
 */

#ifndef FOREIGN_KEY_CONSTRAINT_HPP
#define FOREIGN_KEY_CONSTRAINT_HPP

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <core/base_column.hpp>
#include <string>

namespace CoGaDB {

  class ForeignKeyConstraint {
   public:
    ForeignKeyConstraint();
    ForeignKeyConstraint(std::string name_of_referred_primary_key_col,
                         std::string name_of_referred_primary_key_table,
                         std::string name_of_referring_foreign_key_col,
                         std::string name_of_referring_foreign_key_table);
    ColumnPtr getPrimaryKeyColumn() const;
    std::string getNameOfPrimaryKeyColumn() const;
    std::string getNameOfPrimaryKeyTable() const;
    std::string getNameOfForeignKeyColumn() const;
    std::string getNameOfForeignKeyTable() const;

   private:
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar &name_of_referred_primary_key_col_;
      ar &name_of_referred_primary_key_table_;
      ar &name_of_referring_foreign_key_col_;
      ar &name_of_referring_foreign_key_table_;
    }

    std::string name_of_referred_primary_key_col_;
    std::string name_of_referred_primary_key_table_;
    std::string name_of_referring_foreign_key_col_;
    std::string name_of_referring_foreign_key_table_;
  };

}  // end namespace CoGaDB
#endif /* FOREIGN_KEY_CONSTRAINT_HPP */
