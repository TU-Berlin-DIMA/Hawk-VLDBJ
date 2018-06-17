/*
 * File:   samQueryCreator.cpp
 * Author: john
 *
 * Created on 2. Juli 2015, 14:49
 */

#include "util/genomics_extension/samQueryCreator.hpp"
//#include <iostream>
#include <sstream>

// TODO rename into samQueryBuilder.cpp

SamQueryCreator::SamQueryCreator(const SamIdSelectionPtr selectionLinesByID) {
  selectionLinesWithContigID = selectionLinesByID;
}

// SamQueryCreator::SamQueryCreator(const SamNameSelectionPtr
// selectionLinesByName){
//    selectionLinesWithContigName = selectionLinesByName;
//}

SamQueryCreator::~SamQueryCreator() {}

void SamQueryCreator::_optimize() {
  // concat if same contigId and matching position
}

bool SamQueryCreator::_isLineWithThreeStars() {
  for (unsigned int i = 0; i < selectionLinesWithContigID->size(); i++) {
    if (boost::get<0>((*selectionLinesWithContigID)[i]) == -1 &&
        boost::get<1>((*selectionLinesWithContigID)[i]) == -1 &&
        boost::get<2>((*selectionLinesWithContigID)[i]) == -1)
      return true;
  }
  return false;
}

bool SamQueryCreator::_isContigSame() {
  int contigId = boost::get<2>((*selectionLinesWithContigID)[0]);
  for (unsigned int i = 1; i < selectionLinesWithContigID->size(); i++) {
    if (contigId != boost::get<2>((*selectionLinesWithContigID)[i]))
      return false;
  }
  return true;
}

bool SamQueryCreator::_isRangeSame() {
  int start = boost::get<0>((*selectionLinesWithContigID)[0]);
  int end = boost::get<1>((*selectionLinesWithContigID)[0]);
  for (unsigned int i = 1; i < selectionLinesWithContigID->size(); i++) {
    if ((start != boost::get<0>((*selectionLinesWithContigID)[i])) ||
        (end != boost::get<1>((*selectionLinesWithContigID)[i])))
      return false;
  }
  return true;
}

std::string SamQueryCreator::getQuery() {
  std::stringstream ss;
  std::ostream& out = ss;

  out << QUERY_SELECT << QUERY_FROM;
  unsigned int selLinesIdSize = selectionLinesWithContigID->size();

  // (*,*,*) -> no where clausel
  if (_isLineWithThreeStars()) {
    out << QUERY_ORDER_BY;
    return ss.str();
  }
  out << "where ";

  int tmpStart = 0;
  int tmpEnd = 0;
  int tmpContig = 0;

  for (unsigned int i = 0; i < selLinesIdSize; i++) {
    tmpStart = boost::get<0>((*selectionLinesWithContigID)[i]);
    tmpEnd = boost::get<1>((*selectionLinesWithContigID)[i]);
    tmpContig = boost::get<2>((*selectionLinesWithContigID)[i]);

    out << "(";
    // -1,-1,-1 is already caught above -> no "where ()" possible
    if (tmpContig != -1 && tmpStart == -1 && tmpEnd == -1)
      out << "c_id = " << tmpContig;
    else if (tmpContig != -1)
      out << "c_id = " << tmpContig << " and ";

    if (tmpStart == -1 && tmpEnd == -1) {
      out << ") ";
    } else if (tmpStart == -1) {
      out << "rb_position < " << tmpEnd << ") ";
    } else if (tmpEnd == -1) {
      out << "rb_position > " << tmpStart << ") ";
    } else {
      out << "rb_position between " << tmpStart << " and " << tmpEnd << ") ";
    }
    // Last line without or
    if (i != selLinesIdSize - 1) out << "or ";
  }
  out << QUERY_ORDER_BY;
  return ss.str();
}
