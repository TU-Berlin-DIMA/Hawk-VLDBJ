/*
 * File:   samQueryCreator.hpp
 * Author: john
 *
 * Created on 2. Juli 2015, 14:49
 */
#pragma once

#include <boost/smart_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstdlib>
#include <string>
#include <vector>

// TODO Rename into samQueryBuilder.hpp
class SamQueryCreator {
 public:
  typedef std::vector<boost::tuple<int, int, int> > SamIdSelection;
  // typedef std::vector<boost::tuple<int, int, std::string> > SamNameSelection;
  typedef boost::shared_ptr<SamIdSelection> SamIdSelectionPtr;
  // typedef boost::shared_ptr<SamNameSelection> SamNameSelectionPtr;

  SamQueryCreator(const SamIdSelectionPtr selectionLinesByID);
  // SamQueryCreator(const SamNameSelectionPtr selectionLinesByName);
  // FIXME default Constructor gets created auto by compiler
  ~SamQueryCreator();
  std::string getQuery();

 private:
  void _optimize();
  bool _isContigSame();
  bool _isLineWithThreeStars();
  bool _isRangeSame();
  SamIdSelectionPtr selectionLinesWithContigID;
  // SamNameSelectionPtr selectionLinesWithContigName;
};
const std::string QUERY_SELECT =
    "select sb_read_id, r_qname, r_flag, c_name, rb_position, r_mapq, "
    "sb_base_value, "
    "sb_insert_offset, sb_base_call_quality, rg_name ";
const std::string QUERY_FROM =
    "from sample_base join reference_base on sb_rb_id = "
    "rb_id join contig on rb_c_id = c_id join read on sb_read_id = r_id ";
const std::string QUERY_ORDER_BY =
    "order by c_id, sb_read_id, rb_position, sb_insert_offset;";
// const std::string QUERY_ORDER_BY = "order by rb_c_id, sb_read_id, sb_rb_id,
// sb_insert_offset;";
// const std::string QUERY_ORDER_BY = "order by c_id;";
