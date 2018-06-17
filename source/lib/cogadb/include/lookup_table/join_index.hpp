/*
 * File:   join_index.hpp
 * Author: bress
 *
 * Created on January 28, 2014, 8:07 PM
 */

#ifndef JOIN_INDEX_HPP
#define JOIN_INDEX_HPP

#include <lookup_table/lookup_table.hpp>

namespace CoGaDB {

  class JoinIndexes {
   public:
    static JoinIndexes& instance();

    // allows two perform fetch joins for join queries where the primary key
    // table is prefiltered before the join
    // this is important for the first phase of the invisible join
    JoinIndexPtr getJoinIndex(TablePtr pk_tab, std::string pk_tab_col_name,
                              TablePtr fk_tab, std::string fk_tab_col_name);

    // Reverse Join Indexes allow for efficient tuple reconstruction
    // this is important for the second phase in the invisible join
    const PositionListPtr getReverseJoinIndex(TablePtr pk_tab,
                                              std::string pk_tab_col_name,
                                              TablePtr fk_tab,
                                              std::string fk_tab_col_name);
    /*! \brief returns true if an existing positionlist is a reverse join index,
     *  return false otherwise
     */
    bool isReverseJoinIndex(const PositionListPtr tids) const;

    typedef std::pair<JoinIndexPtr, PositionListPtr> IndexPair;

    TablePtr getSystemTable();
    bool loadJoinIndexesFromDisk();
    bool placeJoinIndexesOnGPU(const hype::ProcessingDeviceMemoryID& mem_id);

   private:
    const JoinIndexPtr createJoinIndex(TablePtr pk_tab,
                                       std::string pk_tab_col_name,
                                       TablePtr fk_tab,
                                       std::string fk_tab_col_name);
    const PositionListPtr createReverseJoinIndex(TablePtr pk_tab,
                                                 std::string pk_tab_col_name,
                                                 TablePtr fk_tab,
                                                 std::string fk_tab_col_name);

    std::string joinExpressionToString(TablePtr pk_tab,
                                       std::string pk_tab_col_name,
                                       TablePtr fk_tab,
                                       std::string fk_tab_col_name);

    typedef std::map<std::string, JoinIndexPtr> Indexes;
    typedef std::map<std::string, PositionListPtr> ReverseJoinIndexes;

    Indexes indeces_;
    ReverseJoinIndexes reverse_indeces_;
  };

  // for star joins, such as invisible join
  PositionListPtr fetchMatchingTIDsFromJoinIndex(JoinIndexPtr join_index,
                                                 PositionListPtr pk_table_tids);
  PositionListPtr fetchMatchingTIDsFromJoinIndexInParallel(
      JoinIndexPtr join_index, PositionListPtr pk_table_tids);
  // for normal pk fk joins
  PositionListPairPtr fetchJoinResultFromJoinIndex(
      JoinIndexPtr join_index, PositionListPtr matching_tids_pk_column);
  // void convertPositionListToBitmap(PositionListPtr tids, char* bitmap, size_t
  // num_rows_of_indexed_table);

  BitmapPtr createBitmapOfMatchingTIDsFromJoinIndex(
      JoinIndexPtr join_index, PositionListPtr matching_tids);
  BitmapPtr createBitmapOfMatchingTIDsFromJoinIndexParallel(
      JoinIndexPtr join_index, PositionListPtr matching_tids);
  // void setBitmapMatchingTIDsFromJoinIndex(JoinIndexPtr join_index,
  // PositionListPtr matching_tids, char* fk_column_matching_rows_bitmap, size_t
  // bitmap_num_bits);

  std::string toString(JoinIndexPtr);
  size_t getSizeInBytes(const JoinIndexPtr);

  bool storeJoinIndex(JoinIndexPtr join_index,
                      const std::string& join_index_name);
  JoinIndexPtr loadJoinIndex(const std::string& join_index_name);

  bool storeReverseJoinIndex(PositionListPtr reverse_join_index,
                             const std::string& join_index_name);
  PositionListPtr loadReverseJoinIndex(const std::string& join_index_name);

  bool placeColumnOnCoprocessor(ClientPtr client,
                                const hype::ProcessingDeviceMemoryID& mem_id,
                                const std::string& table_name,
                                const std::string& column_name);
  bool placeSelectedColumnsOnGPU(ClientPtr client);
  TablePtr getSystemTableJoinIndexes();

  //        static JoinIndexes& JoinIndexes::instance(){
  //            static JoinIndexes join_indx;
  //            return join_indx;
  //        }
  //
  //        std::string JoinIndexes::joinExpressionToString(TablePtr pk_tab,
  //        std::string pk_tab_col_name, TablePtr fk_tab, std::string
  //        fk_tab_col_name){
  //            std::stringstream ss;
  //            ss << "JOIN_INDEX(" <<  pk_tab->getName() << "." <<
  //            pk_tab_col_name << "," << fk_tab->getName() << "." <<
  //            fk_tab_col_name << ")";
  //            return ss.str();
  //        }
  //
  //        JoinIndexPtr JoinIndexes::getJoinIndex(TablePtr pk_tab, std::string
  //        pk_tab_col_name, TablePtr fk_tab, std::string fk_tab_col_name){
  //            assert(pk_tab->hasPrimaryKeyConstraint(pk_tab_col_name)==true);
  //            assert(fk_tab->hasForeignKeyConstraint(fk_tab_col_name)==true);
  //            std::string index_name = joinExpressionToString(pk_tab,
  //            pk_tab_col_name, fk_tab, fk_tab_col_name);
  //            Indexes::iterator it = indeces_.find(index_name);
  //            if(it!=indeces_.end()){
  //                return it->second;
  //            }else{
  //                JoinIndexPtr join_index = createJoinIndex(pk_tab,
  //                pk_tab_col_name, fk_tab, fk_tab_col_name);
  //                this->indeces_.insert(join_index);
  //                return join_index;
  //            }
  //
  //        }
  //
  //        JoinIndexPtr JoinIndexes::createJoinIndex(TablePtr pk_tab,
  //        std::string pk_tab_col_name, TablePtr fk_tab, std::string
  //        fk_tab_col_name){
  //            TablePtr tab = BaseTable::join(pk_tab, pk_tab_col_name, fk_tab,
  //            fk_tab_col_name, LOOKUP, HASH_JOIN);
  //            const std::list<std::string> column_names;
  //            column_names.push_back(pk_tab_col_name, fk_tab_col_name);
  //            tab=BaseTable::sort(tab,column_names, ASCENDING, LOOKUP, CPU);
  //
  //            LookupTablePtr lookup_table =
  //            boost::dynamic_pointer_cast<LookupTable>(tab);
  //            const std::vector<LookupColumnPtr> lookup_columns =
  //            lookup_table->getLookupColumns();
  //            assert(lookup_columns.size()==2);
  //            JoinIndexPtr join_index(new
  //            JoinIndex(lookup_columns[0],lookup_columns[1]));
  //            //TODO: FIXME: sort after column 0, then after column 1
  //
  ////            typedef std::pair<TID,TID> TID_Pair;
  ////            typedef std::vector<TID_Pair>  TID_Pair_Vector;
  ////            TID_Pair_Vector join_pairs;
  ////            PositionListPtr pk_column_tids =
  /// join_index->first->getPositionList();
  ////            PositionListPtr fk_column_tids =
  /// join_index->second->getPositionList();
  ////            for(unsigned int i=0;i<lookup_table->getNumberofRows();++i){
  ////                join_pairs.push_back(TID_Pair(pk_column_tids[i],
  /// fk_column_tids[i]));
  ////            }
  ////            std::sort(join_pairs.begin(), join_pairs.end());
  ////
  ////            PositionListPtr pk_column_tids_sorted;
  ////            PositionListPtr fk_column_tids_sorted;
  ////            for(unsigned int i=0;i<join_pairs.size();++i){
  ////                pk_column_tids_sorted[i]=join_pairs[i].first;
  ////                fk_column_tids_sorted[i]=join_pairs[i].second;
  ////            }
  ////
  ////            LookupColumnPtr pk_lc_sorted(new
  /// LookupColumn(join_index->first->getTable(),pk_column_tids_sorted));
  ////            LookupColumnPtr fk_lc_sorted(new
  /// LookupColumn(join_index->second->getTable(),fk_column_tids_sorted));
  //
  //            return join_index;
  //        }
  //
  //
  //    PositionListPtr fetchMatchingTIDsFromJoinIndex(JoinIndexPtr join_index,
  //    PositionListPtr pk_table_tids);
  //
  //    //fetch TIDs of FK Column from JoinIndex
  //    //matching tids comes from a filter operation on a primary key table,
  //    and we now seek the corresponding foreign key tids
  //    //Note: assumes the Join Index is sorted by the first PositionList (the
  //    Primary Key Table TID list)
  //    PositionListPtr fetchMatchingTIDsFromJoinIndex(JoinIndex join_index,
  //    PositionListPtr matching_tids){
  //        PositionListPtr fk_column_matching_tids;
  //        PositionListPtr pk_column_tids =
  //        join_index->first->getPositionList();
  //        PositionListPtr fk_column_tids =
  //        join_index->second->getPositionList();
  //        //intersect(pk_column_matching_tids, matching_tids) -> matching fks
  //        unsigned int left_pos=0;
  //        unsigned int right_pos=0;
  //        unsigned int
  //        number_of_join_pairs=join_index->first->getPositionList()->size();
  //        //assume that matching tids contains no duplicates and is sorted
  //        while(left_pos < matching_tids->size() && right_pos <
  //        number_of_join_pairs){
  //            if(matching_tids[left_pos] == pk_column_tids[right_pos]){
  //                fk_column_matching_tids->push_back(fk_column_tids[right_pos]);
  //                right_pos++;
  //            }else if(matching_tids[left_pos] < pk_column_tids[right_pos]){
  //                left_pos++;
  //            }else if(matching_tids[left_pos] > pk_column_tids[right_pos]){
  //                right_pos++;
  //            }
  //        }
  //        return fk_column_matching_tids;
  //    }

}  // end namespace CogaDB

#endif /* JOIN_INDEX_HPP */
