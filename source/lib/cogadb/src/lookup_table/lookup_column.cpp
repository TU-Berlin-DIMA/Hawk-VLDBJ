
#include <core/lookup_array.hpp>
#include <lookup_table/lookup_column.hpp>
//#include <core/column_base_typed.hpp>
#include <iostream>

#include <util/utility_functions.hpp>

using namespace std;

namespace CoGaDB {

LookupColumn::LookupColumn(TablePtr table, PositionListPtr tids)
    : table_(table), tids_(tids) {
  assert(tids_ != NULL);
}

const ColumnPtr LookupColumn::getLookupArrayforColumnbyName(
    const Attribut& attr) const {
  string column_name = attr.second;
  // ColumnPtr col = table_->getColumnbyName(attr.second);

  const vector<ColumnPtr> columns = table_->getColumns();
  ColumnPtr col;
  for (unsigned int i = 0; i < columns.size(); i++) {
    if (columns[i]->getName() == attr.second &&
        columns[i]->getType() == attr.first) {
      if (!quiet && verbose)
        cout << "Found Column " << attr.second << " in Table "
             << table_->getName() << endl;
      col = columns[i];
    }
  }

  if (!col) {
    cout << "Error: Could not find Column " << attr.second << " in Table "
         << table_->getName() << endl;
    return ColumnPtr();  // boost::shared_ptr<LookupArray<T> >(  ); //return
                         // NULL Pointer
  }
  if (col->getType() != attr.first) {
    cout << "FATAL ERROR: FOUND Column " << col->getName() << " of Table "
         << table_->getName()
         << " does not match Attribut type for LookupColumn!" << endl;
    cout << "File: " << __FILE__ << " Line:" << __LINE__ << endl;
    return ColumnPtr();  // return NULL Pointer
  }

  //		if(typeid(T)!=col->type()){
  //			return boost::shared_ptr<LookupArray<T> >(  ); //return
  // NULL
  // Pointer
  //		}
  AttributeType type = col->getType();

  //	ColumnPtr ptr;
  return createLookupArrayForColumn(col, tids_);
}

/*
        const ColumnVectorPtr LookupColumn::getDenseValueColumns() const{

                ColumnVectorPtr result(new ColumnVector());

                TableSchema schema = table_->getSchema();
                TableSchema::iterator it;
                for(it=schema.begin();it!=schema.end();it++){
                        ColumnPtr col = getLookupArrayforColumnbyName(*it);
                        if(!col){
                                cout << "Error! LookupColumn requested column '"
   << it->second << "' which dos not exist in Table '" << table_->getName() <<
   "'!" << endl;
                                cout << "Schema inconsistent to original Table!"
   << endl;
                                exit(-1);
                        }
                        if(!quiet && verbose) cout << "Create Lookup Array for
   Column " << it->second << " of table " << table_->getName() << endl;
                        result->push_back(col);
                }
                return result;
        }*/

const ColumnVectorPtr LookupColumn::getLookupArrays() const {
  //
  if (!quiet && verbose) {
    cout << " LookupColumn::getLookupArrays(): ";
    if (table_->isMaterialized()) {
      cout << "Materialized Table: " << table_->getName() << endl;
    } else {
      cout << "Lookup Table: " << table_->getName() << endl;
    }
  }
  ColumnVectorPtr result(new ColumnVector());

  TableSchema schema = table_->getSchema();
  // if(!quiet && verbose && debug) table_->printSchema();
  TableSchema::iterator it;
  for (it = schema.begin(); it != schema.end(); it++) {
    if (!quiet && verbose && debug)
      cout << "Create LookupArray for Column " << table_->getName() << "."
           << it->second << endl;
    ColumnPtr col = getLookupArrayforColumnbyName(*it);
    if (!col) {
      cout << "Error! LookupColumn requested column '" << it->second
           << "' which dos not exist in Table '" << table_->getName() << "'!"
           << endl;
      cout << "Schema inconsistent to original Table!" << endl;
      exit(-1);
    }
    if (!quiet && verbose)
      cout << "Create Lookup Array for Column " << it->second << " of table "
           << table_->getName() << endl;
    result->push_back(col);
  }
  return result;
}

/*
const ColumnPtr LookupColumn::getLookupArrayforColumn(std::string column_name){

        ColumnPtr col = table_->getColumnbyName(column_name);

AttributeType type=col->getType();

//	ColumnPtr ptr;
if(type==INT){
        return ColumnPtr(new LookupArray<int>(column_name,INT,col,tids_));
}else if(type==FLOAT){
        return ColumnPtr(new LookupArray<float>(column_name,FLOAT,col,tids_));
}else if(type==VARCHAR){
        return ColumnPtr(new
LookupArray<string>(column_name,VARCHAR,col,tids_));
}else if(type==BOOLEAN){
        return ColumnPtr(new LookupArray<bool>(column_name,BOOLEAN,col,tids_));
        //ptr=ColumnPtr(new Column<bool>(name,BOOLEAN));
        cout << "Fatal Error! invalid AttributeType: " << type << " for Column:
" << column_name
                  << " Note: bool is currently not supported, will be added
again in the future!"<< endl;
}else{
        cout << "Fatal Error! invalid AttributeType: " << type << " for Column:
" << column_name << endl;
}
        return ColumnPtr();
}
*/

const TablePtr LookupColumn::getTable() const throw() { return table_; }

const PositionListPtr LookupColumn::getPositionList() const throw() {
  return tids_;
}

hype::ProcessingDeviceMemoryID LookupColumn::getMemoryID() const {
  return tids_->getMemoryID();
}

// returns pointer to new LookupColumn which is the result of this call
const shared_pointer_namespace::shared_ptr<LookupColumn>
LookupColumn::aggregate(const LookupColumn& lookup_col,
                        const ProcessorSpecification& proc_spec)
    const {  // it would be sufficient if this was a Postitionlist

  PositionListPtr aggregated_tids =
      createPositionList(0, hype::util::getMemoryID(proc_spec.proc_id));
  //		const PositionListPtr current_tids = tids_;

  if (tids_->empty()) {
    cout << "TID List is empty for Lookup Column of Table: "
         << lookup_col.table_->getName() << endl;
    return shared_pointer_namespace::shared_ptr<LookupColumn>(
        new LookupColumn(table_, aggregated_tids));
  }

  // place data
  const PositionListPtr lookup_col_tids =
      copy_if_required(lookup_col.getPositionList(), proc_spec);
  if (!lookup_col_tids) {
    return shared_pointer_namespace::shared_ptr<LookupColumn>();
  }
  const PositionListPtr current_tids = copy_if_required(tids_, proc_spec);
  if (!current_tids) {
    return shared_pointer_namespace::shared_ptr<LookupColumn>();
  }

  GatherParam param(proc_spec);
  ColumnPtr result = current_tids->gather(lookup_col_tids, param);
  if (!result) return shared_pointer_namespace::shared_ptr<LookupColumn>();
  aggregated_tids = boost::dynamic_pointer_cast<PositionList>(result);
  assert(aggregated_tids != NULL);
  // table stays the same, hence pointer is passed to new lookup table, but
  // positionlist is new, so pointer to new positionlist is passed
  return shared_pointer_namespace::shared_ptr<LookupColumn>(
      new LookupColumn(table_, aggregated_tids));

  //                const PositionListPtr lookup_col_tids =
  //                lookup_col.getPositionList();
  //
  //		PositionListPtr aggregated_tids=createPositionList();
  //		const PositionListPtr current_tids = tids_;
  //
  //		if(tids_->empty()){
  //			cout << "TID List is empty for Lookup Column of Table: "
  //<<
  // lookup_col.table_->getName() << endl;
  //			return
  // shared_pointer_namespace::shared_ptr<LookupColumn>(new
  // LookupColumn(table_, aggregated_tids) );
  //		}
  //
  //		TID tid(0), translated_tid(0);
  //		for(size_t i=0;i<lookup_col_tids->size();i++){
  //			tid=(*lookup_col_tids)[i];
  //			//translated_tid=(*current_tids)[tid];
  ////			try{
  ////			translated_tid=current_tids->at(tid);
  ////			}catch(std::out_of_range& e){
  ////				cout << e.what() << endl;
  ////				cout << "BAD TID: " << tid << " in Lookup Column
  /// at
  /// position
  ///"
  ///<< i << endl;
  ////                     		cout << "In File " << __FILE__ << ":" <<
  ///__LINE__ << endl;
  ////				exit(-1);
  ////			}
  //                        assert(tid<current_tids->size());
  //			translated_tid=(*current_tids)[tid]; //->at(tid);
  //			aggregated_tids->push_back(translated_tid);
  //		}
  //
  //		return shared_pointer_namespace::shared_ptr<LookupColumn>(new
  // LookupColumn(table_, aggregated_tids) ); //table stays the same, hence
  // pointer is passed to new lookup table, but positionlist is new, so pointer
  // to new positionlist is passed
}

const LookupColumnPtr LookupColumn::copy() const {
  PositionListPtr new_tids = boost::dynamic_pointer_cast<PositionList>(
      tids_->copy());  // createPositionList(*tids_); //call copy constructor of
                       // Positionlist (std::vector<TID>)
  if (!new_tids) return LookupColumnPtr();
  return LookupColumnPtr(new LookupColumn(table_, new_tids));
}

const shared_pointer_namespace::shared_ptr<LookupColumn> LookupColumn::copy(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  PositionListPtr new_tids = boost::dynamic_pointer_cast<PositionList>(
      tids_->copy(mem_id));  // createPositionList(*tids_); //call copy
                             // constructor of Positionlist (std::vector<TID>)
  if (!new_tids) return LookupColumnPtr();
  return LookupColumnPtr(new LookupColumn(table_, new_tids));
}

bool LookupColumn::append(LookupColumnPtr lookup_column) {
  if (!lookup_column) return false;

  return this->tids_->append(lookup_column->tids_);
}

void LookupColumn::save_impl(boost::archive::binary_oarchive& ar,
                             const unsigned int version) const {
  // we store join indexes only that index the join of two materialized tables
  assert(table_->isMaterialized());
  std::string table_name = table_->getName();
  size_t number_of_rows = tids_->size();
  TID* tid_array = tids_->data();

  // first store meta data, than tid array of position list
  ar& table_name;
  ar& number_of_rows;
  for (unsigned int index = 0; index < number_of_rows; ++index) {
    ar& tid_array[index];
  }
}

void LookupColumn::load_impl(boost::archive::binary_iarchive& ar,
                             const unsigned int version) {
  std::string table_name;
  size_t number_of_rows;
  // first load meta data, than tid array of position list
  ar& table_name;
  ar& number_of_rows;

  this->table_ = getTablebyName(table_name);
  this->tids_ = createPositionList(
      number_of_rows);  // PositionListPtr(createPositionList(number_of_rows));
  TID* tid_array = tids_->data();

  // load the tids inside the positionlist
  for (unsigned int index = 0; index < number_of_rows; ++index) {
    ar& tid_array[index];
  }
}

}  // end namespace CogaDB
