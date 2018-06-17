#pragma once

#include <core/base_column.hpp>
//#include <core/column.hpp>
#include <core/base_table.hpp>
#include <persistence/storage_manager.hpp>

namespace CoGaDB {

  class LookupColumn {
   public:
    LookupColumn(TablePtr table, PositionListPtr tids);

    //	template <typename T>
    //	const shared_pointer_namespace::shared_ptr<LookupArray<T> >
    // getColumnbyName(std::string s) const;

    const ColumnPtr getLookupArrayforColumnbyName(const Attribut& attr) const;

    // const ColumnVectorPtr getDenseValueColumns() const;

    const ColumnVectorPtr getLookupArrays() const;

    //	const ColumnPtr getLookupArrayforColumn(std::string column_name);

    const TablePtr getTable() const throw();

    const PositionListPtr getPositionList() const throw();

    hype::ProcessingDeviceMemoryID getMemoryID() const;

    const shared_pointer_namespace::shared_ptr<LookupColumn> aggregate(
        const LookupColumn& lookup_col,
        const ProcessorSpecification& proc_spec) const;
    /* \brief returns copy of LookupColumn*/
    const shared_pointer_namespace::shared_ptr<LookupColumn> copy() const;

    const shared_pointer_namespace::shared_ptr<LookupColumn> copy(
        const hype::ProcessingDeviceMemoryID& mem_id) const;

    bool append(LookupColumnPtr lookup_column);

    friend class boost::serialization::access;

   protected:
    TablePtr table_;
    mutable PositionListPtr tids_;

   private:
    void save_impl(boost::archive::binary_oarchive& ar,
                   const unsigned int version) const;
    void load_impl(boost::archive::binary_iarchive& ar,
                   const unsigned int version);

    template <class Archive>
    void save(Archive& ar, const unsigned int version) const {
      COGADB_FATAL_ERROR("Called unimplemented method", "");
    }
    template <class Archive>
    void load(Archive& ar, const unsigned int version) {
      COGADB_FATAL_ERROR("Called unimplemented method", "");
    }

    //        template<>
    //        void
    //        save<boost::archive::binary_oarchive>(boost::archive::binary_oarchive&
    //        ar, const unsigned int version) const{
    //
    //        }
    //
    //        template<>
    //        void
    //        load<boost::archive::binary_oarchive>(boost::archive::binary_oarchive&
    //        ar, const unsigned int version){
    //
    //        }

    ////        template<class Archive>
    ////        void save(Archive & ar, const unsigned int version) const
    //        void save(boost::archive::binary_oarchive& ar, const unsigned int
    //        version) const
    //        {
    //            //we store join indexes only that index the join of two
    //            materialized tables
    //            assert(table_->isMaterialized());
    //            std::string table_name = table_->getName();
    //            size_t number_of_rows = tids_->size();
    //            TID* tid_array = tids_->data();
    //
    //            //first store meta data, than tid array of position list
    //            ar & table_name;
    //            ar & number_of_rows;
    //            for (unsigned int index = 0; index < number_of_rows; ++index){
    //                ar & tid_array[index];
    //            }
    //        }
    ////        template<class Archive>
    ////        void load(Archive & ar, const unsigned int version)
    //        void load(boost::archive::binary_oarchive& ar, const unsigned int
    //        version)
    //        {
    //
    //            std::string table_name;
    //            size_t number_of_rows;
    //            //first load meta data, than tid array of position list
    //            ar & table_name;
    //            ar & number_of_rows;
    //
    //            this->table_=getTablebyName(table_name);
    //            this->tids_=createPositionList(number_of_rows);
    //            //PositionListPtr(createPositionList(number_of_rows));
    //            TID* tid_array = tids_->data();
    //
    //            //load the tids inside the positionlist
    //            for (unsigned int index = 0; index < number_of_rows; ++index){
    //                ar & tid_array[index];
    //            }
    //        }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
  };

  //	typedef LookupColumn::PositionListPtr PositionListPtr;
  //	typedef shared_pointer_namespace::shared_ptr<LookupColumn>
  // LookupColumnPtr;
  //
  //
  //        typedef std::pair<LookupColumnPtr, LookupColumnPtr> JoinIndex;
  //        typedef boost::shared_ptr<JoinIndex> JoinIndexPtr;

  template <>
  inline void LookupColumn::save<boost::archive::binary_oarchive>(
      boost::archive::binary_oarchive& ar, const unsigned int version) const {
    save_impl(ar, version);
  }

  template <>
  inline void LookupColumn::load<boost::archive::binary_iarchive>(
      boost::archive::binary_iarchive& ar, const unsigned int version) {
    load_impl(ar, version);
  }

  // template <typename T>
  //	const ColumnPtr LookupColumn::getLookupArrayforColumnbyName(const
  // Attribut& attr) const{
  //
  //		ColumnPtr column = table_->getColumnbyName(attr.second);
  //		if(!column){
  //			return ColumnPtr();
  ////shared_pointer_namespace::shared_ptr<LookupArray<T> >(  ); //return NULL
  // Pointer
  //		}
  //		if(column->getType()!=attr.first){
  //			cout << "FATAL ERROR: FOUND Column " <<
  // column->getName()
  //<<
  //"
  // of Table " << table_->getName() << " does not match Attribut type for
  // LookupColumn!" << endl;
  //			cout << "File: " << __FILE__ << " Line:" << __LINE__ <<
  // endl;
  //			return ColumnPtr(); //return NULL Pointer
  //		}

  //		if(typeid(T)!=column->type()){
  //			return
  // shared_pointer_namespace::shared_ptr<LookupArray<T>
  //>(
  //); //return NULL Pointer
  //		}

  //		return shared_pointer_namespace::shared_ptr<LookupArray<T> >(
  // new
  // LookupArray<T>(column,tids_) );
  //	}

}  // end namespace CogaDB
