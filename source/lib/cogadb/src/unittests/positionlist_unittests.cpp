/*
 * File:   positionlist_unittests.cpp
 * Author: sebastian
 *
 * Created on 3. November 2013, 12:27
 */

//#include <core/positionlist.hpp>
#include <core/column.hpp>
#include <iostream>

namespace CoGaDB {

namespace unit_tests {

using namespace std;

bool compare_is_equal(PositionList& poslist,
                      std::vector<TID>& reference_vector) {
  if (poslist.size() != reference_vector.size()) {
    COGADB_ERROR("Unequal Sizes!", "");
    return false;
  }

  for (unsigned int i = 0; i < poslist.size(); ++i) {
    if (poslist[i] != reference_vector[i]) {
      std::cout << "Unequal Values! At position: " << i << std::endl;
      std::cout << "PositionList: " << poslist[i] << std::endl;
      std::cout << "Reference Vector: " << reference_vector[i] << std::endl;
      return false;
    }
  }
  return true;
}

bool PositionListTest() {
  if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
    cout << "test for correctness of push back and default constructor" << endl;
  //    {//test for correctness of push back and default constructor
  //
  //        PositionList poslist("",OID);
  //        std::vector<TID> reference_vector;
  //
  //        assert(poslist.empty());
  //        assert(reference_vector.empty());
  //        assert(poslist.size()==reference_vector.size());
  //
  //        for(unsigned int i=0; i<1000; i++){
  //            poslist.push_back(i);
  //            reference_vector.push_back(i);
  //        }
  //
  //        if(!compare_is_equal(poslist,reference_vector)){
  //            COGADB_ERROR("PositionListTest Unittest: push back and default
  //            constructor test failed!","");
  //            return false;
  //        }
  //    }
  ////    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  ////    cout << "test for correctness of fill constructor" << endl;
  ////    {//test for correctness of fill constructor
  ////        for(unsigned int i=0; i<1000; i++){
  ////            unsigned int random_number_of_elements=rand()%(10*1000*1000);
  ////            unsigned int random_initializer=rand()%1000;
  ////            PositionList
  /// poslist(random_number_of_elements,random_initializer);
  ////            std::vector<TID>
  /// reference_vector(random_number_of_elements,random_initializer);
  ////            //check correctness
  ////            if(!compare_is_equal(poslist,reference_vector)){
  ////                COGADB_ERROR("PositionListTest Unittest: fill constructor
  /// test failed!","");
  ////                return false;
  ////            }
  ////        }
  ////    }
  //    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //    cout << "test for correctness of range constructor" << endl;
  //    {//test for correctness of range constructor
  //        for(unsigned int i=0; i<1000; i++){
  //            vector<TID> data;
  //            for(unsigned int j=0; j<1000*1000; j++){
  //                data.push_back(rand()%10000);
  //            }
  //
  //            PositionList poslist("",OID,data.begin(),data.end());
  //            std::vector<TID> reference_vector(data.begin(),data.end());
  //            //check correctness
  //            if(!compare_is_equal(poslist,reference_vector)){
  //                COGADB_ERROR("PositionListTest Unittest: range constructor
  //                test failed!","");
  //                return false;
  //            }
  //        }
  //    }
  //    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //    cout << "test for correctness of copy constructor" << endl;
  //    {//test for correctness of copy constructor
  //        for(unsigned int i=0; i<1000; i++){
  //            vector<TID> data;
  //            for(unsigned int j=0; j<1000*1000; j++){
  //                data.push_back(rand()%10000);
  //            }
  //
  //            PositionList poslist("",OID,data.begin(),data.end());
  //            std::vector<TID> reference_vector(data.begin(),data.end());
  //
  //            PositionList copy_poslist(poslist);
  //            //check correctness
  //            if(!compare_is_equal(copy_poslist,reference_vector)){
  //                COGADB_ERROR("PositionListTest Unittest: copy constructor
  //                test failed!","");
  //                return false;
  //            }
  //        }
  //    }
  //    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //    cout << "test for correctness of iterators" << endl;
  //    {//test for correctness of iterators
  //        for(unsigned int i=0; i<1000; i++){
  //            vector<TID> data;
  //            for(unsigned int j=0; j<1000*1000; j++){
  //                data.push_back(rand()%10000);
  //            }
  //
  //            PositionList poslist(data.begin(),data.end());
  //            //insert data using PositionList iterators
  //            PositionList copy_poslist(poslist.begin(),poslist.end());
  //            std::vector<TID>
  //            reference_vector(poslist.begin(),poslist.end());
  //            //check correctness
  //            if(!compare_is_equal(copy_poslist,reference_vector)){
  //                COGADB_ERROR("PositionListTest Unittest: copy constructor
  //                test failed!","");
  //                return false;
  //            }
  //            if(!compare_is_equal(copy_poslist,data)){
  //                COGADB_ERROR("PositionListTest Unittest: copy constructor
  //                test failed!","");
  //                return false;
  //            }
  //        }
  //    }
  //    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //    cout << "test for correctness of resize and size" << endl;
  //    {//test for correctness of resize and size
  //        PositionList poslist;
  //        std::vector<TID> reference_vector;
  //
  //        for(unsigned int i=0; i<1000; i++){
  //            unsigned int new_size=rand()%10000;
  //            poslist.resize(new_size);
  //            reference_vector.resize(new_size);
  //            assert(poslist.size()==new_size);
  //            assert(reference_vector.size()==new_size);
  //        }
  //
  //    }
  //    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //    cout << "test for correctness of reserve and capacity" << endl;
  //    {//test for correctness of reserve and capacity
  //        PositionList poslist;
  //        std::vector<TID> reference_vector;
  //
  //        for(unsigned int i=0; i<1000; i++){
  //            unsigned int new_size=rand()%10000;
  //            poslist.reserve(new_size);
  //            reference_vector.reserve(new_size);
  //            assert(poslist.capacity()>=new_size);
  //            assert(reference_vector.capacity()>=new_size);
  //        }
  //
  //        if(!compare_is_equal(poslist,reference_vector)){
  //            COGADB_ERROR("PositionListTest Unittest: reserve and capacity
  //            test failed!","");
  //            return false;
  //        }
  //    }
  //    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //    cout << "test for correctness of range insert (empty PositionList)" <<
  //    endl;
  //    {//test for correctness of range insert (empty PositionList)
  //        for(unsigned int i=0; i<1000; i++){
  //            vector<TID> data;
  //            for(unsigned int j=0; j<1000*1000; j++){
  //                data.push_back(rand()%10000);
  //            }
  //
  //
  //            PositionList copy_poslist;
  //            std::vector<TID> copy_reference_vector;
  //            //perform insertion in empty PositionList
  //            copy_poslist.insert(copy_poslist.end(),data.begin(),data.end());
  //            copy_reference_vector.insert(copy_reference_vector.end(),data.begin(),data.end());
  //
  //            //check correctness
  //            if(!compare_is_equal(copy_poslist,copy_reference_vector)){
  //                COGADB_ERROR("PositionListTest Unittest: range insert (empty
  //                PositionList) test failed!","");
  //                return false;
  //            }
  //        }
  //    }
  //    if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //    std::cout << "test for correctness of range insert (non empty
  //    PositionList)" << std::endl;
  //    {//test for correctness of range insert (non empty PositionList)
  //        for(unsigned int i=0; i<1000; i++){
  //            vector<TID> data;
  //            for(unsigned int j=0; j<1000*1000; j++){
  //                data.push_back(rand()%10000);
  //            }
  //
  //            PositionList poslist(data.begin(),data.end());
  //            std::vector<TID> reference_vector(data.begin(),data.end());
  //
  //            PositionList copy_poslist(poslist);
  //            std::vector<TID> copy_reference_vector(reference_vector);
  //            //perform insertion
  //            copy_poslist.insert(copy_poslist.end(),data.begin(),data.end());
  //            copy_reference_vector.insert(copy_reference_vector.end(),data.begin(),data.end());
  //
  //            //check correctness
  //            if(!compare_is_equal(copy_poslist,copy_reference_vector)){
  //                COGADB_ERROR("PositionListTest Unittest: range insert (non
  //                empty PositionList) test failed!","");
  //                return false;
  //            }
  //        }
  //    }

  return true;
}

}  // end namespace unit_tests

}  // end namespace CoGaDB
