/*
 * File:   positionlist.hpp
 * Author: sebastian
 *
 * Created on 2. November 2013, 22:28
 */

#pragma once

//#ifndef POSITIONLIST_HPP
#ifdef POSITIONLIST_HPP
#define POSITIONLIST_HPP

#include <core/global_definitions.hpp>

namespace CoGaDB {

  namespace test {

    enum { DEFAULT_POSITIONLIST_SIZE = 1000 };

    class PositionList {
     public:
      typedef TID value_type;
      typedef value_type& reference;
      typedef const value_type& const_reference;
      typedef value_type* pointer;
      typedef const value_type* const_pointer;
      typedef TID* iterator;
      typedef TID* const const_iterator;
      typedef ptrdiff_t difference_type;
      typedef size_t size_type;

      // default constructor
      // PositionList() : tid_ptr_(0), num_elements_(0),buffer_size_(0) {
      //    tid_ptr_=(TID*) malloc(DEFAULT_POSITIONLIST_SIZE*sizeof(TID));
      //    buffer_size_=DEFAULT_POSITIONLIST_SIZE*sizeof(TID);
      //
      //    assert(tid_ptr_!=NULL);
      //}

      PositionList();

      // fill constructor
      explicit PositionList(size_t number_of_elements, const TID& val = TID(0))
          : tid_ptr_(0), num_elements_(0), buffer_size_(0) {
        if (this->capacity() < number_of_elements) {
          this->reserve(number_of_elements);
        }
        // std::memset(tid_ptr_,val,number_of_elements);
        std::fill_n(tid_ptr_, number_of_elements, val);
        num_elements_ = number_of_elements;
      }

      // range	constructor
      template <class InputIterator>
      PositionList(InputIterator first, InputIterator last)
          : tid_ptr_(0), num_elements_(0), buffer_size_(0) {
        assert(last >= first);
        unsigned int new_size = last - first;
        if (this->capacity() < new_size) {
          this->reserve(new_size);
        }
        // std::memset(tid_ptr_,);
        assert(tid_ptr_ != NULL);
        std::copy(first, last, tid_ptr_);
        this->num_elements_ = new_size;
      }

      PositionList(const PositionList& x)
          : tid_ptr_(0), num_elements_(0), buffer_size_(0) {
        this->reserve(x.num_elements_);
        assert(tid_ptr_ != NULL);
        std::memcpy(tid_ptr_, x.tid_ptr_, x.num_elements_ * sizeof(TID));
        num_elements_ = x.num_elements_;
      }

      PositionList& operator=(const PositionList& other) {
        if (this != &other)  // protect against invalid self-assignment
        {
          // allocate new memory (using realloc, so we don't need to cleanup old
          // memory and malloc new memory!) and copy the elements
          this->reserve(other.num_elements_);  // calls realloc!
          assert(tid_ptr_ != NULL);
          std::memcpy(tid_ptr_, other.tid_ptr_,
                      other.num_elements_ * sizeof(TID));
          num_elements_ = other.num_elements_;
        }
        // by convention, always return *this
        return *this;
      }

      ~PositionList() {
        if (tid_ptr_) free(tid_ptr_);
      }

      inline TID& operator[](const int index) { return tid_ptr_[index]; }

      inline const TID& operator[](const int index) const {
        return tid_ptr_[index];
      }

      // inline bool operator==(PositionList& poslist){
      //    if(poslist.size()!=this->size()){
      //        return false;
      //    }
      //
      //    for(unsigned int i=0;i<poslist.size();++i){
      //        if(poslist[i]!=tid_ptr_[i]){
      //            cout << "Unequal Values! At position: " << i << endl;
      //            cout << "PositionList: " <<  poslist[i] << endl;
      //            cout << "Reference Vector: " <<  reference_vector[i] <<
      //            endl;
      //            return false;
      //        }
      //    }
      //    return true;
      //}

      TID* data() { return tid_ptr_; }

      size_t size() const { return num_elements_; }

      size_t capacity() const { return buffer_size_; }

      void reserve(size_t new_capacity);
      //{
      //
      //    if(new_capacity>buffer_size_){
      //        tid_ptr_=(TID*) realloc(tid_ptr_,new_capacity*sizeof(TID));
      //        buffer_size_=new_capacity;
      //    }
      //
      //}

      void resize(size_t new_size);
      //{
      //
      //    if(new_size>buffer_size_){
      //        tid_ptr_=(TID*) realloc(tid_ptr_,new_size*sizeof(TID));
      //        num_elements_=new_size;
      //    }else{
      //        num_elements_=new_size;
      //    }
      //}

      // void push_back(const TID& val);

      void push_back(const TID& val) __attribute__((always_inline)) {
        // std::cout << "#elements: " << num_elements_ << " buffer_size: " <<
        // buffer_size_ << std::endl;
        if (num_elements_ < buffer_size_ - 1) {
          // insert element
          tid_ptr_[num_elements_] = val;
          num_elements_++;
        } else {
          // std::cout << "realloc!" << std::endl;
          // increase memory by 70%
          size_t new_size = 1.7 * buffer_size_;
          // std::cout << "new size: " << new_size << std::endl;
          // tid_ptr is automatically updated by reserve
          this->reserve(new_size);
          // std::cout << "#elements: " << num_elements_ << " buffer_size: " <<
          // buffer_size_ << std::endl;
          assert(num_elements_ < buffer_size_);
          assert(this->capacity() == new_size);

          tid_ptr_[num_elements_] = val;
          num_elements_++;
        }
      }

      //
      // inline void push_back(const TID& val){
      //    if(num_elements_<buffer_size_-1){
      //        //insert element
      //        tid_ptr_[num_elements_]=val;
      //        num_elements_++;
      //    }else{
      //        //increase memory by 70%
      //        size_t new_size=1.7*buffer_size_;
      //        this->reserve(new_size);
      //
      //        tid_ptr_[num_elements_]=val;
      //        num_elements_++;
      //
      ////        tid_ptr_=(TID*) realloc(tid_ptr_,new_size*sizeof(TID));
      ////        //insert element
      ////        tid_ptr_[num_elements_++]=val;
      ////        //set new buffer capacity
      ////        buffer_size_=new_size;
      //    }
      //}

      TID* begin() {
        if (this->empty()) {
          return NULL;
        } else {
          return tid_ptr_;
        }
      }

      const TID* begin() const {
        if (this->empty()) {
          return NULL;
        } else {
          return tid_ptr_;
        }
      }

      TID* end() {
        if (this->empty()) {
          return NULL;
        } else {
          return &tid_ptr_[num_elements_];  // one element further than last
                                            // element
        }
      }

      const TID* end() const {
        if (this->empty()) {
          return NULL;
        } else {
          return &tid_ptr_[num_elements_];  // one element further than last
                                            // element
        }
      }

      bool empty() const {
        if (num_elements_ > 0)
          return false;
        else
          return true;
      }

      // void insert()

      ////insert single element
      // iterator insert (iterator position, const TID& val){
      //    //we currently only support insertion at the end of the
      //    PositionList!
      //    assert(position==this->end());
      //    this->push_back(val);
      //    //return pointer to inserted element
      //    return &tid_ptr_[this->num_elements_-1];
      //}
      //
      ////insert fill
      // void insert (iterator position, size_t number_of_new_elements, const
      // TID& val){
      //    //we currently only support insertion at the end of the
      //    PositionList!
      //    assert(position==this->end());
      //    for(unsigned int i=0;i<number_of_new_elements;++i){
      //        this->push_back(val);
      //    }
      //    //return pointer to inserted element
      //    //return &tid_ptr_[this->num_elements_-1];
      //}

      // insert range
      template <class InputIterator>
      void insert(iterator position, InputIterator first, InputIterator last) {
        // we currently only support insertion at the end of the PositionList!
        assert(position == this->end());
        assert(last >= first);
        assert(position - this->begin() >= 0);
        unsigned int insert_index = position - this->begin();
        unsigned int number_of_new_elements = last - first;
        unsigned int new_size = num_elements_ + number_of_new_elements;
        if (this->capacity() < new_size) {
          this->reserve(new_size);
        }
        assert(tid_ptr_ != NULL);
        // insert at end of vector
        if (this->empty()) {
          std::copy(first, last, tid_ptr_);
          this->num_elements_ = new_size;
        } else {
          // copy to end of vector
          iterator insert_position = &tid_ptr_[insert_index];
          assert(insert_position != NULL);
          assert(insert_position == this->end());
          std::copy(first, last, insert_position);
          //           unsigned int counter=0;
          //            while (first!=last) {
          //              std::cout << "ITeration: " << counter++ << std::endl;
          //              *insert_position = *first;
          //              ++insert_position; ++first;
          //
          //            }

          this->num_elements_ = new_size;
        }
      }

      void clear();
      //{
      //    num_elements_=0;
      //    //resize array to default size
      //    tid_ptr_=(TID*)
      //    realloc(tid_ptr_,DEFAULT_POSITIONLIST_SIZE*sizeof(TID));
      //    this->buffer_size_=DEFAULT_POSITIONLIST_SIZE*sizeof(TID);
      //}

     private:
      TID* tid_ptr_;
      size_t num_elements_;
      size_t buffer_size_;
    };

    inline bool operator==(const PositionList& poslist1,
                           const PositionList& poslist2) {
      if (poslist1.size() != poslist2.size()) {
        return false;
      }
      for (unsigned int i = 0; i < poslist1.size(); ++i) {
        if (poslist1[i] != poslist2[i]) {
          return false;
        }
      }
      return true;
    }

    inline bool operator!=(const PositionList& poslist1,
                           const PositionList& poslist2) {
      return !operator==(poslist1, poslist2);
    }

    // template <>
    // void PositionList::insert<TID*>(iterator position, TID* first, TID*
    // last){
    //    //we currently only support insertion at the end of the PositionList!
    //    assert(position==this->end());
    //    assert(last>=first);
    //    unsigned int number_of_new_elements=last-first;
    //    unsigned int new_size=num_elements_+number_of_new_elements;
    //    if(this->capacity()<new_size){
    //        this->reserve(new_size);
    //    }
    //    assert(tid_ptr_!=NULL);
    //    //insert at end of vector
    //    if(this->empty()){
    //        std::memcpy(tid_ptr_,first,number_of_new_elements*sizeof(TID));
    //        this->num_elements_=new_size;
    //    }else{
    //        assert(position!=NULL);
    //        std::memcpy(position,first,number_of_new_elements*sizeof(TID));
    //        this->num_elements_=new_size;
    //    }
    //}
  };

  typedef test::PositionList PositionList;
  /* \brief a PositionListPtr is a a references counted smart pointer to a
   * PositionList object*/
  typedef shared_pointer_namespace::shared_ptr<PositionList> PositionListPtr;
  /* \brief a PositionListPair is an STL pair consisting of two PositionListPtr
   * objects
   *  \details This type is returned by binary operators, e.g., joins*/
  typedef std::pair<PositionListPtr, PositionListPtr> PositionListPair;
  /* \brief a PositionListPairPtr is a a references counted smart pointer to a
   * PositionListPair object*/
  typedef shared_pointer_namespace::shared_ptr<PositionListPair>
      PositionListPairPtr;
};

#endif /* POSITIONLIST_HPP */
