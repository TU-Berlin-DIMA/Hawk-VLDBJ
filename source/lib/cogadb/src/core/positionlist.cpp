//
//#include <core/positionlist.hpp>
//
// namespace CoGaDB{
//
//    namespace test{
//
//
// PositionList::PositionList() : tid_ptr_(0), num_elements_(0),buffer_size_(0)
// {
//    this->reserve(DEFAULT_POSITIONLIST_SIZE);
//    assert(tid_ptr_!=NULL);
//}
//
// void PositionList::reserve(size_t new_capacity){
//
//    if(new_capacity>buffer_size_){
//        tid_ptr_=(TID*) realloc(tid_ptr_,new_capacity*sizeof(TID));
//        assert(tid_ptr_!=NULL);
//        buffer_size_=new_capacity;
//    }
//
//}
//
// void PositionList::resize(size_t new_size){
//
//    if(new_size>buffer_size_){
//        tid_ptr_=(TID*) realloc(tid_ptr_,new_size*sizeof(TID));
//        num_elements_=new_size;
//    }else{
//        num_elements_=new_size;
//    }
//}
//
//
////void PositionList::push_back(const TID& val){
////    //std::cout << "#elements: " << num_elements_ << " buffer_size: " <<
/// buffer_size_ << std::endl;
////    if(num_elements_<buffer_size_-1){
////        //insert element
////        tid_ptr_[num_elements_]=val;
////        num_elements_++;
////    }else{
////        //std::cout << "realloc!" << std::endl;
////        //increase memory by 70%
////        size_t new_size=1.7*buffer_size_;
////        //std::cout << "new size: " << new_size << std::endl;
////        //tid_ptr is automatically updated by reserve
////        this->reserve(new_size);
////        //std::cout << "#elements: " << num_elements_ << " buffer_size: " <<
/// buffer_size_ << std::endl;
////        assert(num_elements_<buffer_size_);
////        assert(this->capacity()==new_size);
////
////        tid_ptr_[num_elements_]=val;
////        num_elements_++;
////
////    }
////}
//
//
// void PositionList::clear(){
//    num_elements_=0;
//    //resize array to default size
//    tid_ptr_=(TID*) realloc(tid_ptr_,DEFAULT_POSITIONLIST_SIZE*sizeof(TID));
//    this->buffer_size_=DEFAULT_POSITIONLIST_SIZE*sizeof(TID);
//}
//
//
//        };
//};
