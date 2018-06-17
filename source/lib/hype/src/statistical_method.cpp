
#include <core/algorithm.hpp>
#include <core/statistical_method.hpp>

namespace hype {
namespace core {

StatisticalMethod_Internal::StatisticalMethod_Internal(const std::string& name)
    : name_(name) {}

const boost::shared_ptr<StatisticalMethod_Internal>
getNewStatisticalMethodbyName(std::string name) {
  // StatisticalMethod_InternalFactory aFactory;
  StatisticalMethod_Internal* ptr =
      StatisticalMethodFactorySingleton::Instance().CreateObject(
          name);  //.( 1, createProductNull );
  return boost::shared_ptr<StatisticalMethod_Internal>(ptr);
  // 3. aFactory.CreateObject( 1 );
}

StatisticalMethodFactory& StatisticalMethodFactorySingleton::Instance() {
  static StatisticalMethodFactory factory;
  return factory;
}
}  // end namespace core
}  // end namespace hype
