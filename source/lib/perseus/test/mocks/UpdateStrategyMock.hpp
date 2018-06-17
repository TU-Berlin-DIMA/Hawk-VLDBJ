#ifndef PERSEUS_UPDATESTRATEGYMOCK_HPP
#define PERSEUS_UPDATESTRATEGYMOCK_HPP

#include <gmock/gmock.h>
#include "UpdateStrategy.hpp"

namespace perseus {

  class UpdateStrategyMock : public UpdateStrategy {
   public:
    MOCK_METHOD1(updatePool, void(VariantPool&));
    MOCK_METHOD0(reset, void());
  };
}

#endif  // PERSEUS_UPDATESTRATEGYMOCK_HPP
