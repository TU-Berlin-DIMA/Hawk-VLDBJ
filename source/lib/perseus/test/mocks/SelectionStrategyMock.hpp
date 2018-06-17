#ifndef PERSEUS_SELECTIONSTRATEGYMOCK_HPP
#define PERSEUS_SELECTIONSTRATEGYMOCK_HPP

#include <gmock/gmock.h>
#include "SelectionStrategy.hpp"

namespace perseus {

  class Variant;

  class SelectionStrategyMock : public SelectionStrategy {
   public:
    MOCK_METHOD0(selectVariant, Variant&());
    MOCK_METHOD2(reset, void(const VariantPool&, const bool));
    MOCK_METHOD0(finishQuery, void());
  };
}
#endif  // PERSEUS_SELECTIONSTRATEGYMOCK_HPP
