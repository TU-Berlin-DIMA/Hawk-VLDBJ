/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <map>

namespace hype {
  namespace core {

    template <class AbstractProduct, typename IdentifierType,
              typename ProductCreator = AbstractProduct* (*)()>
    class Factory {
     private:
      typedef std::map<IdentifierType, ProductCreator> AssocMap;
      AssocMap associations_;

     public:
      Factory() : associations_() {}
      ~Factory() {
        associations_.erase(associations_.begin(), associations_.end());
      }

      bool Register(const IdentifierType& id, ProductCreator creator) {
        return associations_.insert(typename AssocMap::value_type(id, creator))
            .second;
      }

      bool Unregister(const IdentifierType& id) {
        return associations_.erase(id) == 1;
      }
      AbstractProduct* CreateObject(const IdentifierType& id) {
        typename AssocMap::const_iterator i = associations_.find(id);
        if (i != associations_.end()) {
          return i->second();
        }
        return NULL;
      }
    };

  }  // end namespace core
}  // end namespace hype
