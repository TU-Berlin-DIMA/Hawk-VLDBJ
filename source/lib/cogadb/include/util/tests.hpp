/*
 * File:   tests.hpp
 * Author: sebastian
 *
 * Created on 18. November 2015, 16:05
 */

#ifndef TESTS_HPP
#define TESTS_HPP

#include <boost/shared_ptr.hpp>

namespace CoGaDB {

  class Client;
  typedef boost::shared_ptr<Client> ClientPtr;

  void SetupTestConfiguration(ClientPtr client);

  bool loadReferenceDatabaseStarSchemaScaleFactor1(ClientPtr client);

  bool loadReferenceDatabaseTPCHScaleFactor1(ClientPtr client);
}

#endif /* TESTS_HPP */
