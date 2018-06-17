/*
 * File:   client.hpp
 * Author: sebastian
 *
 * Created on 27. August 2014, 19:06
 */

#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <boost/shared_ptr.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include <boost/asio.hpp>

namespace CoGaDB {

  class Client {
   public:
    virtual std::string getNextCommand() = 0;
    virtual std::ostream& getOutputStream() = 0;
    virtual ~Client();
  };

  typedef boost::shared_ptr<Client> ClientPtr;

  /* \brief for managing clients on local terminal with stdin and stdout*/
  class LocalClient : public Client {
   public:
    LocalClient();
    virtual std::string getNextCommand();
    virtual std::ostream& getOutputStream();
    ~LocalClient();

   private:
  };

  /* \brief for managing simple clients emulating parallel workloads*/
  class EmulatedClient : public Client {
   public:
    EmulatedClient(std::string file_name);
    virtual std::string getNextCommand();
    virtual std::ostream& getOutputStream();
    ~EmulatedClient();

   private:
    std::ofstream fout_;
  };

  /* \brief for managing clients over network*/
  class RemoteClient : public Client {
   public:
    RemoteClient(boost::asio::ip::tcp::socket* socket);
    virtual std::string getNextCommand();
    virtual std::ostream& getOutputStream();
    virtual std::string getOutput();
    //        virtual void flushStream();
    ~RemoteClient();

   private:
    boost::asio::ip::tcp::socket* socket_;
    std::stringstream stream_;
  };

  bool acceptNetworkConnections(const std::string& port_number,
                                ClientPtr client);
}

#endif /* CLIENT_HPP */
