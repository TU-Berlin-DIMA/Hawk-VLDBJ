
#include <parser/client.hpp>
#include <parser/commandline_interpreter.hpp>

#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <core/variable_manager.hpp>

#include <util/filesystem.hpp>

using boost::asio::ip::tcp;
using namespace std;

namespace CoGaDB {

boost::asio::io_service io_service;

class tcp_connection : public boost::enable_shared_from_this<tcp_connection> {
 public:
  typedef boost::shared_ptr<tcp_connection> pointer;

  static pointer create(boost::asio::io_service& io_service) {
    return pointer(new tcp_connection(io_service));
  }

  tcp::socket& socket() { return socket_; }

  void start() {
    boost::shared_ptr<RemoteClient> client(new RemoteClient(&socket_));
    CommandLineInterpreter cmd(client);
    std::string prompt("CoGaDB>");
    std::string input;

    if (VariableManager::instance().getVariableValueString(
            "result_output_format") == "table") {
      std::stringstream welcome_msg;
      welcome_msg << "CoGaDB version " << COGADB_VERSION << endl;
      welcome_msg << "Enter 'help' for instructions" << endl;
      boost::asio::write(socket_, boost::asio::buffer(welcome_msg.str()));
    }

    // show interactive shell
    while (true) {
      try {
        if (VariableManager::instance().getVariableValueString(
                "result_output_format") == "table") {
          boost::asio::write(socket_, boost::asio::buffer(prompt));
        }
        input = client->getNextCommand();
        input = boost::algorithm::trim_copy(input);
      } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        std::cout << "Close connection with IP '"
                  << socket_.remote_endpoint().address().to_string()
                  << "' from port '" << socket_.remote_endpoint().port()
                  << "'..." << std::endl;

        return;
      }
      //            if (input.empty())
      //                return;
      //            if (input[0] == '#')
      //                return;
      //		if (cmd.getline(prompt, input)) {
      //			cout << endl;
      //			return;
      //		}

      if (input == "quit") {
        io_service.stop();
        // exit(0);
        return;
      }

      if (!cmd.execute(input, client))
        client->getOutputStream() << "Error! Command '" << input << "' failed!"
                                  << endl;
      message_ = client->getOutput();

      boost::asio::async_write(
          socket_, boost::asio::buffer(message_),
          boost::bind(&tcp_connection::handle_write, shared_from_this(),
                      boost::asio::placeholders::error,
                      boost::asio::placeholders::bytes_transferred));
    }
  }

 private:
  tcp_connection(boost::asio::io_service& io_service) : socket_(io_service) {}

  void handle_write(const boost::system::error_code& /*error*/,
                    size_t /*bytes_transferred*/) {}

  tcp::socket socket_;
  std::string message_;
};

class tcp_server {
 public:
  tcp_server(boost::asio::io_service& io_service, unsigned int port)
      : acceptor_(io_service, tcp::endpoint(tcp::v4(), port)) {
    start_accept();
  }

 private:
  void start_accept() {
    tcp_connection::pointer new_connection =
        tcp_connection::create(acceptor_.get_io_service());

    //        boost::asio::ip::tcp::iostream stream;
    //        acceptor_.accept(stream.rdbuf());
    // std::ostream f; f.rdbuf(new_connection->socket());
    acceptor_.async_accept(
        new_connection->socket(),
        boost::bind(&tcp_server::handle_accept, this, new_connection,
                    boost::asio::placeholders::error));
  }

  void handle_accept(tcp_connection::pointer new_connection,
                     const boost::system::error_code& error) {
    // start_accept();
    if (!error) {
      std::cout
          << "Accepted new connection from IP '"
          << new_connection->socket().remote_endpoint().address().to_string()
          << "' from port '"
          << new_connection->socket().remote_endpoint().port() << "'..."
          << std::endl;
      // new_connection->start();
      boost::thread t(boost::bind(&tcp_connection::start, new_connection));
      t.detach();
    }

    start_accept();
  }

  tcp::acceptor acceptor_;
};

Client::~Client() {}

/* \brief for managing clients on local terminal with stdin and stdout*/

LocalClient::LocalClient() : Client() {}
std::string LocalClient::getNextCommand() { return std::string(); }
std::ostream& LocalClient::getOutputStream() { return std::cout; }
LocalClient::~LocalClient() {}

/* \brief for managing simple clients emulating parallel workloads*/

EmulatedClient::EmulatedClient(std::string file_name) : Client(), fout_() {
  fout_.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);
}
std::string EmulatedClient::getNextCommand() {
  return getNextCommandFromGlobalCommandQueue();
}
std::ostream& EmulatedClient::getOutputStream() { return fout_; }
EmulatedClient::~EmulatedClient() {
  fout_.flush();
  fout_.close();
}

/* \brief for managing clients over network*/

RemoteClient::RemoteClient(boost::asio::ip::tcp::socket* socket)
    : Client(), socket_(socket), stream_() {}
std::string RemoteClient::getNextCommand() {
  //      std::string result;
  //      boost::system::error_code error;
  //
  //      size_t len = socket_->read_some(boost::asio::buffer(result), error);
  //
  //      if (error == boost::asio::error::eof)
  //        return ""; // Connection closed cleanly by peer.
  //      else if (error)
  //        throw boost::system::system_error(error); // Some other error.
  //
  //      return result;

  //      std::cout.write(result.c_str(), len);

  // boost::system::error_code error;

  //            boost::asio::streambuf receivedStreamBuffer;
  //
  //            // reserve 512 bytes in output sequence
  //            boost::asio::streambuf::mutable_buffers_type bufs
  //            =receivedStreamBuffer.prepare( 1024 );
  //            boost::asio::read( *socket_,
  //                                   bufs,
  //                                   error );
  //
  boost::asio::streambuf next_command;
  boost::asio::read_until(*socket_, next_command, "\n");

  std::istream command_stream(&next_command);
  // std::istream command_stream(&receivedStreamBuffer );
  std::string ret;
  std::getline(command_stream, ret);
  return ret;  // std::string("TERMINATE");
}
std::ostream& RemoteClient::getOutputStream() { return stream_; }
std::string RemoteClient::getOutput() {
  std::string result = stream_.str();
  stream_.clear();
  // delete content of stream
  stream_.str(std::string());
  return result;
}
//    virtual void RemoteClient::flushStream(){
//        std::string message = stream_.str();
//        boost::asio::async_write(socket_, boost::asio::buffer(message),
//              boost::bind(&tcp_connection::handle_write, shared_from_this(),
//              boost::asio::placeholders::error,
//              boost::asio::placeholders::bytes_transferred));
//    }
RemoteClient::~RemoteClient() {}

bool acceptNetworkConnections(const std::string& port_number,
                              ClientPtr client) {
  // we will be listening on port. check if we have to create a PID File
  // in case we got started by client script
  bool createPID =
      VariableManager::instance().getVariableValueBoolean("createPIDfile");

  try {
    if (createPID) {
      CoGaDB::createPIDFile();
    }

    unsigned int port = boost::lexical_cast<unsigned int>(port_number);

    tcp_server server(io_service, port);
    io_service.run();

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}
}
