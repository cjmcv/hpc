#ifndef MRPC_CLIENT_H_
#define MRPC_CLIENT_H_

#include <iostream>
#include <sstream>
#include <streambuf>
#include "asio.hpp"

#include "message.hpp"

class Client {
public:
  Client(asio::io_context& io_context, std::string host, std::string service):
    socket_(io_context),
    resolver_(io_context) {

    asio::connect(socket_, resolver_.resolve(host, service)); // <host> <port>
  }
  ~Client() {}

  void Call() {
    try {
      // Send.
      {
        std::string func_name = "add";
        std::string A = "rpc hello world again";
        int B = 30;
        char C = 'c';

        message_.Pack(func_name, A, B, C);

        std::string header_str(message_.header(), message_.header_length());
        std::cout << "header_str:" << header_str << std::endl;

        asio::write(socket_, asio::buffer(message_.header(), message_.header_length()));
        asio::write(socket_, asio::buffer(message_.body(), message_.body_length()));
      }
      // Receive.
      {
        asio::error_code error;
        size_t length = socket_.read_some(asio::buffer(message_.header(), message_.header_length()), error);
        //if (error == asio::error::eof)
        //  break; // Connection closed cleanly by peer.
        //else if (error)
        //  throw asio::system_error(error); // Some other error.

        message_.HeaderUnpack();
        size_t body_length = message_.body_length();
        char *body = new char[body_length];
        length = socket_.read_some(asio::buffer(body, body_length), error);
        std::string body_str = std::string(body, length);
        std::cout << "receive body_str :" << body_str << std::endl;
      }
    }
    catch (std::exception& e) {
      std::cerr << "Exception: " << e.what() << "\n";
    }
  }

private:
  asio::ip::tcp::socket socket_;
  asio::ip::tcp::resolver resolver_; 
  
  RpcMessage message_;
};

#endif // MRPC_CLIENT_H_