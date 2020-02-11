#ifndef MRPC_CLIENT_H_
#define MRPC_CLIENT_H_

#include "asio.hpp"
#include "message.h"

class Client {
public:
  Client(asio::io_context& io_context, std::string host, std::string service):
    socket_(io_context),
    resolver_(io_context) {

    asio::connect(socket_, resolver_.resolve(host, service)); // <host> <port>
  }
  ~Client() {}

  template <class... Args>
  void Call(Args&... args) {
    try {
      // Send.
      {
        message_.Pack(args...);

        std::string header_str(message_.header(), message_.header_length());
        std::cout << "header_str:" << header_str << std::endl;

        asio::write(socket_, asio::buffer(message_.header(), message_.header_length()));
        asio::write(socket_, asio::buffer(message_.body(), message_.body_length()));
      }
      // Receive.
      {
        asio::error_code error;
        // Given the length of header, read header.
        size_t length = socket_.read_some(asio::buffer(message_.header(), message_.header_length()), error);
        //if (error == asio::error::eof)
        //  break; // Connection closed cleanly by peer.
        //else if (error)
        //  throw asio::system_error(error); // Some other error.

        // Unpack the header to get the length of body.
        message_.HeaderUnpack();
        // Given the length of body, read body.
        length = socket_.read_some(asio::buffer(message_.body(), message_.body_length()), error);

        message_.Process(RpcMessage::Mode::RESULT);
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