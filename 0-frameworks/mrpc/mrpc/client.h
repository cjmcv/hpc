#ifndef MRPC_CLIENT_H_
#define MRPC_CLIENT_H_

#include "asio.hpp"
#include "message.h"

namespace mrpc {

class Client {
public:
  Client(asio::io_context& io_context, std::string host, std::string service):
    socket_(io_context),
    resolver_(io_context) {

    asio::connect(socket_, resolver_.resolve(host, service)); // <host> <port>
  }
  ~Client() {}

  template <typename Response, typename... Args>
  Response Call(std::string func_name, Args&... args) {
    try {
      // Send.
      {
        message_.Pack(func_name, args...);
        asio::write(socket_, asio::buffer(message_.header(), message_.header_length()));
        asio::write(socket_, asio::buffer(message_.body(), message_.body_length()));
      }
      // Receive.
      {
        asio::error_code error;
        // Read header according to the given length.
        socket_.read_some(asio::buffer(message_.header(), message_.header_length()));
        // Unpack the header to get the length of body.
        message_.UnpackHeader();
        // Given the length of body, read body.
        socket_.read_some(asio::buffer(message_.body(), message_.body_length()));
        // Unpack Body.
        std::string ret_func_name;
        message_.GetFuncName(ret_func_name);
        if (ret_func_name != func_name) {
          std::cout << "Received message: " << ret_func_name << std::endl;
          return -1;
        }

        Response ret;
        message_.GetArgs(ret);
        return ret;
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

} // namespace mrpc

#endif // MRPC_CLIENT_H_