#ifndef MRPC_CLIENT_H_
#define MRPC_CLIENT_H_

#include "asio.hpp"
#include "message.h"

namespace mrpc {

template<typename T>
struct Response {
  T value;
  std::string error_str;
};

class Client {
public:
  Client(asio::io_context& io_context):
    socket_(io_context),
    resolver_(io_context) {}

  ~Client() {}

  inline void Connect(std::string &host, std::string &service) {
    asio::connect(socket_, resolver_.resolve(host, service)); // <host> <port>
  }

  template <typename RespT, typename... Args>
  Response<RespT> Call(std::string func_name, Args&... args) {
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
      Response<RespT> ret;
      std::string ret_func_name;
      message_.GetFuncName(ret_func_name);
      if (ret_func_name != func_name) {
        ret.error_str = "Received message: " + ret_func_name;
        ret.value = 0;
      }
      else {
        ret.error_str = "Success.";
        message_.GetArgs(ret.value);
      }
      return ret;
    }
  }

private:
  asio::ip::tcp::socket socket_;
  asio::ip::tcp::resolver resolver_; 
  
  RpcMessage message_;
};

} // namespace mrpc

#endif // MRPC_CLIENT_H_