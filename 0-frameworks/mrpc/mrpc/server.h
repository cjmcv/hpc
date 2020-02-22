#ifndef MRPC_SERVER_H_
#define MRPC_SERVER_H_

#include "asio.hpp"
#include "session.h"

class Server {
public:
  Server(asio::io_context& io_context, short port)
    : acceptor_(io_context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port)) {
    do_accept();
  }

  template<typename Response, typename... Args>
  inline void Bind(std::string func_name,
      typename _identity<std::function<Response(Args&...)>>::type func) {
    proc_.Bind<Response, Args...>(func_name, func);
  }

private:
  inline void do_accept() {
    acceptor_.async_accept(
      [this](std::error_code ec, asio::ip::tcp::socket socket) {
      if (!ec) {
        std::make_shared<Session>(std::move(socket), &proc_)->start();
      }
      do_accept();
    });
  }

private:
  Processor proc_;
  asio::ip::tcp::acceptor acceptor_;
};

#endif // MRPC_SESSION_H_