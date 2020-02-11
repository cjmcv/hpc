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

private:
  inline void do_accept() {
    acceptor_.async_accept(
      [this](std::error_code ec, asio::ip::tcp::socket socket) {
      if (!ec) {
        std::make_shared<Session>(std::move(socket))->start();
      }
      do_accept();
    });
  }

private:
  asio::ip::tcp::acceptor acceptor_;
};

#endif // MRPC_SESSION_H_