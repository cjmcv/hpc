#ifndef MRPC_SERVER_H_
#define MRPC_SERVER_H_

#include <iostream>
#include <sstream>
#include <streambuf>

#include "asio.hpp"

#include "message.hpp"

using asio::ip::tcp;
class session: public std::enable_shared_from_this<session> {
public:
  session(tcp::socket socket): socket_(std::move(socket)) {}

  void start() { do_read_header(); }

private:
  void do_read_header() {
    std::cout << "do_read_header" << std::endl;
    auto self(shared_from_this());
    socket_.async_read_some(asio::buffer(message_.header(), message_.header_length()),
      [this, self](std::error_code ec, std::size_t length) {
      if (!ec) {
        message_.HeaderUnpack();

        //std::cout << "do_read_header in" << std::endl;
        //std::string header_str(message_.header(), message_.header_length());
        //std::cout << "header_str:" << header_str << std::endl;

        if (length != message_.header_length())
          std::cout << "length != message.header_length()" << std::endl;

        do_read_body();
      }
    });
  }

  void do_read_body() {
    auto self(shared_from_this());
    socket_.async_read_some(asio::buffer(message_.body(), message_.body_length()),
      [this, self](std::error_code ec, std::size_t length) {
      if (!ec) {
        if (length != message_.body_length())
          std::cout << "length != message.header_length()" << std::endl;

        std::string body_str = std::string(message_.body(), message_.body_length());
        std::cout << "body_str :" << body_str << std::endl;

        message_.Process(RpcMessage::Mode::CALCULATION);

        do_write_head();
      }
    });
  }
  void do_write_head() {
    auto self(shared_from_this());
    asio::async_write(socket_, asio::buffer(message_.header(), message_.header_length()),
      [this, self](std::error_code ec, std::size_t /*length*/) {
      if (!ec) {
        do_write_body();
      }
    });
  }
  void do_write_body() {
    auto self(shared_from_this());
    asio::async_write(socket_, asio::buffer(message_.body(), message_.body_length()),
      [this, self](std::error_code ec, std::size_t /*length*/) {
      if (!ec) {
        do_read_header();
      }
    });
  }

  tcp::socket socket_;
  RpcMessage message_;  
};

class Server {
public:
  Server(asio::io_context& io_context, short port)
    : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
    do_accept();
  }

private:
  void do_accept() {
    acceptor_.async_accept(
      [this](std::error_code ec, tcp::socket socket) {
      if (!ec) {
        std::make_shared<session>(std::move(socket))->start();
      }
      do_accept();
    });
  }

private:
  tcp::acceptor acceptor_;
};

#endif // MRPC_SERVER_H_