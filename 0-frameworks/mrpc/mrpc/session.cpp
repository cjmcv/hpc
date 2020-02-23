
#include <iostream>
#include "session.h"

namespace mrpc {

void Session::do_read_header() {
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

void Session::do_read_body() {
  auto self(shared_from_this());
  socket_.async_read_some(asio::buffer(message_.body(), message_.body_length()),
    [this, self](std::error_code ec, std::size_t length) {
    if (!ec) {
      if (length != message_.body_length())
        std::cout << "length != message.header_length()" << std::endl;

      std::string body_str = std::string(message_.body(), message_.body_length());
      std::cout << "body_str :" << body_str << std::endl;

      proc_->Run(message_);

      do_write_head();
    }
  });
}

void Session::do_write_head() {
  auto self(shared_from_this());
  asio::async_write(socket_, asio::buffer(message_.header(), message_.header_length()),
    [this, self](std::error_code ec, std::size_t /*length*/) {
    if (!ec) {
      do_write_body();
    }
  });
}

void Session::do_write_body() {
  auto self(shared_from_this());
  asio::async_write(socket_, asio::buffer(message_.body(), message_.body_length()),
    [this, self](std::error_code ec, std::size_t /*length*/) {
    if (!ec) {
      do_read_header();
    }
  });
}

} // namespace mrpc