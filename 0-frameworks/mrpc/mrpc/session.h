#ifndef MRPC_SESSION_H_
#define MRPC_SESSION_H_

#include "asio.hpp"
#include "message.h"
#include "processor.h"

class Session : public std::enable_shared_from_this<Session> {
public:
  Session(asio::ip::tcp::socket socket, Processor *proc)
    : socket_(std::move(socket)), proc_(proc) {}

  inline void start() { do_read_header(); }

private:
  void do_read_header();
  void do_read_body();

  void do_write_head();
  void do_write_body();

private:
  asio::ip::tcp::socket socket_;
  RpcMessage message_;

  Processor *proc_;
};

#endif // MRPC_SESSION_H_