#ifndef MRPC_MESSAGE_H_
#define MRPC_MESSAGE_H_

#include <iostream>
#include <sstream>
#include <streambuf>

#include "serializer.h"

namespace mrpc {

class MessageBuf : public std::stringbuf {
  const static size_t GROW_SIZE = 5;
  const static size_t MAX_SIZE = 55;

public:
  MessageBuf();
  ~MessageBuf();

  inline char *data() { return data_; }
  inline const int size() const { return pptr() - pbase(); }
  inline const int capacity() const { return capacity_; }

  void Reset();
  void Grow(int grow_size, bool is_from_tail = true);

  inline void GrowTo(int size, bool is_from_tail = true) {
    if (capacity_ < size) { Grow(size - capacity_, is_from_tail); }
  }

protected:
  int overflow(int c);

  inline int underflow() override { return ' '; }
  inline int uflow() override {
    setg(data_, data_, data_ + capacity_);
    return EOF;
  }

private:
  char *data_;
  int capacity_;
};

class Message {
public:
  Message();
  ~Message();

  inline std::string &buffer_str() { return buffer_str_; }

  void PackReady();

  template <class... Args>
  inline void Pack(Args&... args) {
    Serializer::Dump(*ostream_, args...);
    buffer_str_ = buffer_.str();
  }

  void UnpackReady(const char *data, const int size);
  inline void UnpackReady(std::string &str) {
    UnpackReady(str.c_str(), str.length());
  }

  template <class... Args>
  inline void Unpack(Args&... args) {
    Serializer::Load(*istream_, args...);
  }

protected:
  MessageBuf buffer_;

private:   
  std::string buffer_str_;
  std::ostream *ostream_;
  std::istream *istream_;
};

class RpcMessage :public Message{
  enum { 
    HEADER_LENGTH = 5,
    BASE_BODY_LENGTH = 1024,
    BODY_GROW_LENGTH = 1024
  };

public:
  ~RpcMessage() {
    if (body_buffer_ != nullptr) {
      delete[]body_buffer_;
      body_buffer_ = nullptr;
    }
  }

  inline char *body() const { return body_; }
  inline std::size_t body_length() const { return body_length_; }

  inline char *header() { return header_; }
  inline std::size_t header_length() const { return HEADER_LENGTH; }

  template <class... Args>
  void Pack(std::string func_name, Args&... args) {
    Message::PackReady();
    Message::Pack(func_name, args...);

    body_ = (char *)buffer_str().c_str();
    body_length_ = buffer_str().length();

    //std::cout << "body_length_: " << body_length_ << std::endl;
    std::sprintf(header_, "%4d", body_length_);
  }

  bool HeaderUnpack();
  bool HeaderUnpack(const char *header);

  inline void GetFuncName(std::string &func_name) {
    // Save body to the buffer first.
    Message::UnpackReady(body(), body_length());
    Message::Unpack(func_name);
  }

  template <typename T>
  inline void GetArgs(T &t) { Message::Unpack(t); }

private:
  std::string func_name_;

  char header_[HEADER_LENGTH];
  char *body_;
  size_t body_length_;

  // TODO: 固定为较大的长度，一次过开辟空间，发送时则紧跟body的len，当len超过时进行一次扩容。
  char *body_buffer_ = nullptr; // Saved for unpacking.
  size_t body_buffer_length_;
};

} //namespace mrpc
#endif // MRPC_MESSAGE_H_