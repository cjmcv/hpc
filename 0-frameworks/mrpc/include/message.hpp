#ifndef MRPC_MESSAGE_H_
#define MRPC_MESSAGE_H_

#include <iostream>
#include <sstream>
#include <streambuf>

#include "serializer.hpp"

class MessageBuf : public std::stringbuf {
  const static size_t GROW_SIZE = 5;
  const static size_t MAX_SIZE = 55;

public:
  MessageBuf() {
    capacity_ = GROW_SIZE;
    data_ = (char*)malloc(capacity_);
    // set pointers for write buffer and read buffer.
    setp(data_, data_ + capacity_);
    setg(data_, data_, data_ + capacity_);
  }
  ~MessageBuf() {
    if (data_ != nullptr) {
      free(data_);
      data_ = nullptr;
      capacity_ = 0;
    }
  }

  inline char *data() { return data_; }
  inline const int size() const { return pptr() - pbase(); }
  inline const int capacity() const { return capacity_; }

  void Reset() {
    memset(data_, 0, sizeof(char) * capacity_);
    setp(data_, data_ + capacity_);
    setg(data_, data_, data_ + capacity_);
  }
  void PrintCapacity() {
    std::cout << in_avail() << std::endl;
    std::cout << "size of the get area is "
      << egptr() - eback() << " with "
      << egptr() - gptr() << " read positions available\n";
  }
  void Grow(int grow_size, bool is_from_tail = true) {
    int new_size = capacity_ + grow_size;
    char * new_buf = (char *)realloc(data_, new_size);
    // Reset write buffer, (start, The location of the element you are writing, end)
    // (pptr() - data_): The number of elements you have written.
    // (gptr() - data_): The number of elements you have read.
    if (is_from_tail) {
      setp(new_buf, new_buf + (pptr() - data_), new_buf + new_size);
      setg(new_buf, new_buf + (gptr() - data_), new_buf + new_size);
    }
    else {
      setp(new_buf, new_buf + (pptr() - data_) + grow_size, new_buf + new_size);
      setg(new_buf, new_buf + (gptr() - data_) + grow_size, new_buf + new_size);
    }
    data_ = new_buf;
    capacity_ = new_size;
  }
  void GrowTo(int size, bool is_from_tail = true) {
    if (capacity_ < size) {
      Grow(size - capacity_, is_from_tail);
    }
  }

protected:
  int_type overflow(int_type c) {
    std::cout << "overflow" << std::endl;
    // Grow.
    if (capacity_ + GROW_SIZE <= MAX_SIZE) {
      Grow(GROW_SIZE);
    }
    else {
      // Reset.
      setp(data_, data_, data_ + capacity_);
    }
    // Write.
    sputc(c);

    return c;
  }
  int_type underflow() override {
    std::cout << "underflow" << std::endl;
    return ' ';
  }
  int_type uflow() override {
    std::cout << "uflow" << std::endl;
    setg(data_, data_, data_ + capacity_);
    return EOF;
  }

private:
  char *data_;
  int capacity_;
};

class Message {
public:
  Message() {
    ostream_ = new std::ostream(&buffer_);
    istream_ = new std::istream(&buffer_);
  }
  ~Message() {
    delete ostream_;
    delete istream_;
  }

  inline void PackReady() {
    buffer_.Reset();
    ostream_->set_rdbuf(&buffer_);
  }
  template <class... Args>
  inline void Pack(Args&... args) {
    Serializer::Dump(*ostream_, args...);
  }

  inline void UnpackReady(const char *data, const int size) {
    buffer_.Reset();
    // Write string to buffer.
    ostream_->set_rdbuf(&buffer_);
    ostream_->write(data, size);
    // Read buffer by type.
    istream_->set_rdbuf(&buffer_);
  }
  inline void UnpackReady(std::string &str) {
    UnpackReady(str.c_str(), str.length());
  }

  template <class... Args>
  inline void Unpack(Args&... args) {
    Serializer::Load(*istream_, args...);
  }

protected:
  std::ostream *ostream_;
  std::istream *istream_; 
  MessageBuf buffer_;
};

class RpcMessage :public Message{
public:
  enum Mode {
    CALCULATION = 1,
    RESULT = 2
  };
private:
  enum { HEADER_LENGTH = 5 };

public:
  RpcMessage() { }
  ~RpcMessage() {
    if (body_buffer4reading_ != nullptr) {
      delete[]body_buffer4reading_;
      body_buffer4reading_ = nullptr;
    }
  }
public:
  inline char *body() const { return body_; }
  inline std::size_t body_length() const { return body_length_; }

  inline char *header() { return header_; }
  inline std::size_t header_length() const { return HEADER_LENGTH; }

  template <class... Args>
  void Pack(std::string func_name, Args&... args) {
    Message::PackReady();
    Message::Pack(func_name, args...);
    std::cout << "buffer_: " << buffer_.str() << ".(" << 
      buffer_.capacity() << ", " << buffer_.size() << "," << 
      buffer_.str().length();

    body_str_ = buffer_.str(); // Saved as a member string.

    body_ = (char *)body_str_.c_str();
    body_length_ = body_str_.length();

    std::sprintf(header_, "%4d", buffer_.size());
  }

  bool HeaderUnpack() {
    header_[HEADER_LENGTH - 1] = '\0';

    body_length_ = std::atoi(header_);
    if (body_buffer4reading_ != nullptr) {
      delete[]body_buffer4reading_;
      body_buffer4reading_ = nullptr;
    }
    body_buffer4reading_ = new char[body_length_];

    body_ = body_buffer4reading_;

    std::cout << "Unpack body_len:" << body_length_;


    buffer_.GrowTo(body_length_);
    return true;
  }

  bool HeaderUnpack(const char *header) {
    memcpy(header_, header, sizeof(char) * HEADER_LENGTH);
    return HeaderUnpack();
  }

  void Process(Mode mode) {
    // Save body to buffer.
    Message::UnpackReady(body(), body_length());
    // Unpack function name.
    std::string func_name;
    Serializer::ForVector<std::string, char>::Load(*istream_, func_name);

    // Unpack params according to function name.
    if (func_name == "add") {
      if (mode == CALCULATION) {
        std::string A;
        int B;
        char C;
        Message::Unpack(A, B, C);

        std::cout << "unpack: " << func_name << ", A: " << A << ",B : " << B << ", C :" << C << std::endl;

        int res = B + B;
        Pack(func_name, res);
      }
      else {
        int A;
        Message::Unpack(A);
        std::cout << "Receive result(" << func_name << ") : " << A << std::endl;
      }
    }
  }

private:
  std::string func_name_;
  
  char header_[HEADER_LENGTH];
  
  char *body_;
  std::string body_str_; // Saved for packing.
  char *body_buffer4reading_ = nullptr; // Saved for unpacking.

  size_t body_length_;
};

#endif // MRPC_MESSAGE_H_