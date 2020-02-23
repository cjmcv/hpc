#include "message.h"

namespace mrpc {

///////////////
// MessageBuf
///////////////
MessageBuf::MessageBuf() {
  capacity_ = GROW_SIZE;
  data_ = (char*)malloc(capacity_);
  // set pointers for write buffer and read buffer.
  setp(data_, data_ + capacity_);
  setg(data_, data_, data_ + capacity_);
}
MessageBuf::~MessageBuf() {
  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
    capacity_ = 0;
  }
}

void MessageBuf::Reset() {
  memset(data_, 0, sizeof(char) * capacity_);
  setp(data_, data_ + capacity_);
  setg(data_, data_, data_ + capacity_);
}

void MessageBuf::Grow(int grow_size, bool is_from_tail) {
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

int MessageBuf::overflow(int c) {
  std::cout << "overflow" << std::endl;
  // Grow.
  if (capacity_ + GROW_SIZE <= MAX_SIZE)
    Grow(GROW_SIZE);
  else
    setp(data_, data_, data_ + capacity_);
  // Write.
  sputc(c);
  return c;
}

///////////////
// Message
///////////////
Message::Message() {
  ostream_ = new std::ostream(&buffer_);
  istream_ = new std::istream(&buffer_);
}
Message::~Message() {
  delete ostream_;
  delete istream_;
}

void Message::PackReady() {
  buffer_.Reset();
  ostream_->set_rdbuf(&buffer_);
}
void Message::UnpackReady(const char *data, const int size) {
  buffer_.GrowTo(size);
  buffer_.Reset();
  // Write string to buffer.
  ostream_->set_rdbuf(&buffer_);
  ostream_->write(data, size);
  // Read buffer by type.
  istream_->set_rdbuf(&buffer_);
}

///////////////
// RpcMessage
///////////////
bool RpcMessage::HeaderUnpack() {
  header_[HEADER_LENGTH - 1] = '\0';

  body_length_ = std::atoi(header_);
  if (body_buffer_ != nullptr) {
    delete[]body_buffer_;
    body_buffer_ = nullptr;
  }
  body_buffer_ = new char[body_length_];

  body_ =  body_buffer_; // TODO: 使用buffer_.data()代替，核对错误原因。
  memset(body_, 0, body_length_);

  //std::cout << "Unpack body_len:" << body_length_;
  return true;
}

bool RpcMessage::HeaderUnpack(const char *header) {
  memcpy(header_, header, sizeof(char) * HEADER_LENGTH);
  return HeaderUnpack();
}

} // namespace mrpc