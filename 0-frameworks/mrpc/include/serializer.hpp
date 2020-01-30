#ifndef MRPC_SERIALIZER_H_
#define MRPC_SERIALIZER_H_

#include <iostream>

class Serializer {
public:
  template<class T, typename E = void>
  class Base {
  public:
    static inline void Dump(std::ostream& out, const T& object) {
      object.Dump(out);
    }
    static inline void Load(std::istream& in, T& object) {
      object.Load(in);
    }
  };

  template <class T>
  class ForPod {
  public:
    static inline void Dump(std::ostream& out, const T& object) {
      out.write((const char*)(&object), sizeof(T));
    }
    static inline void Load(std::istream& in, T& object) {
      in.read((char*)(&object), sizeof(T));
    }
  };

  template<class TVec, class TObj>
  class ForVector {
  public:
    static inline void Dump(std::ostream& out, const TVec& object) {
      uint32_t size = object.size();
      out.write((const char*)(&size), sizeof(size));
      for (const auto& obj : object) {
        Serializer::Dump(out, obj);
      }
    }

    static inline void Load(std::istream& in, TVec& object) {
      uint32_t size;
      in.read((char*)(&size), sizeof(size));
      object.clear();
      object.reserve(size);
      for (size_t i = 0; i < size; ++i) {
        TObj obj;
        Serializer::Load(in, obj);
        object.push_back(std::move(obj));
      }
    }
  };

public:
  template <class T>
  static inline void Dump(std::ostream& out, const T& t) {
    Base<T>::Dump(out, t);
  }
  template<class T, class... Args>
  static inline void Dump(std::ostream& out, const T& first, const Args&... args) {
    Dump(out, first);
    Dump(out, args...);
  }

  template <class T>
  static inline void Load(std::istream& in, T& t) {
    Base<T>::Load(in, t);
  }
  template <class T, class... Args>
  static inline void Load(std::istream& in, T& first, Args&... args) {
    Load(in, first);
    Load(in, args...);
  }
};

template<class T>
class Serializer::Base<T, typename std::enable_if<!std::is_class<T>::value>::type> : public Serializer::ForPod<T> {};
template <> class Serializer::Base<std::string> : public Serializer::ForVector<std::string, char> {};

#define HANDYPACK(...)                                  \
  inline virtual void Dump(std::ostream& out) const {   \
    Serializer::Dump(out, __VA_ARGS__);                 \
  }                                                     \
                                                        \
  inline virtual void Load(std::istream& in) {          \
    Serializer::Load(in, __VA_ARGS__);                  \
  }

#endif // MRPC_SERIALIZER_H_