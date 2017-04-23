#ifndef SESSION_HPP_
#define SESSION_HPP_
#include <iostream>

class Session {
public:
  static Session* GetNewSession() {
    if (session) {
      Session * tmp = session;
      session = NULL;
      delete tmp;
    }
    session = new Session();
    return session;
  }
  static Session* GetSession() {
    if (!session) {
      session = new Session();
    }
    return session;
  }
  ~Session() {
    if (session) {
      delete session;
      session = NULL;
    }
  }
  bool gpu;
  unsigned device;
  size_t batch_size;
private:
  Session():gpu(false), device(0), batch_size(1) {}
  static Session* session;
};

Session* Session::session = NULL;

#endif // SESSION_HPP_