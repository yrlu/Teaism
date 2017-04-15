#ifndef SESSION_HPP_
#define SESSION_HPP_

class Session {
public:
  static Session * GetSession() {
    if (!session) {
      session = new Session();
    }
    return session;
  }
  ~Session() {
    delete session;
  }
  bool gpu = false;
  unsigned device = 0;
private:
  Session() {}
  static Session* session;
};

Session* Session::session = NULL;

#endif // SESSION_HPP_