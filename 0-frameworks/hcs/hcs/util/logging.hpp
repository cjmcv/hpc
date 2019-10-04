/*!
* \brief Log.
*/

#ifndef HCS_LOG_H_
#define HCS_LOG_H_

#include <iostream>
#include <sstream>

namespace hcs {

enum LogLevel {
  INFO = 0,
  WARNING = 1,
  ERROR = 2
};

class LogMessage : public std::basic_ostringstream<char> {
public:
  LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

  ~LogMessage() {
    if (severity_ >= min_log_level_)
      GenerateLogMessage();
  }

protected:
  void GenerateLogMessage() {
    fprintf(stderr, "<%c>", "IWE"[severity_]);
    if (fname_) {
      fprintf(stderr, " %s:%d] ", fname_, line_);
    }
    fprintf(stderr, "%s.\n", str().c_str());

    if (severity_ == ERROR) {
      std::abort();
    }
  }

public:
  static int min_log_level_;

private:
  const char* fname_;
  int line_;
  int severity_;
};

int LogMessage::min_log_level_ = WARNING;

#define LOG(severity) _HCS_LOG_##severity

#define _HCS_LOG_INFO    LogMessage(nullptr, 0, LogLevel::INFO)
#define _HCS_LOG_WARNING LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _HCS_LOG_ERROR   LogMessage(__FILE__, __LINE__, LogLevel::ERROR)

} // hcs.
#endif //HCS_LOG_H_