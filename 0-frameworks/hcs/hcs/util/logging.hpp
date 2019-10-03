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
    // Read the min log level once during the first call to logging.
    if (severity_ >= MinVLogLevel())
      GenerateLogMessage();
  }

  // TODO: Change min_vlog_level.
  // If MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int MinVLogLevel() {
    static int min_vlog_level = INFO;
    return min_vlog_level;
  }

protected:
  void GenerateLogMessage() {
    if (fname_) {
      fprintf(stderr, "<%c> %s:%d] %s.\n", "IWE"[severity_], 
        fname_, line_, str().c_str());
    }
    else {
      fprintf(stderr, "<%c>%s.\n", "IWE"[severity_], str().c_str());
    }
  }

private:
  const char* fname_;
  int line_;
  int severity_;
};

#define LOG(severity) _HCS_LOG_##severity

#define _HCS_LOG_INFO    LogMessage(nullptr, 0, LogLevel::INFO)
#define _HCS_LOG_WARNING LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _HCS_LOG_ERROR   LogMessage(__FILE__, __LINE__, LogLevel::ERROR)

} // hcs.
#endif //HCS_LOG_H_
