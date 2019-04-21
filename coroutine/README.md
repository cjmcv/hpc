# Coroutines
A coroutine is a function that can suspend execution to be resumed later. Coroutines are stackless: they suspend execution by returning to the caller. This allows for sequential code that executes asynchronously (e.g. to handle non-blocking I/O without explicit callbacks), and also supports algorithms on lazy-computed infinite sequences and other uses.

## asyncio (Python)
asyncio is a library to write concurrent code using the async/await syntax.

asyncio is used as a foundation for multiple Python asynchronous frameworks that provide high-performance network and web-servers, database connection libraries, distributed task queues, etc.

asyncio is often a perfect fit for IO-bound and high-level structured network code.

Note: Python introduced the concept of coroutines in 3.4, and 3.5 defined the syntax for coroutines.

* Version: 3.5
* [Documentation 1](https://docs.python.org/3/library/asyncio.html)
* [Documentation 2](https://docs.python.org/3.5/library/asyncio-task.html)

## cpp20
Coroutine will be added to C++20 as a standard in 2020.
* [C++20 最新进展](https://www.oschina.net/news/104653/201902-kona-iso-c-committee-trip-report-c20)
* [cppreference](https://en.cppreference.com/w/cpp/language/coroutines)

## libco (C++)
libco is a coroutine library which is widely used in wechat back-end service. It has been running on tens of thousands of machines since 2013.
* Version: Clone from github on 2019.4.1
* [Github](https://github.com/Tencent/libco)
* [C/C++协程库libco：微信怎样漂亮地完成异步化改造](https://www.open-open.com/lib/view/open1481699603028.html)
* [Closure源代码分析](https://blog.csdn.net/MakeZero/article/details/80552509)
---
