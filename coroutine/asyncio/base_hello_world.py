#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
brief: 
	Hello world.
    Record the basic usage of async, await and loop.

    async: It is used to declare a coroutine function.
    await: It is used to suspend its own coroutine and wait for another to complete.
    loop:   The event loop is the core of every asyncio application. 
          Event loops run asynchronous tasks and callbacks, perform network IO 
          operations, and run subprocesses.
"""

import asyncio

async def hello_world():
    print("Hello World!")
    await asyncio.sleep(1)
    print("Hello World again!")
    
loop = asyncio.get_event_loop()
# Blocking call which returns when the hello_world() coroutine is done
loop.run_until_complete(hello_world())
loop.close()