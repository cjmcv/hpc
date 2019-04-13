#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
brief: 
	Executes nested coroutines.
"""

import asyncio

async def compute(x, y):
    print("Compute %s + %s ..." % (x, y))
    await asyncio.sleep(1.0)
    return x + y

async def print_sum(x, y):
    result = await compute(x, y)
    print("%s + %s = %s" % (x, y, result))
    
async def display(loop):
    end_time = loop.time() + 6.0
    while True:
        await print_sum(loop.time(), end_time)
        #print(loop.time())
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(1)

loop = asyncio.get_event_loop()
# Blocking call which returns when the display_date() coroutine is done
loop.run_until_complete(display(loop))
loop.close()