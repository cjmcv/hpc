#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
brief: 
	Record the basic usage of future.
"""

import asyncio

async def slow_operation(future):
    await asyncio.sleep(1)
    future.set_result('Future is done!')

def got_result(future):
    print(future.result())
    loop.stop()

loop = asyncio.get_event_loop()
future = asyncio.Future()
asyncio.ensure_future(slow_operation(future))

print("Start loop.run_until_complete(future)")
loop.run_until_complete(future)
print(future.result())

print("Start future.add_done_callback(got_result)")
future.add_done_callback(got_result)
try:
    loop.run_forever()
finally:
    loop.close()