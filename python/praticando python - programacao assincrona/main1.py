import asyncio


async def process1(future):
    print("Executando process1")
    await asyncio.sleep(2)
    future.set_result("process1 data")
    print("Fim process1")


async def process2(future):
    print("Executando process2")
    result = await future
    print("Fim process2 - result: {}".format(result))


async def main():
    future = asyncio.Future()
    task1 = asyncio.create_task(process1(future))
    task2 = asyncio.create_task(process2(future))

    await task1
    await task2

asyncio.run(main())
