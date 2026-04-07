import asyncio


async def process(name, time):
    print(f"Tarefa {name} iniciada")
    await asyncio.sleep(time)
    print(f"Tarefa {name} concluída\n")


async def main():
    await asyncio.gather(
        process("A1", 2),
        process("A2", 1),
        process("A3", 3),
        process("A4", 0),
        process("A5", 0))


asyncio.run(main())
