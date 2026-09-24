"""
Demo: how multiple async functions run concurrently while one is "awaiting".

Run this and watch the print order + timestamps. Each function simulates
a slow operation (like an API call) using asyncio.sleep().
"""

import asyncio
import time

start_time = time.time()


def elapsed():
    return f"{time.time() - start_time:.2f}s"


async def make_tea():
    print(f"[{elapsed()}] Starting to boil water for tea...")
    await asyncio.sleep(3)  # simulates a slow operation (e.g. an API call)
    print(f"[{elapsed()}] Tea is ready!")
    return "tea"


async def toast_bread():
    print(f"[{elapsed()}] Putting bread in toaster...")
    await asyncio.sleep(2)
    print(f"[{elapsed()}] Toast is ready!")
    return "toast"


async def fry_egg():
    print(f"[{elapsed()}] Cracking egg into pan...")
    await asyncio.sleep(1)
    print(f"[{elapsed()}] Egg is ready!")
    return "egg"


async def check_phone():
    print(f"[{elapsed()}] Checking phone notifications...")
    await asyncio.sleep(0.5)
    print(f"[{elapsed()}] Done checking phone.")
    return "phone checked"


# ---------------------------------------------------------------------------
# Version A: sequential (each awaited one after another — no concurrency)
# ---------------------------------------------------------------------------
async def sequential_version():
    print("\n=== SEQUENTIAL (one after another) ===")
    global start_time
    start_time = time.time()

    tea = await make_tea()       # fully finishes (3s) before toast_bread even starts
    toast = await toast_bread()  # fully finishes (2s) before fry_egg even starts
    egg = await fry_egg()        # fully finishes (1s) before check_phone even starts
    phone = await check_phone()

    print(f"All done at [{elapsed()}]:", tea, toast, egg, phone)
    # Total time ≈ 3 + 2 + 1 + 0.5 = 6.5 seconds


# ---------------------------------------------------------------------------
# Version B: concurrent (all four start immediately, run "at the same time")
# ---------------------------------------------------------------------------
async def concurrent_version():
    print("\n=== CONCURRENT (asyncio.gather) ===")
    global start_time
    start_time = time.time()

    # All four coroutines are scheduled and start running right away.
    # await here only waits for ALL of them to finish, not one-by-one.
    tea, toast, egg, phone = await asyncio.gather(
        make_tea(),
        toast_bread(),
        fry_egg(),
        check_phone(),
    )

    print(f"All done at [{elapsed()}]:", tea, toast, egg, phone)
    # Total time ≈ 3 seconds (the slowest one, make_tea), NOT the sum of all four


if __name__ == "__main__":
    asyncio.run(sequential_version())
    asyncio.run(concurrent_version())
