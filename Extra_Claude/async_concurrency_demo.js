/**
 * Demo: how multiple async functions run concurrently while one is "awaiting".
 * JS equivalent of the Python asyncio demo.
 *
 * Run with: node async_concurrency_demo.js
 */

const startTime = Date.now();

function elapsed() {
  return `${((Date.now() - startTime) / 1000).toFixed(2)}s`;
}

// A "sleep" helper — JS's setTimeout wrapped as a Promise, equivalent to asyncio.sleep()
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function makeTea() {
  console.log(`[${elapsed()}] Starting to boil water for tea...`);
  await sleep(3000); // simulates a slow operation (e.g. an API call)
  console.log(`[${elapsed()}] Tea is ready!`);
  return "tea";
}

async function toastBread() {
  console.log(`[${elapsed()}] Putting bread in toaster...`);
  await sleep(2000);
  console.log(`[${elapsed()}] Toast is ready!`);
  return "toast";
}

async function fryEgg() {
  console.log(`[${elapsed()}] Cracking egg into pan...`);
  await sleep(1000);
  console.log(`[${elapsed()}] Egg is ready!`);
  return "egg";
}

async function checkPhone() {
  console.log(`[${elapsed()}] Checking phone notifications...`);
  await sleep(500);
  console.log(`[${elapsed()}] Done checking phone.`);
  return "phone checked";
}

// ---------------------------------------------------------------------------
// Version A: sequential (each awaited one after another — no concurrency)
// ---------------------------------------------------------------------------
async function sequentialVersion() {
  console.log("\n=== SEQUENTIAL (one after another) ===");

  const tea = await makeTea();       // fully finishes (3s) before toastBread even starts
  const toast = await toastBread();  // fully finishes (2s) before fryEgg even starts
  const egg = await fryEgg();        // fully finishes (1s) before checkPhone even starts
  const phone = await checkPhone();

  console.log(`All done at [${elapsed()}]:`, tea, toast, egg, phone);
  // Total time ≈ 3 + 2 + 1 + 0.5 = 6.5 seconds
}

// ---------------------------------------------------------------------------
// Version B: concurrent (all four start immediately, run "at the same time")
// ---------------------------------------------------------------------------
async function concurrentVersion() {
  console.log("\n=== CONCURRENT (Promise.all) ===");

  // All four functions are CALLED (and start running) immediately here —
  // calling an async function starts executing it right away, up to its
  // first await. await Promise.all(...) only waits for ALL of them to
  // finish, not one-by-one.
  const [tea, toast, egg, phone] = await Promise.all([
    makeTea(),
    toastBread(),
    fryEgg(),
    checkPhone(),
  ]);

  console.log(`All done at [${elapsed()}]:`, tea, toast, egg, phone);
  // Total time ≈ 3 seconds (the slowest one, makeTea), NOT the sum of all four
}

async function main() {
  await sequentialVersion();
  await concurrentVersion();
}

main();
