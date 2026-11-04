// Before testing a deployed copy (E2E_BASE_URL), wake its API: the free server
// sleeps when unused and takes up to a minute to start. Does nothing locally.
export default async function wakeRemote() {
  const remote = process.env.E2E_BASE_URL;
  if (!remote) return;
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${remote.replace(/\/$/, "")}/api/health`, { signal: AbortSignal.timeout(90_000) });
      if (res.ok) return;
    } catch {
      // still waking
    }
    await new Promise((resolve) => setTimeout(resolve, 3_000));
  }
  throw new Error(`${remote}/api/health didn't answer within 3 minutes`);
}
