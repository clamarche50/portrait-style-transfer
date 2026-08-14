import { spawn } from "node:child_process";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const host = "127.0.0.1";
const port = 48731;
const origin = `http://${host}:${port}`;

export default async function globalSetup() {
  const projectRoot = fileURLToPath(new URL("../../", import.meta.url));
  const nextCli = fileURLToPath(
    new URL("../../node_modules/next/dist/bin/next", import.meta.url),
  );
  const server = spawn(
    process.execPath,
    [nextCli, "start", "--hostname", host, "--port", String(port)],
    {
      cwd: projectRoot,
      env: { ...process.env, NODE_ENV: "production" },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let output = "";
  server.stdout.on("data", (chunk) => {
    output = `${output}${chunk}`.slice(-8_000);
  });
  server.stderr.on("data", (chunk) => {
    output = `${output}${chunk}`.slice(-8_000);
  });

  const deadline = Date.now() + 30_000;
  let ready = false;
  while (Date.now() < deadline) {
    if (server.exitCode !== null) {
      throw new Error(`Next.js exited before Playwright started:\n${output}`);
    }
    try {
      const response = await fetch(origin, { signal: AbortSignal.timeout(1_000) });
      if (response.ok) {
        ready = true;
        break;
      }
    } catch {
      // The server is still starting.
    }
    await delay(250);
  }
  if (!ready) {
    server.kill();
    throw new Error(`Next.js did not become ready within 30 seconds:\n${output}`);
  }

  return async () => {
    if (server.exitCode === null) {
      server.kill();
      await Promise.race([once(server, "exit"), delay(5_000)]);
    }
  };
}
