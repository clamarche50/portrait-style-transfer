import { startProdServer } from "../../node_modules/vinext/dist/server/prod-server.js";
import { fileURLToPath } from "node:url";

export default async function globalSetup() {
  const { server } = await startProdServer({
    host: "127.0.0.1",
    port: 48731,
    outDir: fileURLToPath(new URL("../../dist", import.meta.url)),
    noCompression: true,
  });

  return async () => {
    await new Promise((resolve) => server.close(resolve));
  };
}
