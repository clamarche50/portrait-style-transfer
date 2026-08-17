import assert from "node:assert/strict";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the product editor with final metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Portrait Studio/);
  assert.match(html, /Move the light/);
  assert.match(html, /Keep the person/);
  assert.match(html, /Create portrait/);
  assert.match(html, /Nothing uploads until you click Create portrait/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("server-renders privacy and library routes", async () => {
  const [privacy, styles] = await Promise.all([render("/privacy"), render("/styles")]);
  assert.equal(privacy.status, 200);
  assert.equal(styles.status, 200);
  assert.match(await privacy.text(), /Your face is not/);
  assert.match(await styles.text(), /Build a language/);
});
