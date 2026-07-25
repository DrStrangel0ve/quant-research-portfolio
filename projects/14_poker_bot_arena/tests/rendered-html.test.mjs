import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html", host: "localhost" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the finished poker arena and metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Poker Lab — Play the CFR\+ Policy<\/title>/i);
  assert.match(html, /Play the policy/);
  assert.match(html, /CFR\+ 20K/);
  assert.match(html, /RLCard CFR/);
  assert.match(html, /Exact Punisher/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("ships a real trained policy and bespoke social preview", async () => {
  const policyUrl = new URL("../public/poker-policies.json", import.meta.url);
  const socialUrl = new URL("../public/og.png", import.meta.url);
  const payload = JSON.parse(await readFile(policyUrl, "utf8"));
  assert.equal(payload.format, "quantlab-leduc-arena-v1");
  assert.equal(Object.keys(payload.trained.policy).length, 288);
  assert.equal(Object.keys(payload.rlcard_reference.policy).length, 84);
  await access(socialUrl);
});
