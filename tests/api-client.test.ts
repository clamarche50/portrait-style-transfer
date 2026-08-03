import { beforeEach, describe, expect, it, vi } from "vitest";

import { addStyleExample, createJob, getJob } from "@/lib/api/client";
import { defaultSettings } from "@/lib/validation/portrait";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("API client security and contracts", () => {
  beforeEach(() => {
    document.cookie = "pst_csrf=token%2Fwith%2Fslashes; path=/";
  });

  it("echoes the readable CSRF cookie on unsafe requests only", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ id: "job-1" }))
      .mockResolvedValueOnce(jsonResponse({ id: "job-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await createJob({
      input_asset_id: "input-1",
      reference_asset_id: "reference-1",
      settings: defaultSettings,
    });
    await getJob("job-1");

    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("X-CSRF-Token"))
      .toBe("token/with/slashes");
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).has("X-CSRF-Token"))
      .toBe(false);
  });

  it("uploads a style example as an asset before attaching its id", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ id: "asset-1" }))
      .mockResolvedValueOnce(jsonResponse({ id: "example-1" }));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File([new Uint8Array([1, 2, 3])], "reference.png", { type: "image/png" });

    await addStyleExample("style-1", file);

    expect(fetchMock.mock.calls[0][0]).toContain("/assets/upload");
    expect(fetchMock.mock.calls[0][1]?.body).toBeInstanceOf(FormData);
    expect(fetchMock.mock.calls[1][0]).toContain("/styles/style-1/examples");
    expect(fetchMock.mock.calls[1][1]?.body).toBe(JSON.stringify({ asset_id: "asset-1" }));
  });
});
