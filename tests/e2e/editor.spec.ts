import { expect, test } from "@playwright/test";

const pixelPng = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64");

test("uploads a pair, observes progress, downloads, and deletes", async ({ page }) => {
  const asset = (id: string, kind: string) => ({ id, kind, mime_type: "image/png", width: 512, height: 512, byte_size: pixelPng.length, created_at: new Date().toISOString(), expires_at: new Date(Date.now() + 86_400_000).toISOString(), analysis: { quality_score: .9, warnings: [] } });
  let uploadRequests = 0;
  let uploads = 0;
  await page.route("**/api/v1/styles", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/v1/assets/upload", async (route) => {
    uploadRequests += 1;
    if (uploadRequests === 1) {
      await route.fulfill({
        status: 403,
        headers: { "X-CSRF-Token": "rotated-test-token" },
        json: { error: { code: "CSRF_FAILED", message: "The CSRF token is missing or invalid." } },
      });
      return;
    }
    if (uploadRequests === 2) expect(route.request().headers()["x-csrf-token"]).toBe("rotated-test-token");
    uploads += 1;
    await route.fulfill({ json: asset(uploads === 1 ? "input-id" : "reference-id", uploads === 1 ? "INPUT" : "REFERENCE") });
  });
  await page.route("**/api/v1/jobs", (route) => route.fulfill({ json: { id: "job-id", status: "SUCCEEDED", stage: "COMPLETED", progress: 100, input_asset_id: "input-id", reference_asset_id: "reference-id", algorithm_profile: "ai_dgpst_v1", settings: { algorithm_profile: "ai_dgpst_v1", style_strength: .75, structure_strength: .9, inference_steps: 30, random_seed: 0, background_mode: "KEEP", background_color: null, output_format: "PNG", jpeg_quality: 95 }, output_url: "data:image/png;base64," + pixelPng.toString("base64"), created_at: new Date().toISOString() } }));
  await page.route("**/api/v1/jobs/job-id/diagnostics", (route) => route.fulfill({ json: {
    job_id: "job-id",
    diagnostics: { profile: "ai_dgpst_v1", engine: { name: "DGPST", structure_strength: .9 } },
    artifacts: [{ asset_id: "artifact-id", kind: "OTHER", download_url: "/api/v1/jobs/job-id/artifact.png" }],
  } }));
  await page.route("**/api/v1/jobs/job-id/download-url", (route) => route.fulfill({ json: { url: "/api/v1/jobs/job-id/output.png", expires_at: new Date().toISOString(), expires_in_seconds: 300 } }));
  await page.route("**/api/v1/jobs/job-id/output.png", (route) => route.fulfill({
    body: pixelPng,
    contentType: "image/png",
    headers: { "Content-Disposition": 'attachment; filename="portrait-result.png"' },
  }));
  await page.route("**/api/v1/jobs/job-id/artifact.png", (route) => route.fulfill({ body: pixelPng, contentType: "image/png" }));
  await page.route("**/api/v1/jobs/job-id", (route) => route.fulfill({ status: 204 }));

  await page.goto("/");
  const inputs = page.locator('input[type="file"]');
  await inputs.nth(0).setInputFiles({ name: "input.png", mimeType: "image/png", buffer: pixelPng });
  await inputs.nth(1).setInputFiles({ name: "reference.png", mimeType: "image/png", buffer: pixelPng });
  await page.getByRole("button", { name: "Create portrait" }).click();
  await expect(page.getByText("Your finish is ready")).toBeVisible();
  await expect(page.getByText("Style transferred. Review the likeness.")).toBeVisible();
  await page.getByText("Processing diagnostics").click();
  await expect(page.getByText('"name": "DGPST"')).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download PNG" }).click();
  expect((await download).suggestedFilename()).toBe("portrait-result.png");
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: /Delete/ }).last().click();
});

test("does not expose classical correction controls for an AI job", async ({ page }) => {
  const now = new Date().toISOString();
  const succeeded = {
    id: "job-correction",
    status: "SUCCEEDED",
    stage: "COMPLETED",
    progress: 100,
    input_asset_id: "input-id",
    reference_asset_id: "reference-id",
    algorithm_profile: "ai_dgpst_v1",
    settings: { algorithm_profile: "ai_dgpst_v1", style_strength: .75, structure_strength: .9, inference_steps: 30, random_seed: 0, background_mode: "KEEP", background_color: null, output_format: "PNG", jpeg_quality: 95 },
    input_preview_url: "data:image/png;base64," + pixelPng.toString("base64"),
    output_url: "data:image/png;base64," + pixelPng.toString("base64"),
    created_at: now,
    expires_at: new Date(Date.now() + 86_400_000).toISOString(),
  };
  await page.route("**/api/v1/jobs/job-correction/diagnostics", (route) => route.fulfill({ json: { job_id: "job-correction", diagnostics: {}, artifacts: [] } }));
  await page.route("**/api/v1/jobs/job-correction", (route) => route.fulfill({ json: succeeded }));

  await page.goto("/jobs/job-correction");
  await expect(page.getByRole("tab", { name: "Alignment" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Save & rerun" })).toHaveCount(0);
});
