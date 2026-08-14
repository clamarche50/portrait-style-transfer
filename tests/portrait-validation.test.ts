import { describe, expect, it } from "vitest";
import { defaultSettings, settingsSchema, validatePortraitFile } from "@/lib/validation/portrait";

describe("portrait file validation", () => {
  it("accepts a small JPEG", () => {
    const file = new File([new Uint8Array([0xff, 0xd8, 0xff])], "portrait.jpg", { type: "image/jpeg" });
    expect(validatePortraitFile(file)).toBeNull();
  });

  it("rejects SVG input", () => {
    const file = new File(["<svg/ >"], "portrait.svg", { type: "image/svg+xml" });
    expect(validatePortraitFile(file)).toMatch(/JPEG, PNG, or WebP/);
  });

  it("rejects files above 15 MB", () => {
    const file = new File([new Uint8Array(15 * 1024 * 1024 + 1)], "large.png", { type: "image/png" });
    expect(validatePortraitFile(file)).toMatch(/smaller than 15 MB/);
  });
});

describe("transfer settings", () => {
  it("uses the versioned DGPST AI profile", () => {
    expect(settingsSchema.parse(defaultSettings).algorithm_profile).toBe("ai_dgpst_v1");
  });

  it("rejects out-of-range AI controls", () => {
    expect(() => settingsSchema.parse({ ...defaultSettings, style_strength: 2 })).toThrow();
    expect(() => settingsSchema.parse({ ...defaultSettings, structure_strength: -0.1 })).toThrow();
    expect(() => settingsSchema.parse({ ...defaultSettings, inference_steps: 9 })).toThrow();
  });

  it("rejects removed classical-engine settings", () => {
    expect(() => settingsSchema.parse({ ...defaultSettings, dense_alignment: true })).toThrow();
  });

  it("keeps background color payloads consistent with the selected mode", () => {
    expect(defaultSettings.background_color).toBeNull();
    expect(() => settingsSchema.parse({ ...defaultSettings, background_mode: "SOLID" })).toThrow();
    expect(settingsSchema.parse({ ...defaultSettings, background_mode: "SOLID", background_color: "#d7d2c8" }).background_color).toBe("#d7d2c8");
  });
});
