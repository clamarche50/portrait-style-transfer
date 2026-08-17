import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SettingsPanel } from "@/components/editor/SettingsPanel";
import { defaultSettings } from "@/lib/validation/portrait";

describe("classical transfer settings panel", () => {
  it("shows the Source 2014 badge and classical-engine controls", () => {
    const onChange = vi.fn();
    render(<SettingsPanel settings={defaultSettings} onChange={onChange} />);

    expect(screen.getByText("Source 2014")).toBeTruthy();
    expect(screen.getByText("Texture & contrast")).toBeTruthy();
    expect(screen.getByText("Broad lighting")).toBeTruthy();
    expect(screen.getByText("Global range")).toBeTruthy();
    expect(screen.getByText("Eye highlights")).toBeTruthy();
    expect(screen.getByText("Dense alignment")).toBeTruthy();
    expect(screen.queryByText(/Style strength/)).toBeNull();
    expect(screen.queryByText(/Inference steps/)).toBeNull();

    const [textureSlider] = screen.getAllByRole("slider");
    fireEvent.change(textureSlider, {
      target: { value: "0.5" },
    });
    expect(onChange).toHaveBeenLastCalledWith({ ...defaultSettings, transfer_strength: 0.5 });
  });

  it("toggles background modes and exposes the solid color picker", () => {
    const onChange = vi.fn();
    render(<SettingsPanel settings={defaultSettings} onChange={onChange} />);

    fireEvent.click(screen.getByRole("button", { name: "Color" }));
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ background_mode: "SOLID", background_color: "#d7d2c8" }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Ref." }));
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ background_mode: "REFERENCE", background_color: null }),
    );
  });

  it("switches the output format without exposing AI-era controls", () => {
    const onChange = vi.fn();
    render(
      <SettingsPanel
        settings={{ ...defaultSettings, output_format: "JPEG" }}
        onChange={onChange}
      />,
    );

    expect(screen.getByRole("button", { name: "JPEG" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "PNG" })).toBeTruthy();
    expect(screen.queryByRole("spinbutton", { name: /Random seed/ })).toBeNull();
    expect(screen.queryByRole("slider", { name: /JPEG quality/ })).toBeNull();
  });
});
