import { z } from "zod";
import type { TransferSettings } from "@/lib/api/types";

export const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
export const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;

export const portraitFileSchema = z
  .instanceof(File)
  .refine((file) => ALLOWED_IMAGE_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_TYPES)[number]), "Use a JPEG, PNG, or WebP image.")
  .refine((file) => file.size <= MAX_UPLOAD_BYTES, "Choose an image smaller than 15 MB.");

export const settingsSchema = z.object({
  algorithm_profile: z.literal("ai_dgpst_v1"),
  style_strength: z.number().min(0).max(1),
  structure_strength: z.number().min(0).max(1),
  inference_steps: z.number().int().min(10).max(50),
  random_seed: z.number().int().min(0).max(2_147_483_647),
  background_mode: z.enum(["KEEP", "BLUR", "SOLID", "REFERENCE"]),
  background_color: z.string().regex(/^#[0-9a-f]{6}$/i).nullable(),
  output_format: z.enum(["PNG", "JPEG"]),
  jpeg_quality: z.number().int().min(70).max(100),
}).strict().superRefine((settings, context) => {
  if (settings.background_mode === "SOLID" && settings.background_color === null) {
    context.addIssue({ code: "custom", path: ["background_color"], message: "Choose a solid background color." });
  }
  if (settings.background_mode !== "SOLID" && settings.background_color !== null) {
    context.addIssue({ code: "custom", path: ["background_color"], message: "A background color is only valid in Color mode." });
  }
});

export const defaultSettings: TransferSettings = {
  algorithm_profile: "ai_dgpst_v1",
  style_strength: 0.75,
  structure_strength: 0.9,
  inference_steps: 30,
  random_seed: 0,
  background_mode: "KEEP",
  background_color: null,
  output_format: "PNG",
  jpeg_quality: 95,
};

export function validatePortraitFile(file: File): string | null {
  const parsed = portraitFileSchema.safeParse(file);
  return parsed.success ? null : parsed.error.issues[0]?.message ?? "This image is not supported.";
}
