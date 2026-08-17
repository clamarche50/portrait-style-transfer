import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";
import { Providers } from "./providers";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto");
  const protocol = forwardedProtocol === "http" || forwardedProtocol === "https"
    ? forwardedProtocol
    : host.startsWith("localhost") || host.startsWith("127.0.0.1") ? "http" : "https";
  const metadataBase = new URL(process.env.NEXT_PUBLIC_APP_URL ?? `${protocol}://${host}`);
  return {
    metadataBase,
    title: {
      default: "Portrait Studio — local, multiscale headshot style transfer",
      template: "%s · Portrait Studio",
    },
    description: "Transfer the light, tone, texture, and color of a reference headshot while preserving the input portrait.",
    openGraph: {
      title: "Portrait Studio",
      description: "Move the light. Keep the person.",
      type: "website",
      images: [{ url: "/og.png", width: 1734, height: 909, alt: "Portrait Studio — Move the light. Keep the person." }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Portrait Studio",
      description: "Move the light. Keep the person.",
      images: ["/og.png"],
    },
    icons: { icon: "/og.png", shortcut: "/og.png" },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <a className="skip-link" href="#main-content">Skip to the editor</a>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
