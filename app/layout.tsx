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
      default: "Portrait Studio — AI portrait style transfer",
      template: "%s · Portrait Studio",
    },
    description: "Guide a headshot's color, lighting, and texture with a reference portrait using InstantStyle. Review every AI-generated edit for likeness changes.",
    openGraph: {
      title: "Portrait Studio",
      description: "Move the light. Shape the finish. Review the AI-generated result.",
      type: "website",
    },
    twitter: {
      card: "summary",
      title: "Portrait Studio",
      description: "Move the light. Shape the finish. Review the AI-generated result.",
    },
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
