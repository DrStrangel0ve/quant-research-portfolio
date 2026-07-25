import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host");
  const protocol = requestHeaders.get("x-forwarded-proto") ?? "https";
  const metadataBase = new URL(host ? `${protocol}://${host}` : "https://poker-lab.invalid");
  return {
    metadataBase,
    title: "Poker Lab — Exact CFR+ and Neural No-Limit",
    description:
      "Play exact Leduc CFR+ or Royal Micro Hold'em against neural CFR, belief-aware search, and interpretable baselines.",
    icons: {
      icon: "/og.png",
    },
    openGraph: {
      title: "Poker Lab — Play the Policy",
      description: "Exact small-game audit meets neural no-limit poker research.",
      images: [{ url: "/og.png", width: 1200, height: 630, alt: "Poker Lab" }],
      type: "website",
    },
    twitter: {
      card: "summary_large_image",
      title: "Poker Lab — Exact and Neural Poker",
      description: "CFR+, neural CFR, range search, and eleven playable opponents.",
      images: ["/og.png"],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
