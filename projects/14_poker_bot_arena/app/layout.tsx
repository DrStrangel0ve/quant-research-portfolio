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
    title: "Poker Lab — Play the CFR+ Policy",
    description:
      "Play heads-up Leduc Hold'em against a from-scratch CFR+ bot, RLCard's reference checkpoint, and exact or heuristic opponents.",
    icons: {
      icon: "/og.png",
    },
    openGraph: {
      title: "Poker Lab — Play the Policy",
      description: "A trained, exactly audited imperfect-information poker arena.",
      images: [{ url: "/og.png", width: 1200, height: 630, alt: "Poker Lab" }],
      type: "website",
    },
    twitter: {
      card: "summary_large_image",
      title: "Poker Lab — Play the Policy",
      description: "CFR+, exact audit, and six playable opponents.",
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
