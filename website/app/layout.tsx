import type { Metadata } from "next";
import { EB_Garamond, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const garamond = EB_Garamond({
  variable: "--font-garamond",
  subsets: ["latin"],
  display: "swap",
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "IsaacMCP — The AI Copilot for Robotics Simulation",
  description:
    "Seamlessly bridge LLMs with NVIDIA Isaac Sim using the Model Context Protocol. 80+ tools, 12 plugins, self-healing simulations.",
  openGraph: {
    title: "IsaacMCP — The AI Copilot for Robotics Simulation",
    description:
      "Bridge AI coding assistants with NVIDIA Isaac Sim via MCP. Intelligent diagnostics, autonomous fixes, experiment engines, and more.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${garamond.variable} ${inter.variable} ${jetbrainsMono.variable} antialiased`}
        style={{
          fontFamily: "var(--font-garamond), Georgia, serif",
        }}
      >
        {children}
      </body>
    </html>
  );
}
