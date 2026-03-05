import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "IsaacMCP — The AI Copilot for Robotics Simulation",
  description:
    "Seamlessly bridge LLMs with NVIDIA Isaac Sim using the Model Context Protocol. 54 tools, 10 plugins, self-healing simulations.",
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
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
