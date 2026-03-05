"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const links = [
  { label: "Features", href: "#features" },
  { label: "Packs", href: "#packs" },
  { label: "Quick Start", href: "#quickstart" },
  { label: "Install", href: "#install" },
  { label: "Deploy", href: "#deploy" },
  { label: "FAQ", href: "#faq" },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 z-50 w-full transition-all duration-300 ${scrolled
          ? "border-b border-border bg-background/90 backdrop-blur-md"
          : "bg-transparent"
        }`}
      style={{ zIndex: 50 }}
    >
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        {/* Logo */}
        <a href="#" className="flex items-center gap-1.5 group">
          <span
            className="text-xs tracking-[0.15em] uppercase"
            style={{ fontFamily: "var(--font-jetbrains), monospace" }}
          >
            <span className="text-accent group-hover:text-accent-hover transition-colors">
              Isaac
            </span>
            <span className="text-foreground">MCP</span>
          </span>
          <span className="ml-2 text-[0.55rem] text-muted tracking-[0.1em] uppercase hidden sm:inline"
            style={{ fontFamily: "var(--font-jetbrains), monospace" }}
          >
            v2.0
          </span>
        </a>

        {/* Desktop Links */}
        <div className="hidden items-center gap-6 md:flex">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-sm text-muted transition-colors hover:text-accent"
              style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: "0.65rem", letterSpacing: "0.08em", textTransform: "uppercase" }}
            >
              {l.label}
            </a>
          ))}
          <a
            href="https://github.com/yanitedhacker/IsaacMCP"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-bevel"
          >
            GitHub ↗
          </a>
        </div>

        {/* Mobile Hamburger */}
        <button
          onClick={() => setOpen(!open)}
          className="flex flex-col gap-1.5 md:hidden"
          aria-label="Toggle menu"
        >
          <span
            className={`block h-[1.5px] w-5 bg-foreground transition-transform duration-300 ${open ? "translate-y-[7px] rotate-45" : ""
              }`}
          />
          <span
            className={`block h-[1.5px] w-5 bg-foreground transition-opacity duration-300 ${open ? "opacity-0" : ""
              }`}
          />
          <span
            className={`block h-[1.5px] w-5 bg-foreground transition-transform duration-300 ${open ? "-translate-y-[7px] -rotate-45" : ""
              }`}
          />
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden border-t border-border bg-background/95 backdrop-blur-md md:hidden"
          >
            <div className="flex flex-col gap-3 px-6 py-5">
              {links.map((l) => (
                <a
                  key={l.href}
                  href={l.href}
                  onClick={() => setOpen(false)}
                  className="text-sm text-muted transition-colors hover:text-accent"
                  style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: "0.7rem", letterSpacing: "0.08em", textTransform: "uppercase" }}
                >
                  {l.label}
                </a>
              ))}
              <a
                href="https://github.com/yanitedhacker/IsaacMCP"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-bevel mt-2 text-center"
              >
                GitHub ↗
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
