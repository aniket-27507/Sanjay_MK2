export default function Footer() {
  return (
    <footer className="section-technical border-t border-border py-16">
      <div className="divider-check mb-12" />

      <div className="mx-auto max-w-6xl px-6">
        <div className="grid gap-12 sm:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="lg:col-span-2">
            <div
              className="text-sm tracking-[0.12em] uppercase"
              style={{ fontFamily: "var(--font-jetbrains), monospace" }}
            >
              <span className="text-accent">Isaac</span>
              <span className="text-foreground">MCP</span>
            </div>
            <p className="mt-4 max-w-sm text-sm leading-relaxed text-muted">
              The ultimate AI copilot for robotics simulation. Bridge LLMs with
              NVIDIA Isaac Sim using the Model Context Protocol.
            </p>
            <div
              className="mt-6 text-muted"
              style={{
                fontFamily: "var(--font-jetbrains), monospace",
                fontSize: "0.6rem",
                letterSpacing: "0.08em",
              }}
            >
              STATUS: ████████████████ OPERATIONAL
            </div>
          </div>

          {/* Resources */}
          <div>
            <h4
              className="mb-5 text-muted"
              style={{
                fontFamily: "var(--font-jetbrains), monospace",
                fontSize: "0.65rem",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
              }}
            >
              Resources
            </h4>
            <ul className="space-y-3 text-sm">
              <li>
                <a
                  href="https://github.com/yanitedhacker/IsaacMCP"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-accent"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="#install"
                  className="text-muted transition-colors hover:text-accent"
                >
                  Installation Guide
                </a>
              </li>
              <li>
                <a
                  href="#deploy"
                  className="text-muted transition-colors hover:text-accent"
                >
                  Deployment Guide
                </a>
              </li>
            </ul>
          </div>

          {/* Technology */}
          <div>
            <h4
              className="mb-5 text-muted"
              style={{
                fontFamily: "var(--font-jetbrains), monospace",
                fontSize: "0.65rem",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
              }}
            >
              Technology
            </h4>
            <ul className="space-y-3 text-sm">
              <li>
                <a
                  href="https://modelcontextprotocol.io"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-accent"
                >
                  Model Context Protocol
                </a>
              </li>
              <li>
                <a
                  href="https://developer.nvidia.com/isaac-sim"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-accent"
                >
                  NVIDIA Isaac Sim
                </a>
              </li>
              <li>
                <a
                  href="https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/index.html"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-accent"
                >
                  Omniverse Kit API
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 border-t border-border pt-8 text-center">
          <p className="text-sm text-muted">
            Built with care for Robotics Engineers.
          </p>
          <p
            className="mt-3 text-muted"
            style={{
              fontFamily: "var(--font-jetbrains), monospace",
              fontSize: "0.55rem",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
            }}
          >
            DOCUMENT_ID: ISAAC-MCP-001 · REV_003 · © {new Date().getFullYear()}
          </p>
        </div>
      </div>
    </footer>
  );
}
