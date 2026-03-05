export default function Footer() {
  return (
    <footer className="border-t border-border py-16">
      <div className="mx-auto max-w-7xl px-6">
        <div className="grid gap-12 sm:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="lg:col-span-2">
            <div className="text-xl font-bold tracking-tight">
              <span className="text-accent">Isaac</span>MCP
            </div>
            <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted">
              The ultimate AI copilot for robotics simulation. Bridge LLMs with
              NVIDIA Isaac Sim using the Model Context Protocol.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted">
              Resources
            </h4>
            <ul className="space-y-2.5 text-sm">
              <li>
                <a
                  href="https://github.com/yanitedhacker/IsaacMCP"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-foreground"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="#install"
                  className="text-muted transition-colors hover:text-foreground"
                >
                  Installation Guide
                </a>
              </li>
              <li>
                <a
                  href="#deploy"
                  className="text-muted transition-colors hover:text-foreground"
                >
                  Deployment Guide
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted">
              Technology
            </h4>
            <ul className="space-y-2.5 text-sm">
              <li>
                <a
                  href="https://modelcontextprotocol.io"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-foreground"
                >
                  Model Context Protocol
                </a>
              </li>
              <li>
                <a
                  href="https://developer.nvidia.com/isaac-sim"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-foreground"
                >
                  NVIDIA Isaac Sim
                </a>
              </li>
              <li>
                <a
                  href="https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/index.html"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted transition-colors hover:text-foreground"
                >
                  Omniverse Kit API
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 border-t border-border pt-8 text-center text-sm text-muted">
          Built with care for Robotics Engineers.
        </div>
      </div>
    </footer>
  );
}
