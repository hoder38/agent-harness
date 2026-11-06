# agent-harness agent instructions

Spec and status: `ARCHITECTURE.md`. Usage: `README.md`.

## Working on nodeReact (the web front end)

nodeReact is a **separate git repo** at `/home/hoder/nodeReact` (compose project `anomopi`, nginx network
`anomopi_default`). It hosts the agent pages (chat, `/Memory`, `/Todos`, `/Inbox`). Changes there are committed
in that repo, never in agent-harness. Its own rules live in `/home/hoder/nodeReact/AGENTS.md`; this section
mirrors them, so update both when they change.

**Toolchain**
- Run every build, test and command for nodeReact inside a Node 14 container (`node:14-bullseye-slim`, git
  installed, `npm ci --legacy-peer-deps`). Never use the host's Node 22: it gives misleading failures (webpack
  chunk conflict, OpenSSL errors, babel mismatch).
- Do not build or test inside the running `*-release` containers or write to `public/`; use a throwaway copy
  (e.g. under `/tmp`) mounted into a temporary container.
- Front-end tests: `npx jest --config jest.front.cjs --forceExit src/front` inside the Node 14 container (18
  suites / 1331 tests passed at phase 7 with the lockfile as-is; no babel upgrade is needed).
- Release build: `npm run build` in the same container (emits `release.js`).
- `package-lock.json` is lockfileVersion 1. Do not upgrade babel or add packages that rewrite it to v3 or need
  Node > 14; plain `npm ci` must keep working.

**Stack limits**
- React 17 and chart.js **2.9** (v2 API, `import Chart from 'chart.js'`, as in `src/front/components/StockInfo.js`).
  Prefer what is already installed over new dependencies.

**Adding an agent page** (pattern from the Todos/Inbox commit)
- Component in `src/front/components/<Name>.js`.
- Path constant in `src/front/constants.js` (`export const <NAME>_PAGE = '/<Name>'`).
- `<Route path={<NAME>_PAGE} component={<Name>} />` in `src/front/components/App.js`.
- Menu entry (`title`, `hash`, glyphicon `css`, next free `key`) in `src/back/controllers/basic-router.js`.
- Call the harness only through `src/front/agent.js` (`agentApi`, `getAgentToken`, `useAgentThread`); the browser
  talks to `/api/agent/*` via nginx with a short-lived JWT (ARCHITECTURE.md §4a).
- Tests next to the others in `src/front/*.test.js`.
- Leave unrelated features (storage, password, stock, Bitfinex, …) untouched.

**Harness side**
- Reference copies of the integration live in `integrations/nodereact/` (`agentToken.js`, `useAgentThread.js`,
  `nginx-agent.conf`). Keep them in step when the contract changes.
- A new page usually needs a new `/api/agent/*` route in `api/app/routes/`; never send PRIVATE content to pages
  that are not owner-only.
