// nodeReact (Express 4) — GET /api/agent-token  (ARCHITECTURE.md §4a)
// Mount behind the existing owner-session auth middleware, e.g.:
//   app.get('/api/agent-token', requireOwnerSession, agentToken);
// Requires `jsonwebtoken` and AGENT_JWT_SECRET / AGENT_OWNER_ID in nodeReact's .env.
const crypto = require('crypto');
const jwt = require('jsonwebtoken');

module.exports = function agentToken(req, res) {
  const secret = process.env.AGENT_JWT_SECRET;
  if (!secret) return res.status(500).json({ error: 'AGENT_JWT_SECRET not configured' });
  const token = jwt.sign(
    { sub: process.env.AGENT_OWNER_ID || 'owner', jti: crypto.randomUUID() },
    secret,
    { algorithm: 'HS256', audience: 'agent-harness', issuer: 'nodereact', expiresIn: '10m' },
  );
  res.set('Cache-Control', 'no-store').json({ token, expiresIn: 600 });
};
