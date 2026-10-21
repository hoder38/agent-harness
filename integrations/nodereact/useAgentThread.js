// React 17 chat hook for agent-harness (no useStream; plain fetch + EventSource).
import { useCallback, useEffect, useRef, useState } from 'react';

let cached = { token: null, exp: 0 };
async function getToken() {
  if (cached.token && Date.now() < cached.exp - 60_000) return cached.token;
  const r = await fetch('/api/agent-token', { credentials: 'include' });
  if (!r.ok) throw new Error('agent token: ' + r.status);
  const { token, expiresIn } = await r.json();
  cached = { token, exp: Date.now() + expiresIn * 1000 };
  return token;
}

async function api(path, opts = {}) {
  const token = await getToken();
  const r = await fetch('/api/agent' + path, {
    ...opts,
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + token, ...(opts.headers || {}) },
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

export function useAgentThread(threadId) {
  const [messages, setMessages] = useState([]);
  const [run, setRun] = useState(null); // { id, decision, status }
  const esRef = useRef(null);

  useEffect(() => {
    api(`/threads/${threadId}`).then((t) => setMessages(t.messages)).catch(console.error);
    return () => esRef.current && esRef.current.close();
  }, [threadId]);

  const send = useCallback(async (text, { mode = 'interactive' } = {}) => {
    setMessages((m) => [...m, { role: 'user', content: text }, { role: 'assistant', content: '' }]);
    const res = await api(`/threads/${threadId}/runs`, {
      method: 'POST',
      body: JSON.stringify({ text, mode, client_msg_id: `${threadId}-${Date.now()}` }),
    });
    if (res.command_reply) {
      setMessages((m) => [...m.slice(0, -1), { role: 'assistant', content: res.command_reply }]);
      return;
    }
    setRun({ id: res.run_id, decision: res.decision, status: 'running' });
    const token = await getToken();
    const es = new EventSource(`/api/agent/runs/${res.run_id}/stream?token=${encodeURIComponent(token)}`);
    esRef.current = es;
    const appendToLast = (fn) =>
      setMessages((m) => [...m.slice(0, -1), { ...m[m.length - 1], content: fn(m[m.length - 1].content) }]);
    es.addEventListener('token', (e) => appendToLast((c) => c + JSON.parse(e.data)));
    es.addEventListener('decision', (e) => setRun((r) => ({ ...r, decision: JSON.parse(e.data) })));
    es.addEventListener('status', (e) => setRun((r) => ({ ...r, status: JSON.parse(e.data).status })));
    const finish = (status) => (e) => {
      const data = e.data ? JSON.parse(e.data) : {};
      if (status === 'done') appendToLast(() => data.text);
      if (status === 'error') appendToLast((c) => c + `\n\n⚠️ ${data.message}`);
      setRun((r) => ({ ...r, status }));
      es.close();
    };
    es.addEventListener('done', finish('done'));
    es.addEventListener('error', (e) => (e.data ? finish('error')(e) : null)); // ignore transport errors (auto-retry)
    es.addEventListener('cancelled', finish('cancelled'));
  }, [threadId]);

  const cancel = useCallback(() => run && api(`/runs/${run.id}/cancel`, { method: 'POST' }), [run]);
  const setKillSwitch = (state) => api('/killswitch', { method: 'PUT', body: JSON.stringify({ state }) });

  return { messages, run, send, cancel, setKillSwitch };
}
