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

// pending: the agent asked a clarifying question (§12a) and waits; answer with resume(text) or skip().
export function useAgentThread(threadId) {
  const [messages, setMessages] = useState([]);
  const [run, setRun] = useState(null); // { id, decision, status }
  const [pending, setPending] = useState(null); // { run_id, question, options, round }
  const esRef = useRef(null);

  useEffect(() => {
    api(`/threads/${threadId}`)
      .then((t) => {
        setMessages(t.messages);
        setPending(t.pending);
      })
      .catch(console.error);
    return () => esRef.current && esRef.current.close();
  }, [threadId]);

  const follow = useCallback(async (res) => {
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
    // The answer is being rewritten (failed citation check, or moved off the GPU): drop the streamed text.
    es.addEventListener('reset', () => appendToLast(() => ''));
    es.addEventListener('decision', (e) => setRun((r) => ({ ...r, decision: JSON.parse(e.data) })));
    es.addEventListener('status', (e) => setRun((r) => ({ ...r, status: JSON.parse(e.data).status })));
    const finish = (status) => (e) => {
      const data = e.data ? JSON.parse(e.data) : {};
      // done.text is the full answer incl. the Sources / Unverified tail, so it replaces the streamed text.
      if (status === 'done') appendToLast(() => data.text);
      if (status === 'error') appendToLast((c) => c + `\n\n⚠️ ${data.message}`);
      if (status === 'awaiting_input') {
        const opts = (data.options || []).map((o, i) => `\n${i + 1}. ${o}`).join('');
        appendToLast(() => data.question + opts);
        setPending({ run_id: data.run_id, question: data.question, options: data.options || [], round: data.round });
      }
      setRun((r) => ({ ...r, status }));
      es.close();
    };
    es.addEventListener('done', finish('done'));
    es.addEventListener('interrupt', finish('awaiting_input'));
    es.addEventListener('error', (e) => (e.data ? finish('error')(e) : null)); // ignore transport errors (auto-retry)
    es.addEventListener('cancelled', finish('cancelled'));
  }, []);

  const send = useCallback(async (text, { mode = 'interactive' } = {}) => {
    setMessages((m) => [...m, { role: 'user', content: text }, { role: 'assistant', content: '' }]);
    setPending(null); // a message while a question is open is its answer (the server routes it)
    const res = await api(`/threads/${threadId}/runs`, {
      method: 'POST',
      body: JSON.stringify({ text, mode, client_msg_id: `${threadId}-${Date.now()}` }),
    });
    await follow(res);
  }, [threadId, follow]);

  // Answer the open question (an option string or free text); empty/null = skip.
  const resume = useCallback(async (value) => {
    if (!pending) return;
    setMessages((m) => [...m, { role: 'user', content: value || '/skip' }, { role: 'assistant', content: '' }]);
    const runId = pending.run_id;
    setPending(null);
    const res = await api(`/runs/${runId}/resume`, { method: 'POST', body: JSON.stringify({ value: value || null }) });
    await follow(res);
  }, [pending, follow]);
  const skip = useCallback(() => resume(null), [resume]);

  const cancel = useCallback(async () => {
    const id = pending ? pending.run_id : run && run.id;
    if (!id) return;
    setPending(null);
    await api(`/runs/${id}/cancel`, { method: 'POST' });
  }, [run, pending]);
  const setKillSwitch = (state) => api('/killswitch', { method: 'PUT', body: JSON.stringify({ state }) });

  return { messages, run, pending, send, resume, skip, cancel, setKillSwitch };
}
