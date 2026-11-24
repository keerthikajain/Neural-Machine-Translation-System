import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const apiOrigin = typeof SAARTHI_API_ORIGIN === "string" ? SAARTHI_API_ORIGIN : "";

const pages = [
  { id: "translate-view", heading: "translate-heading", icon: "translate", label: "Translate" },
  { id: "history-view", heading: "history-heading", icon: "history", label: "History" },
  { id: "phrases-view", heading: "phrases-heading", icon: "phrases", label: "Quick Phrases" },
  { id: "settings-view", heading: "settings-heading", icon: "settings", label: "Settings" },
];
const phrases = [
  "Good morning!",
  "How are you?",
  "Thank you so much.",
  "Where are you going?",
  "I would like to visit India.",
  "Have a nice day!",
  "Where is the nearest railway station?",
  "The train leaves at 8 in the morning.",
];
const storageKey = "saarthi.history.v1";
const historyLimit = 30;
const storageError = "Browser storage is unavailable. Check your browser settings to manage saved translations.";

function IconLibrary() {
  return (
    <svg className="icon-library" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
      <symbol id="i-translate" viewBox="0 0 24 24"><path d="M3 4h10a2 2 0 0 1 2 2v5M3 4a1 1 0 0 0-1 1v9a2 2 0 0 0 2 2h5l3 3v-3m-7-3 3-6 3 6m-5-2h4m4 1h8m-4-3v3m-3 3c1.5 2.5 3.5 4 6 5m-1-8c-1 4-3 6-6 8"/></symbol>
      <symbol id="i-history" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 1 2.6 6.4M3 5v7h7m2-5v5l3 2"/></symbol>
      <symbol id="i-phrases" viewBox="0 0 24 24"><path d="M3 5h7l2 2 2-2h7v14h-7l-2 2-2-2H3zm9 2v14M6 9h3m-3 4h3m6-4h3m-3 4h3"/></symbol>
      <symbol id="i-settings" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="m10 3-1 3-2 1-3-.5L2.5 9l2 2v2l-2 2L4 17.5l3-.5 2 1 1 3h4l1-3 2-1 3 .5 1.5-2.5-2-2v-2l2-2L20 6.5l-3 .5-2-1-1-3z"/></symbol>
      <symbol id="i-arrow" viewBox="0 0 24 24"><path d="M4 12h15m-6-6 6 6-6 6"/></symbol>
      <symbol id="i-send" viewBox="0 0 24 24"><path d="m22 2-7 20-4-9-9-4zm0 0L11 13"/></symbol>
      <symbol id="i-copy" viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></symbol>
    </svg>
  );
}

function Icon({ name }) {
  return <svg aria-hidden="true"><use href={`#i-${name}`} /></svg>;
}

function Sidebar({ active }) {
  return (
    <aside className="sidebar">
      <a className="sidebar-brand" href="#translate-view"><img src="/logo.svg" alt="" width="38" height="38" />Saarthi</a>
      <nav className="navigation" aria-label="Main navigation">
        {pages.map(({ id, icon, label }) => (
          <a key={id} href={`#${id}`} aria-current={active === id ? "page" : undefined}>
            <Icon name={icon} />{label}
          </a>
        ))}
      </nav>
      <p className="sidebar-note">Runs on your device</p>
    </aside>
  );
}

function Masthead() {
  return (
    <header className="masthead">
      <div className="masthead-brand">
        <img src="/logo.svg" alt="" width="64" height="64" />
        <div><span className="wordmark">Saarthi</span><p>Bridging languages, building connections.</p></div>
      </div>
    </header>
  );
}

function HistoryList({ id, entries }) {
  return (
    <ul id={id} className="history-list">
      {entries.map(({ source, translation }, index) => (
        <li className="history-entry" key={index}>
          <p>{source}</p><p lang="hi">{translation}</p>
        </li>
      ))}
    </ul>
  );
}

function TranslateView({ hidden, text, output, busy, status, entries, sourceRef, outputRef, onText, onClear, onSubmit, onCopy }) {
  return (
    <section id="translate-view" className="view" aria-labelledby="translate-heading" hidden={hidden}>
      <h1 id="translate-heading" tabIndex={-1}>Translate</h1>
      <div className="dashboard">
        <form id="translator" noValidate onSubmit={onSubmit}>
          <div className="direction" aria-label="Translation direction">
            <span className="language"><span className="language-mark" aria-hidden="true">EN</span><span><strong>English</strong><small>Source</small></span></span>
            <svg className="direction-arrow" role="img" aria-label="to"><use href="#i-arrow" /></svg>
            <span className="language"><span className="language-mark" lang="hi" aria-hidden="true">हि</span><span><strong>Hindi</strong><small>Target</small></span></span>
          </div>
          <div className="panels">
            <section className="source card">
              <div className="panel-header">
                <label htmlFor="source">English text</label>
                <button id="clear" className="quiet" type="button" disabled={busy} onClick={onClear}>Clear</button>
              </div>
              <textarea id="source" name="text" ref={sourceRef} value={text} onChange={event => onText(event.target.value)}
                maxLength={1200} spellCheck placeholder="Type or paste your text here" aria-describedby="count" required readOnly={busy} />
              <span id="count" className="count">{`${text.length.toLocaleString("en")} / 1,200`}</span>
            </section>
            <div className="translation-action">
              <button className="primary" id="translate" type="submit" disabled={busy}>
                <Icon name="send" /><span>{busy ? "Translating…" : "Translate"}</span>
              </button>
              <p id="status" role="status" className={status.error ? "error" : undefined}>{status.message}</p>
            </div>
            <section className="target card" aria-labelledby="target-label" aria-busy={busy}>
              <div className="panel-header">
                <h2 id="target-label">Hindi translation</h2>
                <button id="copy" className="quiet" type="button" disabled={!output} onClick={onCopy}><Icon name="copy" />Copy</button>
              </div>
              <div id="output" className="translation" lang="hi" tabIndex={0} ref={outputRef} hidden={!output}>{output}</div>
              <div id="placeholder" className="placeholder" hidden={Boolean(output)}>
                <Icon name="translate" /><p>Your Hindi translation will appear here.</p>
              </div>
            </section>
          </div>
        </form>
        <section className="card recent-card" aria-labelledby="recent-heading">
          <div className="side-heading"><h2 id="recent-heading">Recent Translations</h2><a href="#history-view">See all <span aria-hidden="true">→</span></a></div>
          <HistoryList id="recent-list" entries={entries.slice(0, 3)} />
          <p id="recent-empty" className="empty" hidden={entries.length > 0}>Your recent translations will appear here.</p>
        </section>
      </div>
    </section>
  );
}

function HistoryView({ hidden, entries, onDelete }) {
  const [query, setQuery] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [error, setError] = useState(false);
  const search = useRef(null);
  const clear = useRef(null);
  const remove = useRef(null);

  useLayoutEffect(() => {
    if (hidden) {
      setConfirm(false);
      setError(false);
    } else if (confirm) {
      remove.current.focus();
    }
  }, [hidden, confirm]);

  const needle = query.trim().toLowerCase();
  const matches = entries.filter(entry => entry.source.toLowerCase().includes(needle) || entry.translation.includes(needle));
  function deleteHistory() {
    if (!onDelete()) {
      setError(true);
      return;
    }
    setQuery("");
    setConfirm(false);
    setError(false);
    search.current.focus();
  }

  return (
    <section id="history-view" className="view" aria-labelledby="history-heading" hidden={hidden}>
      <h1 id="history-heading" tabIndex={-1}>Translation History</h1>
      <div className="card feature-card">
        <div className="history-tools">
          <label>Search history <input id="history-search" ref={search} type="search" placeholder="Search English or Hindi" value={query} onChange={event => setQuery(event.target.value)} /></label>
          <button id="history-clear" className="quiet" type="button" ref={clear} disabled={!entries.length}
            onClick={() => setConfirm(true)}>Clear history</button>
        </div>
        <div id="history-confirm" className="confirm" hidden={!confirm}>
          <p>Remove all translations from this browser’s history?</p>
          <button id="history-delete" className="quiet" type="button" ref={remove} onClick={deleteHistory}>Remove all</button>
          <button id="history-cancel" className="quiet" type="button" onClick={() => {
            setConfirm(false);
            setError(false);
            clear.current.focus();
          }}>Cancel</button>
        </div>
        <HistoryList id="history-list" entries={matches} />
        <p id="history-empty" className="empty" role="status" hidden={!error && matches.length > 0}>
          {error ? "Could not clear saved history. Check your browser settings." :
            entries.length ? "No translations match your search." : "You have no translations yet."}
        </p>
      </div>
    </section>
  );
}

function PhrasesView({ hidden, busy, onPhrase }) {
  return (
    <section id="phrases-view" className="view" aria-labelledby="phrases-heading" hidden={hidden}>
      <h1 id="phrases-heading" tabIndex={-1}>Quick Phrases</h1>
      <p className="view-intro">Choose a phrase to fill the English input.</p>
      <div className="card feature-card">
        <ul id="phrase-list" className="phrase-list">
          {phrases.map(phrase => (
            <li key={phrase}><button type="button" className="phrase" disabled={busy} onClick={() => onPhrase(phrase)}>
              {phrase}<span aria-hidden="true">→</span>
            </button></li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function SettingsView({ hidden, saving, error, onSave }) {
  return (
    <section id="settings-view" className="view" aria-labelledby="settings-heading" hidden={hidden}>
      <h1 id="settings-heading" tabIndex={-1}>Settings</h1>
      <div className="card feature-card settings-card">
        <h2>Translation history</h2>
        <p id="history-setting-help">History lasts until you reload or close this tab by default. Enable saving to keep up to 30 translations in this browser across visits. Turning it off removes the saved copy.</p>
        <label className="setting">
          <input id="save-history" type="checkbox" aria-describedby="history-setting-help" checked={saving}
            onChange={event => onSave(event.target.checked)} /> Save history in this browser
        </label>
        <p id="settings-status" role="status" hidden={!error}>{error}</p>
      </div>
    </section>
  );
}

function readHistory() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(storageKey));
    if (Array.isArray(saved)) {
      return {
        entries: saved.filter(entry => typeof entry?.source === "string" &&
          typeof entry?.translation === "string").slice(0, historyLimit),
        saving: true,
        error: "",
      };
    }
    return { entries: [], saving: false, error: "" };
  } catch {
    return { entries: [], saving: false, error: storageError };
  }
}

function useTranslationHistory() {
  const [history, setHistory] = useState(readHistory);
  const current = useRef(history);

  function update(entries, saving = current.current.saving, keepOnError = false) {
    let next = { entries, saving, error: "" };
    try {
      if (saving) {
        window.localStorage.setItem(storageKey, JSON.stringify(entries));
      } else if (current.current.saving) {
        window.localStorage.removeItem(storageKey);
      }
    } catch {
      next = { ...(keepOnError ? next : current.current), error: storageError };
    }
    current.current = next;
    setHistory(next);
    return !next.error;
  }

  return {
    ...history,
    add: entry => update([entry, ...current.current.entries].slice(0, historyLimit), current.current.saving, true),
    clear: () => update([]),
    setSaving: saving => update(current.current.entries, saving),
  };
}

function pageFromHash() {
  return pages.find(page => page.id === window.location.hash.slice(1)) ?? pages[0];
}

function App() {
  const [navigation, setNavigation] = useState(() => ({ page: pageFromHash(), focus: false }));
  const [text, setText] = useState("");
  const [output, setOutput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState({ message: "", error: false });
  const source = useRef(null);
  const outputNode = useRef(null);
  const focusSource = useRef(false);
  const inFlight = useRef(false);
  const history = useTranslationHistory();

  useEffect(() => {
    function navigate() {
      if (window.location.hash !== "#main") {
        setNavigation({ page: pageFromHash(), focus: true });
      }
    }
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);

  useLayoutEffect(() => {
    if (!navigation.focus) return;
    const destination = focusSource.current ? source.current : document.getElementById(navigation.page.heading);
    focusSource.current = false;
    destination.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [navigation]);

  function announce(message, error = false) {
    setStatus({ message, error });
  }

  function updateText(value) {
    setText(value);
    setOutput("");
    announce("");
  }

  function usePhrase(phrase) {
    updateText(phrase);
    focusSource.current = true;
    window.location.hash = pages[0].id;
  }

  async function translate(event) {
    event.preventDefault();
    if (inFlight.current) return;
    const sourceText = text.trim();
    if (!sourceText) {
      announce("Enter English text to translate.", true);
      source.current.focus();
      return;
    }

    inFlight.current = true;
    setBusy(true);
    setOutput("");
    announce("Translating…");
    try {
      const response = await fetch(apiOrigin + "/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sourceText }),
      });
      const result = await response.json();
      if (!response.ok || typeof result?.translation !== "string" || !result.translation) {
        announce(result?.error || "Translation failed. Try again.", true);
        return;
      }
      setOutput(result.translation);
      const saved = history.add({ source: sourceText, translation: result.translation });
      announce(saved ? "Translation ready." : "Translation ready. History could not be saved.");
    } catch {
      announce(apiOrigin
        ? "The translator may be starting up. Wait a moment and try again."
        : "Could not reach the local translator. Check that Saarthi is running and try again.", true);
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(output);
      announce("Translation copied.");
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(outputNode.current);
      selection.removeAllRanges();
      selection.addRange(range);
      announce("Translation selected. Use your device’s copy command.");
    }
  }

  const active = navigation.page.id;
  return (
    <>
      <IconLibrary />
      <a className="skip" href="#main">Skip to content</a>
      <div className="app-shell">
        <Sidebar active={active} />
        <main id="main" tabIndex={-1}>
          <Masthead />
          <TranslateView hidden={active !== "translate-view"} text={text} output={output} busy={busy}
            status={status} entries={history.entries} sourceRef={source} outputRef={outputNode} onText={updateText}
            onClear={() => { updateText(""); source.current.focus(); }} onSubmit={translate} onCopy={copy} />
          <HistoryView hidden={active !== "history-view"} entries={history.entries} onDelete={history.clear} />
          <PhrasesView hidden={active !== "phrases-view"} busy={busy} onPhrase={usePhrase} />
          <SettingsView hidden={active !== "settings-view"} saving={history.saving} error={history.error} onSave={history.setSaving} />
        </main>
      </div>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
