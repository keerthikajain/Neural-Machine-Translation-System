const form = document.querySelector("#translator");
const source = document.querySelector("#source");
const output = document.querySelector("#output");
const placeholder = document.querySelector("#placeholder");
const count = document.querySelector("#count");
const translate = document.querySelector("#translate");
const copy = document.querySelector("#copy");
const clear = document.querySelector("#clear");
const status = document.querySelector("#status");
const target = document.querySelector(".target");
const settingsStatus = document.querySelector("#settings-status");
const saveHistory = document.querySelector("#save-history");
const historySearch = document.querySelector("#history-search");
const historyClear = document.querySelector("#history-clear");
const historyConfirm = document.querySelector("#history-confirm");
const historyDelete = document.querySelector("#history-delete");
const views = [...document.querySelectorAll(".view")];
const navigation = [...document.querySelectorAll(".navigation a")];
const storageKey = "saarthi.history.v1";
const historyLimit = 30;
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
let busy = false;
let history = [];
let focusSource = false;

function storageError() {
  settingsStatus.textContent = "Browser storage is unavailable. Check your browser settings to manage saved translations.";
  settingsStatus.hidden = false;
}

function restoreHistory() {
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved === null) return;
    const entries = JSON.parse(saved);
    if (Array.isArray(entries)) {
      history = entries.filter(entry => typeof entry?.source === "string" &&
        typeof entry?.translation === "string").slice(0, historyLimit);
      saveHistory.checked = true;
    }
  } catch {
    storageError();
  }
}

function persistHistory() {
  try {
    if (saveHistory.checked) {
      localStorage.setItem(storageKey, JSON.stringify(history));
    } else {
      localStorage.removeItem(storageKey);
    }
    settingsStatus.hidden = true;
    return true;
  } catch {
    storageError();
    return false;
  }
}

function historyEntry(entry) {
  const item = document.createElement("li");
  item.className = "history-entry";
  const english = document.createElement("p");
  english.textContent = entry.source;
  const hindi = document.createElement("p");
  hindi.lang = "hi";
  hindi.textContent = entry.translation;
  item.append(english, hindi);
  return item;
}

function renderHistory() {
  const query = historySearch.value.trim().toLowerCase();
  const matches = history.filter(entry => entry.source.toLowerCase().includes(query) ||
    entry.translation.includes(query));
  document.querySelector("#history-list").replaceChildren(...matches.map(historyEntry));
  document.querySelector("#recent-list").replaceChildren(...history.slice(0, 3).map(historyEntry));
  document.querySelector("#recent-empty").hidden = history.length > 0;
  const empty = document.querySelector("#history-empty");
  empty.hidden = matches.length > 0;
  empty.textContent = history.length ? "No translations match your search." : "You have no translations yet.";
  historyClear.disabled = !history.length;
}

function navigate(moveFocus) {
  if (location.hash === "#main") return;
  const selected = views.find(view => "#" + view.id === location.hash) || views[0];
  views.forEach(view => { view.hidden = view !== selected; });
  navigation.forEach(link => {
    if (link.hash === "#" + selected.id) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  historyConfirm.hidden = true;
  if (moveFocus) {
    const destination = focusSource ? source : selected.querySelector("h1");
    focusSource = false;
    destination.focus({preventScroll: true});
    window.scrollTo(0, 0);
  }
}

function usePhrase(phrase) {
  source.value = phrase;
  source.dispatchEvent(new Event("input"));
  if (views[0].hidden) {
    focusSource = true;
    location.hash = views[0].id;
  } else {
    source.focus();
  }
}

function renderPhrases(list, entries) {
  entries.forEach(phrase => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "phrase";
    button.textContent = phrase;
    const arrow = document.createElement("span");
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "→";
    button.append(arrow);
    button.addEventListener("click", () => usePhrase(phrase));
    item.append(button);
    list.append(item);
  });
}

function setBusy(value) {
  busy = value;
  source.readOnly = value;
  translate.disabled = value;
  translate.querySelector("span").textContent = value ? "Translating…" : "Translate";
  clear.disabled = value;
  document.querySelectorAll(".phrase").forEach(button => { button.disabled = value; });
  target.setAttribute("aria-busy", String(value));
}

restoreHistory();
renderHistory();
renderPhrases(document.querySelector("#quick-preview"), phrases.slice(0, 5));
renderPhrases(document.querySelector("#phrase-list"), phrases);
navigate(false);
window.addEventListener("hashchange", () => navigate(true));
saveHistory.addEventListener("change", () => {
  if (!persistHistory()) saveHistory.checked = !saveHistory.checked;
});
historySearch.addEventListener("input", renderHistory);
historyClear.addEventListener("click", () => {
  historyConfirm.hidden = false;
  historyDelete.focus();
});
document.querySelector("#history-cancel").addEventListener("click", () => {
  historyConfirm.hidden = true;
  historyClear.focus();
});
historyDelete.addEventListener("click", () => {
  const previous = history;
  history = [];
  if (saveHistory.checked && !persistHistory()) {
    history = previous;
    document.querySelector("#history-empty").hidden = false;
    document.querySelector("#history-empty").textContent = "Could not clear saved history. Check your browser settings.";
    return;
  }
  historyConfirm.hidden = true;
  historySearch.value = "";
  renderHistory();
  historySearch.focus();
});

function announce(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
}

function resetOutput() {
  output.textContent = "";
  output.hidden = true;
  placeholder.hidden = false;
  copy.disabled = true;
}

source.addEventListener("input", () => {
  count.textContent = `${source.value.length.toLocaleString("en")} / 1,200`;
  if (!busy) {
    resetOutput();
    announce("");
  }
});

clear.addEventListener("click", () => {
  if (busy) return;
  source.value = "";
  source.dispatchEvent(new Event("input"));
  source.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const text = source.value.trim();
  if (!text) {
    announce("Enter English text to translate.", true);
    source.focus();
    return;
  }

  setBusy(true);
  resetOutput();
  announce("Translating…");
  try {
    const response = await fetch("/api/translate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    const result = await response.json();
    if (!response.ok) {
      announce(result.error || "Translation failed. Try again.", true);
      return;
    }
    output.textContent = result.translation;
    output.hidden = false;
    history.unshift({source: text, translation: result.translation});
    history.length = Math.min(history.length, historyLimit);
    const saved = !saveHistory.checked || persistHistory();
    renderHistory();
    placeholder.hidden = true;
    copy.disabled = false;
    announce(saved ? "Translation ready." : "Translation ready. History could not be saved.");
  } catch {
    announce("Could not reach the local translator. Check that Saarthi is running and try again.", true);
  } finally {
    setBusy(false);
  }
});

copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(output.textContent);
    announce("Translation copied.");
  } catch {
    const range = document.createRange();
    range.selectNodeContents(output);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    announce("Translation selected. Use your device’s copy command.");
  }
});
