(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const phrasebook = [
    { icon: "☀️", en: "Good morning", hi: "सुप्रभात" },
    { icon: "💗", en: "How are you?", hi: "आप कैसे हैं?" },
    { icon: "🤝", en: "Thank you", hi: "धन्यवाद" },
    { icon: "📍", en: "Where are you going?", hi: "आप कहाँ जा रहे हैं?" },
    { icon: "🍴", en: "I would like to eat.", hi: "मैं खाना चाहता हूँ।" },
    { icon: "✈️", en: "See you later", hi: "फिर मिलते हैं।" },
    { icon: "📖", en: "Can you help me?", hi: "क्या आप मेरी मदद कर सकते हैं?" },
    { icon: "🙂", en: "Have a nice day!", hi: "आपका दिन शुभ हो!" },
    { icon: "💬", en: "I am learning artificial intelligence.", hi: "मैं आर्टिफिशियल इंटेलिजेंस सीख रहा हूँ।" },
    { icon: "☕", en: "Where is the nearest coffee shop?", hi: "निकटतम कॉफी शॉप कहाँ है?" },
    { icon: "💗", en: "Thank you so much.", hi: "बहुत बहुत धन्यवाद।" }
  ];
  const key = "saarthi-phrase-history-v1";
  let history = [];
  try {
    const stored = JSON.parse(localStorage.getItem(key) || "[]");
    if (Array.isArray(stored)) history = stored.filter(item => item && typeof item.en === "string" && typeof item.hi === "string").slice(0, 60);
  } catch (_) {}
  let result = "";
  const views = ["translate", "history", "phrases", "about", "settings", "help"];
  const show = (name) => {
    if (!views.includes(name)) return;
    views.forEach(view => {
      const panel = $("view-" + view);
      panel.classList.toggle("active", view === name);
    });
    document.querySelectorAll("[data-view]").forEach(el => {
      const active = el.dataset.view === name;
      el.classList.toggle("active", active);
      if (el.classList.contains("tab")) el.setAttribute("aria-current", active ? "page" : "false");
    });
    $("page-title").textContent = name === "translate" ? "Translate with Saarthi" : name === "phrases" ? "Quick Phrases" : name.charAt(0).toUpperCase() + name.slice(1);
    $("sidebar").classList.remove("open");
    $("backdrop").classList.add("hidden");
    if (name === "history") renderHistory();
  };
  const notify = message => { $("translation-notice").textContent = message; };
  const clearResult = () => {
    result = "";
    $("hindi-result").textContent = "";
    $("result-empty").classList.remove("hidden");
    $("result-full").classList.add("hidden");
    document.querySelector(".result-panel").classList.remove("has-result");
  };
  const showResult = text => {
    result = text;
    $("hindi-result").textContent = result;
    $("result-empty").classList.add("hidden");
    $("result-full").classList.remove("hidden");
    document.querySelector(".result-panel").classList.add("has-result");
  };
  const renderHistory = () => {
    const container = $("history-list");
    container.replaceChildren();
    const search = $("history-search").value.toLowerCase().trim();
    const entries = history.filter(h => (h.en + " " + h.hi).toLowerCase().includes(search));
    if (!entries.length) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "No phrasebook translations in your history yet.";
      container.append(p);
      return;
    }
    entries.forEach(h => {
      const card = document.createElement("div");
      card.className = "history-card";
      const content = document.createElement("div");
      const en = document.createElement("p");
      en.textContent = h.en;
      const hi = document.createElement("p");
      hi.className = "hi";
      hi.lang = "hi";
      hi.textContent = h.hi;
      const time = document.createElement("small");
      time.textContent = new Date(h.when).toLocaleString();
      content.append(en, hi, time);
      const copy = document.createElement("button");
      copy.className = "icon-btn";
      copy.type = "button";
      copy.textContent = "▢";
      copy.setAttribute("aria-label", "Copy translation");
      copy.addEventListener("click", () => copyText(h.hi));
      card.append(content, copy);
      container.append(card);
    });
  };
  const copyText = async text => {
    if (!text) return notify("No translation to copy yet.");
    try {
      await navigator.clipboard.writeText(text);
      notify("Translation copied.");
    } catch (_) {
      notify("Copy permission unavailable. Please select the translation text manually.");
    }
  };
  const speak = (text, lang) => {
    if (!text) return notify("No text available to read.");
    if (!("speechSynthesis" in window)) return notify("Text-to-speech is unavailable in this browser.");
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    speechSynthesis.speak(utterance);
  };
  const translate = () => {
    const input = $("english-input").value.trim();
    if (!input) return notify("Enter English text first.");
    const match = phrasebook.find(p => p.en.toLowerCase().replace(/[.!?]+$/, "") === input.toLowerCase().replace(/[.!?]+$/, ""));
    if (!match) {
      clearResult();
      return notify("This is a UI preview. Free-text translation requires Saarthi's trained model and a connected API. Try a Quick Phrase.");
    }
    showResult(match.hi);
    notify("Phrasebook translation shown — not model-generated.");
    history.unshift({ en: input, hi: match.hi, when: new Date().toISOString() });
    history = history.slice(0, 60);
    try { localStorage.setItem(key, JSON.stringify(history)); } catch (_) {}
  };
  const grid = $("phrase-grid");
  phrasebook.slice(0, 8).forEach(p => {
    const button = document.createElement("button");
    button.className = "phrase-card";
    const icon = document.createElement("span");
    icon.textContent = p.icon;
    const labels = document.createElement("div");
    const en = document.createElement("strong");
    en.textContent = p.en;
    const hi = document.createElement("small");
    hi.lang = "hi";
    hi.textContent = p.hi;
    labels.append(en, hi);
    button.append(icon, labels);
    button.addEventListener("click", () => {
      $("english-input").value = p.en;
      $("character-count").textContent = p.en.length + "/5000";
      show("translate");
      translate();
    });
    grid.append(button);
  });
  document.querySelectorAll("[data-view]").forEach(b => b.addEventListener("click", () => show(b.dataset.view)));
  $("get-started").addEventListener("click", () => {
    $("welcome").classList.add("hidden");
    $("workspace").classList.remove("hidden");
    show("translate");
  });
  $("open-menu").addEventListener("click", () => {
    $("sidebar").classList.add("open");
    $("backdrop").classList.remove("hidden");
  });
  $("close-menu").addEventListener("click", () => {
    $("sidebar").classList.remove("open");
    $("backdrop").classList.add("hidden");
  });
  $("backdrop").addEventListener("click", () => {
    $("sidebar").classList.remove("open");
    $("backdrop").classList.add("hidden");
  });
  $("english-input").addEventListener("input", () => {
    $("character-count").textContent = $("english-input").value.length + "/5000";
    notify("");
  });
  $("clear-input").addEventListener("click", () => {
    $("english-input").value = "";
    $("character-count").textContent = "0/5000";
    notify("");
    clearResult();
  });
  $("clear-result").addEventListener("click", () => { clearResult(); notify(""); });
  $("translate-btn").addEventListener("click", translate);
  $("copy-result").addEventListener("click", () => copyText(result));
  $("copy-result-bottom").addEventListener("click", () => copyText(result));
  $("speak-result").addEventListener("click", () => speak(result, "hi-IN"));
  $("history-search").addEventListener("input", renderHistory);
  $("speak-input").addEventListener("click", () => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return notify("Voice input is unavailable in this browser.");
    const recognition = new Recognition();
    recognition.lang = "en-IN";
    recognition.onresult = event => {
      $("english-input").value = event.results[0][0].transcript.slice(0, 5000);
      $("character-count").textContent = $("english-input").value.length + "/5000";
      notify("Voice input captured.");
    };
    recognition.onerror = event => notify("Microphone input failed: " + event.error);
    recognition.start();
  });
  // Welcome is intentional on initial load; navigation does not require login.
})();
