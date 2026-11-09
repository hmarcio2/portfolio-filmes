(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var sessionId = null;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function movieCard(m) {
    var card = el("article", "movie");
    if (m.poster_url) {
      var img = el("img");
      img.src = m.poster_url;
      img.alt = "Pôster de " + m.title;
      img.loading = "lazy";
      card.appendChild(img);
    }
    var body = el("div");
    body.appendChild(el("h4", null, m.title));
    var meta = [m.year, (m.genres || []).join(", "), m.rating ? "nota " + m.rating : null].filter(Boolean).join(" · ");
    body.appendChild(el("div", "meta", meta));
    body.appendChild(el("p", null, m.overview));
    card.appendChild(body);
    return card;
  }

  async function ask(question) {
    var log = $("log");
    log.appendChild(el("div", "msg user", question));
    var bot = el("div", "msg bot");
    var steps = el("div", "steps");
    var movies = el("div", "movies");
    var text = el("span", null, "…");
    bot.appendChild(steps);
    bot.appendChild(movies);
    bot.appendChild(text);
    log.appendChild(bot);
    var started = false;
    $("send").disabled = true;

    try {
      var res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: question }),
      });
      if (!res.ok) {
        var detail = "";
        try { detail = (await res.json()).error; } catch (e) { /* corpo não-JSON */ }
        throw new Error(detail || "Erro " + res.status);
      }
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      for (;;) {
        var r = await reader.read();
        if (r.done) break;
        buffer += decoder.decode(r.value, { stream: true });
        var blocks = buffer.split("\n\n");
        buffer = blocks.pop();
        blocks.forEach(function (block) {
          var m = /^event: (\w+)\ndata: (.*)$/s.exec(block);
          if (!m) return;
          var data = JSON.parse(m[2]);
          if (m[1] === "meta") sessionId = data.session_id;
          else if (m[1] === "delta") {
            if (!started) { text.textContent = ""; started = true; }
            text.textContent += data.text;
          } else if (m[1] === "tool") {
            var similar = data.name === "similar_movies";
            var chip = el("span", "step", similar ? "buscando filmes parecidos" : "buscando filmes");
            if (data.input) chip.title = similar ? "Parecidos com: " + (data.input.title || "") : (data.input.description || "");
            steps.appendChild(chip);
          } else if (m[1] === "movies") {
            if (data.reference) {
              var ref = data.reference.title + (data.reference.year ? " (" + data.reference.year + ")" : "");
              movies.appendChild(el("p", "muted small ref", "Parecidos com " + ref));
            }
            data.movies.forEach(function (mv) { movies.appendChild(movieCard(mv)); });
          } else if (m[1] === "error") {
            bot.className = "msg error";
            if (!started) text.textContent = "";
            text.textContent += (text.textContent ? "\n\n" : "") + data.error;
          }
          bot.scrollIntoView({ block: "end" });
        });
      }
    } catch (err) {
      bot.className = "msg error";
      text.textContent = err.message || "Não foi possível falar com o recomendador.";
    } finally {
      $("send").disabled = false;
      $("question").focus();
    }
  }

  $("form").addEventListener("submit", function (e) {
    e.preventDefault();
    var q = $("question").value.trim();
    if (!q) return;
    $("question").value = "";
    ask(q);
  });

  [
    "Um suspense psicológico tipo Ilha do Medo, mas mais leve",
    "Animação para ver em família que fale de emoções",
    "Ficção científica sobre linguagem e comunicação com alienígenas",
    "Algo parecido com Matrix, mas mais recente",
  ].forEach(function (q) {
    var b = el("button", null, q);
    b.type = "button";
    b.addEventListener("click", function () { $("question").value = q; $("form").requestSubmit(); });
    var li = el("li");
    li.appendChild(b);
    $("examples").appendChild(li);
  });

  fetch("/info").then(function (r) { return r.json(); }).then(function (i) {
    var parts = ["Catálogo: " + i.movies + " filmes (" + i.source + ")."];
    if (i.attribution) parts.push(i.attribution);
    $("info").textContent = parts.join(" ");
  }).catch(function () { /* informativo */ });
})();
