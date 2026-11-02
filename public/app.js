(function () {
  var meta = window.META;
  var form = document.getElementById("form");
  var submit = document.getElementById("submit");
  var inputs = Array.prototype.slice.call(form.querySelectorAll("input"));

  function fieldOf(input) { return input.closest(".field"); }
  function setError(input, msg) {
    var f = fieldOf(input);
    f.querySelector(".err").textContent = msg || "";
    input.classList.toggle("invalid", !!msg);
  }
  function clearErrors() { inputs.forEach(function (i) { setError(i, ""); }); }

  function fill(values) {
    inputs.forEach(function (i) { i.value = values ? values[i.name] : ""; });
    clearErrors();
    if (!values) showEmpty();
  }

  function pretty(name) { return name.toLowerCase(); }

  function showEmpty() {
    document.getElementById("empty").hidden = false;
    document.getElementById("output").hidden = true;
  }

  function show(res) {
    document.getElementById("empty").hidden = true;
    document.getElementById("output").hidden = false;

    var v = document.getElementById("verdict");
    v.textContent = res.label;
    v.className = "verdict " + res.label.toLowerCase();

    document.getElementById("bar").style.width = (res.p_malignant * 100).toFixed(1) + "%";
    document.getElementById("p-benign").textContent = (res.p_benign * 100).toFixed(1) + "%";
    document.getElementById("p-malignant").textContent = (res.p_malignant * 100).toFixed(1) + "%";

    var ul = document.getElementById("drivers");
    ul.innerHTML = "";
    res.drivers.forEach(function (d) {
      var li = document.createElement("li");
      li.textContent = "";
      var strong = document.createElement("strong");
      strong.textContent = pretty(d.feature);
      li.appendChild(strong);
      li.appendChild(document.createTextNode(" pushed the result towards " + d.direction));
      ul.appendChild(li);
    });

    var note = document.getElementById("range-note");
    if (res.out_of_range && res.out_of_range.length) {
      note.hidden = false;
      note.textContent = "Some values are outside the range seen in the dataset (" +
        res.out_of_range.map(pretty).join(", ") + "), so treat this estimate with extra caution.";
    } else {
      note.hidden = true;
    }
  }

  document.getElementById("fill-benign").addEventListener("click", function () { fill(meta.samples.benign); });
  document.getElementById("fill-malignant").addEventListener("click", function () { fill(meta.samples.malignant); });
  document.getElementById("clear").addEventListener("click", function () { fill(null); });
  inputs.forEach(function (i) { i.addEventListener("input", function () { setError(i, ""); }); });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    clearErrors();
    var payload = {}, bad = false;

    inputs.forEach(function (i) {
      var raw = i.value.trim().replace(",", ".");
      if (raw === "" || isNaN(Number(raw))) { setError(i, "Enter a number"); bad = true; return; }
      payload[i.name] = Number(raw);
    });
    if (bad) { inputs.find(function (i) { return i.classList.contains("invalid"); }).focus(); return; }

    submit.disabled = true;
    submit.textContent = "Predicting...";
    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (r) { return r.json().then(function (j) { return { status: r.status, body: j }; }); })
      .then(function (r) {
        if (r.status === 200 && r.body.ok) { show(r.body); }
        else if (r.body.errors) {
          inputs.forEach(function (i) { if (r.body.errors[i.name]) setError(i, r.body.errors[i.name]); });
        }
      })
      .catch(function () { alert("Could not reach the server. Is app.py still running?"); })
      .then(function () { submit.disabled = false; submit.textContent = "Predict"; });
  });
})();
