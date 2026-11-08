# Command builder

Pick what the run should do, and copy the command it builds. Every field maps to
a real flag of `experiments/train.py`: see [Environments and presets](environments.md),
[Constraining the body](your-robot.md), [Fitness and tasks](fitness-tasks.md) and
[Long runs](long-runs.md). Options left at their default are not written into the
command, so the result stays short.

<div class="cb-panel" id="cb-panel">
<div class="cb-grid">
<label class="cb-field">Terrains
<select id="cb-terrains">
<option value="flat">flat (preset)</option>
<option value="terrain">terrain (preset)</option>
<option value="all">all (preset)</option>
<option value="custom">individual terrains</option>
</select>
</label>
<label class="cb-field">Task
<select id="cb-task">
<option value="steer">steer (default)</option>
<option value="forward">forward</option>
<option value="turn">turn</option>
</select>
</label>
<label class="cb-field">Payload (kg)
<input id="cb-payload" type="number" min="0" step="0.05" placeholder="0">
</label>
<label class="cb-field">Batches
<input id="cb-batches" type="number" min="1" value="30">
</label>
<label class="cb-field">Workers
<input id="cb-workers" type="number" min="1" placeholder="auto">
</label>
<label class="cb-field">Seed
<input id="cb-seed" type="number" min="0" placeholder="random each run">
</label>
</div>
<div id="cb-terr-list" class="cb-checks" hidden>
<label><input id="cb-t-flat" type="checkbox"> flat</label>
<label><input id="cb-t-rough" type="checkbox"> rough</label>
<label><input id="cb-t-steps" type="checkbox"> steps</label>
<label><input id="cb-t-step" type="checkbox"> step</label>
<label><input id="cb-t-slope" type="checkbox"> slope</label>
<label><input id="cb-t-slippery" type="checkbox"> slippery</label>
<label><input id="cb-t-rubble" type="checkbox"> rubble</label>
<label><input id="cb-t-ledges" type="checkbox"> ledges</label>
<label><input id="cb-t-soft" type="checkbox"> soft</label>
<label><input id="cb-t-downhill" type="checkbox"> downhill</label>
<label><input id="cb-t-steep" type="checkbox"> steep</label>
</div>
<fieldset class="cb-group">
<legend>Body options</legend>
<div class="cb-checks">
<label><input id="cb-no-wheels" type="checkbox"> no-wheels</label>
<label><input id="cb-no-passive" type="checkbox"> no-passive</label>
<label><input id="cb-no-segments" type="checkbox"> no-segments</label>
</div>
<div class="cb-grid">
<label class="cb-field">Max servos
<input id="cb-max-servos" type="number" min="1" value="16">
</label>
<label class="cb-field">Legs, min
<input id="cb-legs-min" type="number" min="0" placeholder="any">
</label>
<label class="cb-field">Legs, max
<input id="cb-legs-max" type="number" min="0" placeholder="any">
</label>
<label class="cb-field">Max size (mm)
<input id="cb-max-size" type="number" min="1" placeholder="no limit">
</label>
<label class="cb-field">Bed (mm)
<input id="cb-bed" type="number" min="1" value="220">
</label>
</div>
</fieldset>
<fieldset class="cb-group">
<legend>Score rules (experimental)</legend>
<div class="cb-checks">
<label><input id="cb-r-worse_turn" type="checkbox"> worse_turn</label>
<label><input id="cb-r-turn_forward" type="checkbox"> turn_forward</label>
<label><input id="cb-r-tilt" type="checkbox"> tilt</label>
<label><input id="cb-r-slip" type="checkbox"> slip</label>
<label><input id="cb-r-servo_ground" type="checkbox"> servo_ground</label>
</div>
<p class="cb-note">A rule is a measurement before it is a default: run it in a short
experiment and check what it does to the archive before using it for a real run.</p>
</fieldset>
<div class="cb-out no-copy">
<pre><code id="cb-cmd"></code></pre>
<button id="cb-copy" type="button" class="md-button md-button--primary">Copy</button>
</div>
</div>

<style>
.cb-panel { border: 1px solid var(--md-default-fg-color--lighter); border-radius: .2rem; padding: 1rem; margin: 1.5rem 0; }
.cb-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem 1rem; }
.cb-field { display: flex; flex-direction: column; gap: .2rem; font-size: .75rem; color: var(--md-default-fg-color--light); }
.cb-field select, .cb-field input { font: inherit; font-size: .8rem; color: var(--md-default-fg-color); background: var(--md-default-bg-color); border: 1px solid var(--md-default-fg-color--lighter); border-radius: .15rem; padding: .3rem .4rem; }
.cb-field select:focus, .cb-field input:focus { outline: none; border-color: var(--md-primary-fg-color); }
.cb-checks { display: flex; flex-wrap: wrap; gap: .3rem 1.2rem; margin: .5rem 0; font-size: .8rem; }
.cb-checks[hidden] { display: none; }
.cb-checks input { margin-right: .3rem; accent-color: var(--md-primary-fg-color); }
.cb-group { border: 1px solid var(--md-default-fg-color--lighter); border-radius: .15rem; padding: .5rem .8rem .8rem; margin: .8rem 0; }
.cb-group legend { padding: 0 .3rem; font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: var(--md-default-fg-color--light); }
.cb-note { margin: .4rem 0 0; font-size: .7rem; color: var(--md-default-fg-color--light); }
.cb-out { display: flex; flex-wrap: wrap; align-items: flex-start; gap: .8rem; margin-top: .8rem; }
.cb-out pre { flex: 1 1 20rem; margin: 0; background: var(--md-code-bg-color); border-radius: .2rem; padding: .6rem .8rem; overflow-x: auto; }
.cb-out code { font-size: .7rem; white-space: pre; color: var(--md-default-fg-color); }
.cb-out .md-button { margin: 0; }
@media screen and (max-width: 44.9375em) { .cb-out .md-button { width: 100%; } }
</style>

<script>
(function () {
  "use strict";
  var TERRAINS = ["flat", "rough", "steps", "step", "slope", "slippery",
                  "rubble", "ledges", "soft", "downhill", "steep"];
  var RULES = ["worse_turn", "turn_forward", "tilt", "slip", "servo_ground"];

  function $(id) { return document.getElementById(id); }

  function num(id) {
    var raw = $(id).value.trim();
    if (raw === "") return null;
    var value = Number(raw);
    return isFinite(value) ? value : null;
  }

  function terrainSpec() {
    var preset = $("cb-terrains").value;
    if (preset !== "custom") return preset;
    var picked = TERRAINS.filter(function (name) {
      return $("cb-t-" + name).checked;
    });
    return picked.length ? picked.join(",") : "flat";
  }

  function build() {
    var parts = ["python experiments/train.py"];
    var terrains = terrainSpec();
    if (terrains !== "flat") parts.push("--terrains " + terrains);
    var task = $("cb-task").value;
    if (task !== "steer") parts.push("--task " + task);
    var payload = num("cb-payload");
    if (payload !== null && payload > 0) parts.push("--payload " + payload);
    var batches = num("cb-batches");
    if (batches !== null && batches !== 30) parts.push("--batches " + batches);
    var workers = num("cb-workers");
    if (workers !== null && workers > 0) parts.push("--workers " + workers);
    var seed = num("cb-seed");
    if (seed !== null && seed >= 0) parts.push("--seed " + seed);

    var body = [];
    if ($("cb-no-wheels").checked) body.push("no-wheels");
    if ($("cb-no-passive").checked) body.push("no-passive");
    if ($("cb-no-segments").checked) body.push("no-segments");
    var servos = num("cb-max-servos");
    if (servos !== null && servos >= 1 && servos !== 16) body.push("max-servos=" + servos);
    var lo = num("cb-legs-min");
    var hi = num("cb-legs-max");
    if (lo === null && hi !== null) lo = 0;
    if (lo !== null) {
      if (hi !== null && hi > lo) body.push("legs=" + lo + ".." + hi);
      else body.push("legs=" + lo);
    }
    var size = num("cb-max-size");
    if (size !== null && size > 0) body.push("max-size=" + size);
    var bed = num("cb-bed");
    if (bed !== null && bed > 0 && bed !== 220) body.push("bed=" + bed);
    if (body.length) parts.push('--body "' + body.join(",") + '"');

    var rules = RULES.filter(function (name) { return $("cb-r-" + name).checked; });
    if (rules.length) parts.push('--rules "' + rules.join(",") + '"');

    $("cb-cmd").textContent = parts.join(" ");
  }

  function refresh() {
    $("cb-terr-list").hidden = $("cb-terrains").value !== "custom";
    build();
  }

  $("cb-panel").addEventListener("input", refresh);
  $("cb-panel").addEventListener("change", refresh);
  $("cb-copy").addEventListener("click", function () {
    var button = $("cb-copy");
    navigator.clipboard.writeText($("cb-cmd").textContent).then(function () {
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = "Copy"; }, 1200);
    });
  });

  refresh();
})();
</script>
