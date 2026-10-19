/* ephacRTools — client-side multichannel image viewer
 *
 * Float32 channel data is sent once from R (Shiny custom message
 * "diagviewer-load"); contrast, colour, zoom/pan and pixel hover are handled
 * entirely in the browser.
 */
(function () {
  "use strict";

  var DEFAULT_COLORS = {
    "FITC": "#00ff00", "GFP": "#00ff00", "TX RED": "#ff00ff", "MCHERRY": "#ff00ff",
    "DAPI": "#3a7bff", "BF": "#ffffff"
  };
  var FALLBACK = ["#00ff00", "#ff00ff", "#3a7bff", "#ffffff", "#ffff00", "#00ffff"];
  var SLIDER_STEPS = 1000;
  // shared display settings (per channel label) for "lock across wells"
  var lockedSettings = {};

  function hexToRgb(h) {
    var n = parseInt(h.replace("#", ""), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function decodeF32(b64) {
    var bin = atob(b64), n = bin.length, buf = new Uint8Array(n);
    for (var i = 0; i < n; i++) buf[i] = bin.charCodeAt(i);
    return new Float32Array(buf.buffer);
  }

  function percentiles(arr, ps) {
    var n = arr.length, step = Math.max(1, Math.floor(n / 200000));
    var s = new Float32Array(Math.ceil(n / step)), k = 0;
    for (var i = 0; i < n; i += step) s[k++] = arr[i];
    s = s.subarray(0, k).sort();
    return ps.map(function (p) { return s[Math.min(k - 1, Math.max(0, Math.round(p * (k - 1))))]; });
  }

  function fmt(v) {
    if (!isFinite(v)) return "–";
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1000) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toPrecision(3);
  }

  function el(tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "style") e.style.cssText = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(function (c) { if (c) e.appendChild(c); });
    return e;
  }

  // ------------------------------------------------------------------------
  function Viewer(root, id) {
    this.root = root;
    this.id = id;
    this.layout = root.getAttribute("data-layout") || "split";
    this.view = { scale: 1, tx: 0, ty: 0 };
    this.hover = null;
    this.channels = [];
    this.pending = false;
    this.rois = [];
    this.calib = null;
    root.innerHTML = "";
    root.classList.add("dv-root");
    this.buildChrome();
    var self = this;
    // re-size canvases when the container width changes (e.g. the viewer was
    // filled while its tab was hidden, or the sidebar was toggled)
    this.lastWidth = 0;
    if (window.ResizeObserver) {
      this.ro = new ResizeObserver(function (entries) {
        var w = Math.round(entries[0].contentRect.width);
        if (w === self.lastWidth) return;
        self.lastWidth = w;
        requestAnimationFrame(function () {
          if (self.panels && self.panels.length) { self.sizeCanvases(); self.drawAll(); }
        });
      });
      this.ro.observe(root);
    }
  }

  Viewer.prototype.buildChrome = function () {
    var self = this;
    this.titleEl = el("span", { "class": "dv-title" });
    this.btnLayout = el("button", { "class": "btn btn-default btn-xs", type: "button",
      onclick: function () { self.setLayout(self.layout === "split" ? "composite" : "split"); } });
    this.bfBox = el("input", { type: "checkbox" });
    this.bfBox.checked = !!lockedSettings.__bf;
    this.bfBox.addEventListener("change", function () {
      lockedSettings.__bf = self.bfBox.checked; self.scheduleRender(true);
    });
    this.roiBox = el("input", { type: "checkbox" });
    this.roiBox.checked = lockedSettings.__roi !== false;
    this.roiBox.addEventListener("change", function () {
      lockedSettings.__roi = self.roiBox.checked; self.scheduleRender(false);
    });
    this.roiLabel = el("label", { "class": "dv-lock", title: "Show the hole ROI outline", style: "display:none" },
      [this.roiBox, document.createTextNode(" ROI")]);
    this.calibBox = el("input", { type: "checkbox" });
    this.calibBox.checked = lockedSettings.__calib !== false;
    this.calibBox.addEventListener("change", function () {
      lockedSettings.__calib = self.calibBox.checked;
      if (self.calibBox.checked) self.applyCalib(); else self.autoContrast();
    });
    this.calibLabel = el("label", { "class": "dv-lock", style: "display:none",
      title: "Use the plate-wide display range computed from the ROI measurements" },
      [this.calibBox, document.createTextNode(" plate range")]);
    this.lockBox = el("input", { type: "checkbox" });
    this.lockBox.checked = !!lockedSettings.__lock;
    this.lockBox.addEventListener("change", function () {
      lockedSettings.__lock = self.lockBox.checked;
      if (self.lockBox.checked) self.storeLocked();
    });
    var toolbar = el("div", { "class": "dv-toolbar" }, [
      this.titleEl,
      el("span", { "class": "dv-spacer" }),
      this.btnLayout,
      el("button", { "class": "btn btn-default btn-xs", type: "button", text: "Fit",
        onclick: function () { self.resetView(); } }),
      el("button", { "class": "btn btn-default btn-xs", type: "button", text: "Auto contrast",
        onclick: function () { self.autoContrast(); } }),
      this.roiLabel, this.calibLabel,
      el("label", { "class": "dv-lock", title: "Add the brightfield channel to the composite" },
        [this.bfBox, document.createTextNode(" BF in composite")]),
      el("label", { "class": "dv-lock", title: "Keep contrast/colour settings when switching wells" },
        [this.lockBox, document.createTextNode(" lock settings")])
    ]);
    this.stage = el("div", { "class": "dv-stage" });
    this.readout = el("div", { "class": "dv-readout", text: "hover over the image for pixel values" });
    this.controls = el("div", { "class": "dv-controls" });
    this.root.appendChild(toolbar);
    this.root.appendChild(this.stage);
    this.root.appendChild(this.readout);
    this.root.appendChild(this.controls);
  };

  Viewer.prototype.load = function (msg) {
    var self = this;
    this.w = msg.w; this.h = msg.h;
    this.key = msg.key || null;
    this.rois = msg.rois || [];
    this.calib = msg.calib || null;
    this.roiLabel.style.display = this.rois.length ? "" : "none";
    this.scale = msg.scale || 1;
    this.titleEl.textContent = msg.title || "";
    var colors = msg.colors || {};
    var keep = lockedSettings.__lock;
    this.channels = msg.labels.map(function (lab, i) {
      var data = decodeF32(msg.channels[i]);
      var p = percentiles(data, [0, 0.005, 0.998, 1]);
      var prev = keep ? lockedSettings[lab] : null;
      var ch = {
        label: lab, data: data, isBF: /^(BF|brightfield|DIC|TL)$/i.test(lab),
        dmin: p[0], dmax: p[3] > p[0] ? p[3] : p[0] + 1,
        min: prev ? prev.min : p[1],
        max: prev ? prev.max : (p[2] > p[1] ? p[2] : p[3]),
        gamma: prev ? prev.gamma : 1,
        visible: prev ? prev.visible : true,
        color: prev ? prev.color : (colors[lab] || DEFAULT_COLORS[lab.toUpperCase()] || FALLBACK[i % FALLBACK.length]),
        u8: new Uint8ClampedArray(self.w * self.h)
      };
      ch.auto = [p[1], p[2] > p[1] ? p[2] : p[3]];
      return ch;
    });
    this.setCalib(this.calib, true);
    this.channels.forEach(function (c) { self.computeChannel(c); });
    this.buildControls();
    this.buildPanels();
    this.resetView();
    this.emitDisplay();
  };

  // map a value to [0,255] using min/max/gamma
  Viewer.prototype.computeChannel = function (c) {
    var d = c.data, o = c.u8, lo = c.min, span = c.max - c.min, g = c.gamma, n = d.length;
    if (span <= 0) span = 1e-9;
    if (g === 1) {
      for (var i = 0; i < n; i++) o[i] = (d[i] - lo) / span * 255;
    } else {
      for (var j = 0; j < n; j++) {
        var t = (d[j] - lo) / span;
        o[j] = t <= 0 ? 0 : t >= 1 ? 255 : Math.pow(t, g) * 255;
      }
    }
  };

  // ---------------------------------------------------------------- panels
  Viewer.prototype.buildPanels = function () {
    var self = this;
    this.stage.innerHTML = "";
    this.panels = [];
    var defs = [];
    if (this.layout === "split") {
      this.channels.forEach(function (c, i) { if (c.visible) defs.push({ name: c.label, chans: [i], color: c.color }); });
    }
    defs.push({ name: "Composite", chans: null });
    this.stage.className = "dv-stage dv-" + this.layout;
    this.stage.style.setProperty("--dv-cols", Math.min(defs.length, this.layout === "split" ? 3 : 1));
    defs.forEach(function (def) {
      var cv = el("canvas", { "class": "dv-canvas" });
      var off = document.createElement("canvas");
      off.width = self.w; off.height = self.h;
      var wrap = el("div", { "class": "dv-panel" }, [
        cv, el("span", { "class": "dv-panel-label", text: def.name,
                         style: def.color ? "border-left:4px solid " + def.color : "" })
      ]);
      self.stage.appendChild(wrap);
      var p = { def: def, canvas: cv, off: off, offCtx: off.getContext("2d"), wrap: wrap };
      self.panels.push(p);
      self.attachMouse(p);
    });
    this.btnLayout.textContent = this.layout === "split" ? "Composite view" : "Split channels";
    this.sizeCanvases();
    this.renderAll();
  };

  Viewer.prototype.sizeCanvases = function () {
    var self = this;
    this.panels.forEach(function (p) {
      var w = p.wrap.clientWidth;
      if (!w) return;                 // hidden (e.g. inactive tab): size later
      var h = Math.round(w * self.h / self.w);
      var dpr = window.devicePixelRatio || 1;
      p.canvas.width = Math.round(w * dpr); p.canvas.height = Math.round(h * dpr);
      p.canvas.style.height = h + "px";
      p.cssW = w; p.cssH = h; p.dpr = dpr;
    });
  };

  // canvases are sized from their container; re-size if it changed since
  Viewer.prototype.ensureSized = function () {
    if (!this.panels || !this.panels.length) return;
    var p = this.panels[0];
    if (p.wrap.clientWidth && p.wrap.clientWidth !== p.cssW) this.sizeCanvases();
  };

  Viewer.prototype.renderAll = function () {
    var self = this;
    this.panels.forEach(function (p) { self.renderOff(p); });
    this.drawAll();
  };

  Viewer.prototype.renderOff = function (p) {
    var n = this.w * this.h, img = p.offCtx.createImageData(this.w, this.h), px = img.data;
    var chans = p.def.chans ? p.def.chans.map(function (i) { return this.channels[i]; }, this)
                            : this.channels.filter(function (c) { return c.visible && (!c.isBF || this.bfBox.checked); }, this);
    var single = p.def.chans && chans.length === 1;
    var cols = chans.map(function (c) { return single ? [255, 255, 255] : hexToRgb(c.color); });
    for (var i = 0, k = 0; i < n; i++, k += 4) {
      var r = 0, g = 0, b = 0;
      for (var j = 0; j < chans.length; j++) {
        var v = chans[j].u8[i];
        if (v) { r += v * cols[j][0]; g += v * cols[j][1]; b += v * cols[j][2]; }
      }
      px[k] = r / 255; px[k + 1] = g / 255; px[k + 2] = b / 255; px[k + 3] = 255;
    }
    p.offCtx.putImageData(img, 0, 0);
  };

  Viewer.prototype.drawAll = function () {
    var self = this;
    this.panels.forEach(function (p) { self.drawPanel(p); });
  };

  Viewer.prototype.drawPanel = function (p) {
    if (!p.cssW) return;
    var ctx = p.canvas.getContext("2d"), v = this.view;
    var base = p.cssW / this.w;               // fit-to-width scale
    var s = base * v.scale * p.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, p.canvas.width, p.canvas.height);
    ctx.imageSmoothingEnabled = v.scale * base < 2;
    ctx.setTransform(s, 0, 0, s, v.tx * base * p.dpr, v.ty * base * p.dpr);
    ctx.drawImage(p.off, 0, 0);
    if (this.rois.length && this.roiBox.checked) {
      ctx.lineWidth = 1.5 * p.dpr / s;
      ctx.strokeStyle = "#00e5ff";
      this.rois.forEach(function (r) {
        ctx.beginPath();
        for (var i = 0; i < r.x.length; i++) {
          if (i === 0) ctx.moveTo(r.x[i], r.y[i]); else ctx.lineTo(r.x[i], r.y[i]);
        }
        ctx.closePath(); ctx.stroke();
      });
    }
    if (this.hover) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      var x = (this.hover.x + 0.5) * s + v.tx * base * p.dpr;
      var y = (this.hover.y + 0.5) * s + v.ty * base * p.dpr;
      ctx.strokeStyle = "rgba(255,255,0,0.7)"; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0); ctx.lineTo(x, p.canvas.height);
      ctx.moveTo(0, y); ctx.lineTo(p.canvas.width, y);
      ctx.stroke();
    }
  };

  Viewer.prototype.scheduleRender = function (full) {
    var self = this;
    this.needFull = this.needFull || full;
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(function () {
      self.pending = false;
      self.ensureSized();
      if (self.needFull) self.renderAll(); else self.drawAll();
      self.needFull = false;
    });
  };

  Viewer.prototype.resetView = function () {
    this.view = { scale: 1, tx: 0, ty: 0 };
    this.sizeCanvases();
    this.drawAll();
  };

  Viewer.prototype.setLayout = function (l) {
    this.layout = l;
    this.buildPanels();
  };

  // image coordinates of a mouse event on panel p
  Viewer.prototype.toImage = function (p, e) {
    var r = p.canvas.getBoundingClientRect(), base = r.width / this.w, v = this.view;
    return {
      x: ((e.clientX - r.left) / base - v.tx) / v.scale,
      y: ((e.clientY - r.top) / base - v.ty) / v.scale
    };
  };

  Viewer.prototype.attachMouse = function (p) {
    var self = this, drag = null;
    p.canvas.addEventListener("wheel", function (e) {
      e.preventDefault();
      var pt = self.toImage(p, e), v = self.view;
      var f = e.deltaY < 0 ? 1.25 : 0.8;
      var ns = Math.min(64, Math.max(1, v.scale * f));
      v.tx = v.tx + pt.x * (v.scale - ns);
      v.ty = v.ty + pt.y * (v.scale - ns);
      v.scale = ns;
      if (ns === 1) { v.tx = 0; v.ty = 0; }
      self.scheduleRender(false);
    }, { passive: false });
    p.canvas.addEventListener("mousedown", function (e) {
      drag = { x: e.clientX, y: e.clientY, tx: self.view.tx, ty: self.view.ty };
    });
    window.addEventListener("mouseup", function () { drag = null; });
    p.canvas.addEventListener("mousemove", function (e) {
      if (p.wrap.clientWidth !== p.cssW) { self.sizeCanvases(); self.drawAll(); }
      if (drag) {
        var base = p.canvas.getBoundingClientRect().width / self.w;
        self.view.tx = drag.tx + (e.clientX - drag.x) / base;
        self.view.ty = drag.ty + (e.clientY - drag.y) / base;
      }
      var pt = self.toImage(p, e);
      var x = Math.floor(pt.x), y = Math.floor(pt.y);
      if (x >= 0 && y >= 0 && x < self.w && y < self.h) {
        self.hover = { x: x, y: y };
        self.showReadout(x, y);
      } else {
        self.hover = null;
      }
      self.scheduleRender(false);
    });
    p.canvas.addEventListener("mouseleave", function () {
      self.hover = null; self.scheduleRender(false);
    });
    p.canvas.addEventListener("dblclick", function () { self.resetView(); });
  };

  Viewer.prototype.showReadout = function (x, y) {
    var i = y * this.w + x;
    this.readout.innerHTML = "";
    this.readout.appendChild(el("span", { "class": "dv-xy", text: "x " + x * this.scale + "  y " + y * this.scale +
      (this.scale > 1 ? "  (" + this.scale + "\u00d7 binned view)" : "") }));
    this.channels.forEach(function (c) {
      this.readout.appendChild(el("span", { "class": "dv-val" }, [
        el("i", { style: "background:" + c.color }),
        document.createTextNode(c.label + " " + fmt(c.data[i]))
      ]));
    }, this);
  };

  // -------------------------------------------------------------- controls
  // slider position <-> value, cubic so that the low end gets fine control
  function posToVal(c, t) { return c.dmin + (c.dmax - c.dmin) * Math.pow(t / SLIDER_STEPS, 3); }
  function valToPos(c, v) {
    var t = (v - c.dmin) / (c.dmax - c.dmin);
    return Math.round(Math.cbrt(Math.min(1, Math.max(0, t))) * SLIDER_STEPS);
  }

  Viewer.prototype.buildControls = function () {
    var self = this;
    this.controls.innerHTML = "";
    this.channels.forEach(function (c, idx) {
      var vis = el("input", { type: "checkbox", title: "show channel" });
      vis.checked = c.visible;
      var col = el("input", { type: "color", value: c.color, "class": "dv-color" });
      var minS = el("input", { type: "range", min: 0, max: SLIDER_STEPS, "class": "dv-range" });
      var maxS = el("input", { type: "range", min: 0, max: SLIDER_STEPS, "class": "dv-range" });
      var minN = el("input", { type: "number", "class": "dv-num", step: "any" });
      var maxN = el("input", { type: "number", "class": "dv-num", step: "any" });
      var gam = el("input", { type: "range", min: 0.2, max: 3, step: 0.05, value: c.gamma, "class": "dv-gamma",
                              title: "gamma" });
      var gamL = el("span", { "class": "dv-gamma-l" });
      var hist = el("canvas", { "class": "dv-hist", width: 200, height: 28 });

      function sync() {
        minS.value = valToPos(c, c.min); maxS.value = valToPos(c, c.max);
        minN.value = fmt(c.min); maxN.value = fmt(c.max);
        gamL.textContent = "γ " + Number(c.gamma).toFixed(2);
        self.drawHist(c, hist);
      }
      function changed(full) {
        self.computeChannel(c); sync(); self.scheduleRender(true);
        if (self.lockBox.checked) self.storeLocked();
        if (full) self.emitDisplay();
      }
      vis.addEventListener("change", function () {
        c.visible = vis.checked;
        if (self.layout === "split") self.buildPanels(); else self.scheduleRender(true);
        if (self.lockBox.checked) self.storeLocked();
      });
      col.addEventListener("input", function () { c.color = col.value; self.scheduleRender(true); });
      col.addEventListener("change", function () {
        c.color = col.value;
        if (self.layout === "split") self.buildPanels();
        changed(true);
      });
      minS.addEventListener("input", function () { c.min = Math.min(posToVal(c, +minS.value), c.max); changed(); });
      maxS.addEventListener("input", function () { c.max = Math.max(posToVal(c, +maxS.value), c.min); changed(); });
      minN.addEventListener("change", function () { var v = parseFloat(minN.value); if (isFinite(v)) c.min = v; changed(); });
      maxN.addEventListener("change", function () { var v = parseFloat(maxN.value); if (isFinite(v)) c.max = v; changed(); });
      gam.addEventListener("input", function () { c.gamma = +gam.value; changed(); });
      c.sync = function () { gam.value = c.gamma; vis.checked = c.visible; col.value = c.color; sync(); };

      var row = el("div", { "class": "dv-ch" }, [
        el("div", { "class": "dv-ch-head" }, [
          vis, col, el("b", { text: c.label }),
          el("span", { "class": "dv-range-txt", text: "data " + fmt(c.dmin) + " – " + fmt(c.dmax) }),
          el("button", { "class": "btn btn-default btn-xs", type: "button", text: "auto",
            onclick: function () { c.min = c.auto[0]; c.max = c.auto[1]; c.gamma = 1; c.sync(); changed(); } })
        ]),
        hist,
        el("div", { "class": "dv-ch-sl" }, [el("span", { text: "min" }), minS, minN]),
        el("div", { "class": "dv-ch-sl" }, [el("span", { text: "max" }), maxS, maxN]),
        el("div", { "class": "dv-ch-sl" }, [gamL, gam])
      ]);
      self.controls.appendChild(row);
      sync();
    });
  };

  Viewer.prototype.drawHist = function (c, cv) {
    if (!c.hist) {
      var bins = new Float64Array(cv.width), d = c.data, n = d.length, step = Math.max(1, Math.floor(n / 200000));
      for (var i = 0; i < n; i += step) bins[valToPos(c, d[i]) * (cv.width - 1) / SLIDER_STEPS | 0]++;
      var mx = 0;
      for (var j = 0; j < bins.length; j++) { bins[j] = Math.log1p(bins[j]); mx = Math.max(mx, bins[j]); }
      c.hist = { bins: bins, max: mx };
    }
    var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#888";
    for (var b = 0; b < W; b++) {
      var hh = c.hist.max ? c.hist.bins[b] / c.hist.max * H : 0;
      ctx.fillRect(b, H - hh, 1, hh);
    }
    var x0 = valToPos(c, c.min) / SLIDER_STEPS * W, x1 = valToPos(c, c.max) / SLIDER_STEPS * W;
    ctx.fillStyle = c.color; ctx.globalAlpha = 0.25; ctx.fillRect(x0, 0, Math.max(1, x1 - x0), H);
    ctx.globalAlpha = 1;
  };

  // plate-wide display ranges {channel: [min, max]}
  Viewer.prototype.setCalib = function (calib, silent) {
    var had = !!this.calib;
    this.calib = calib && Object.keys(calib).length ? calib : null;
    if (!this.calib && had && !silent && this.calibBox.checked) this.autoContrast();
    this.calibLabel.style.display = this.calib ? "" : "none";
    if (this.calib && this.calibBox.checked) this.applyCalib(silent);
  };

  Viewer.prototype.applyCalib = function (silent) {
    var self = this, cal = this.calib;
    if (!cal) return;
    this.channels.forEach(function (c) {
      var r = cal[c.label];
      if (!r) return;
      c.min = r[0]; c.max = r[1]; c.gamma = 1;
      if (!silent) { self.computeChannel(c); if (c.sync) c.sync(); }
    });
    if (!silent) this.scheduleRender(true);
  };

  Viewer.prototype.autoContrast = function () {
    var self = this;
    this.channels.forEach(function (c) {
      c.min = c.auto[0]; c.max = c.auto[1]; c.gamma = 1;
      self.computeChannel(c); if (c.sync) c.sync();
    });
    if (this.lockBox.checked) this.storeLocked();
    this.scheduleRender(true);
  };

  Viewer.prototype.storeLocked = function () {
    this.channels.forEach(function (c) {
      lockedSettings[c.label] = { min: c.min, max: c.max, gamma: c.gamma, visible: c.visible, color: c.color };
    });
  };

  Viewer.prototype.emitDisplay = function () {
    if (!window.Shiny || !Shiny.setInputValue) return;
    var out = {};
    this.channels.forEach(function (c) { out[c.label] = c.color; });
    Shiny.setInputValue(this.id + "_display", out);
  };

  // ------------------------------------------------------------- registry
  var instances = {};

  function getViewer(id) {
    var root = document.getElementById(id);
    if (!root) return null;
    var v = instances[id];
    if (!v || v.root !== root || !root.isConnected) v = instances[id] = new Viewer(root, id);
    return v;
  }

  function loadWhenReady(msg, tries) {
    var v = getViewer(msg.id);
    if (!v) {
      if ((tries || 0) < 40) setTimeout(function () { loadWhenReady(msg, (tries || 0) + 1); }, 50);
      return;
    }
    v.load(msg);
  }

  function resizeAll() {
    Object.keys(instances).forEach(function (k) {
      var v = instances[k];
      if (v.root.isConnected && v.panels) { v.sizeCanvases(); v.drawAll(); }
    });
  }
  window.addEventListener("resize", resizeAll);
  // shinydashboard / bootstrap fire "shown" when a tab or modal becomes visible
  if (window.jQuery) jQuery(document).on("shown shown.bs.tab shown.bs.modal", function () {
    setTimeout(resizeAll, 0);
  });

  // Keyboard shortcuts: forwarded to Shiny only while `scopeId` is visible
  // and the user is not typing into a form field.
  // Send a scoring event: captures the notes field value and the well that
  // is actually displayed, so the server never attributes a stale note or
  // keystroke to the wrong well.
  function sendEvent(inputId, viewerId, notesId, key) {
    var v = instances[viewerId], notes = document.getElementById(notesId);
    var payload = { key: key, t: Date.now(), well: v ? v.key : null,
                    notes: notes ? notes.value : "" };
    if (notes && /^(arrowright|arrowleft|s)$/.test(key)) notes.value = "";
    Shiny.setInputValue(inputId, payload, { priority: "event" });
  }

  function bindKeys(inputId, scopeId, viewerId, notesId) {
    document.addEventListener("keydown", function (e) {
      var scope = document.getElementById(scopeId);
      if (!scope || scope.offsetParent === null) return;
      var t = e.target.tagName;
      if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
      var k = e.key;
      if (!/^([1-9]|n|N|ArrowLeft|ArrowRight|s|S)$/.test(k)) return;
      e.preventDefault();
      sendEvent(inputId, viewerId, notesId, k.toLowerCase());
    });
  }

  var plates = {};
  function plateState(id) {
    return plates[id] || (plates[id] = { fill: {}, frame: {}, framePx: 3, labelled: [], hidden: [], selected: [] });
  }
  function asSet(a) { var o = {}; (a || []).forEach(function (w) { o[w] = true; }); return o; }
  function applyPlate(id) {
    var g = document.getElementById(id), st = plates[id];
    if (!g || !st) return;
    var lab = asSet(st.labelled), hid = asSet(st.hidden), sel = asSet(st.selected);
    g.querySelectorAll("[data-well]").forEach(function (el) {
      var w = el.getAttribute("data-well");
      var fill = el.querySelector(".imgplate-fill"), frame = el.querySelector(".imgplate-frame");
      if (fill) fill.style.background = st.fill[w] || "transparent";
      if (frame) frame.style.border = st.frame[w] ? st.framePx + "px solid " + st.frame[w] : "none";
      el.classList.toggle("imgplate-labelled", !!lab[w]);
      el.classList.toggle("imgplate-selected", !!sel[w]);
      el.classList.toggle("imgplate-hidden", !!hid[w]);
      var img = el.querySelector("img");
      if (img) img.style.visibility = hid[w] ? "hidden" : "";
    });
  }

  window.DiagViewer = { get: getViewer, bindKeys: bindKeys, sendEvent: sendEvent, instances: instances };

  function register() {
    Shiny.addCustomMessageHandler("diagviewer-load", function (msg) { loadWhenReady(msg, 0); });
    // plate grid styling (fill / frame colours, scored, hidden and selected
    // wells) is applied here so the 384 tiles are not rebuilt by the server
    // on every change; the last state is re-applied when the grid re-renders
    Shiny.addCustomMessageHandler("diagplate-select", function (msg) {
      plateState(msg.grid).selected = msg.wells || [];
      applyPlate(msg.grid);
    });
    Shiny.addCustomMessageHandler("diagplate-style", function (msg) {
      var st = plateState(msg.grid);
      st.fill = msg.fill || {}; st.frame = msg.frame || {}; st.framePx = msg.framePx || 3;
      st.labelled = msg.labelled || []; st.hidden = msg.hidden || [];
      applyPlate(msg.grid);
    });
    if (window.jQuery) jQuery(document).on("shiny:value", function (e) {
      if (plates[e.name]) setTimeout(function () { applyPlate(e.name); }, 0);
    });
    Shiny.addCustomMessageHandler("diagviewer-calib", function (msg) {
      var v = instances[msg.id];
      if (v && v.channels.length && v.root.isConnected) v.setCalib(msg.calib, false);
    });
    Shiny.addCustomMessageHandler("diagviewer-bindkeys", function (msg) { bindKeys(msg.input, msg.scope, msg.viewer, msg.notes); });
  }
  if (window.Shiny && Shiny.addCustomMessageHandler) register();
  else document.addEventListener("DOMContentLoaded", register);
})();
