const canvas = document.getElementById('sim');
const ctx = canvas.getContext('2d');
const viewTitle = document.getElementById('viewTitle');
const slidersEl = document.getElementById('sliders');

const runBtn = document.getElementById('runBtn');
const resetBtn = document.getElementById('resetBtn');
const modeBtn = document.getElementById('modeBtn');
const viewBtn = document.getElementById('viewBtn');
const saveBtn = document.getElementById('saveBtn');
const sizeSel = document.getElementById('sizeSel');
const monitor = document.getElementById('monitor');

let running = false;
let started = false;
let clickMode = 'poke';
let viewMode = 'energy';
let size = 256;

const params = {
  dt: 1.0,
  D_E: 0.02,
  kappa: 0.8,
  A_adv: 0.2,
  theta: 0.32,
  alpha: 0.05,
  beta: 0.12,
  damping: 0.001,
  floor: 0.00005,
  mass_decay: 0.0002,
  mass_release: 0.0006,
  D_u: 0.16,
  D_v: 0.08,
  F: 0.035,
  k: 0.065,
  poke: 1.2,
  poke_sigma: 5.0,
  wave_c2: 0.8,
  wave_damp: 0.25,
  wave_couple: 0.01,
  display_gamma: 1.0,
  bubble_thresh: 0.15,
  gravity_iters: 12
};

const sliderDefs = [
  ['dt', 0.25, 3.0, 0.5],
  ['D_E', 0.0, 0.2, 0.001],
  ['kappa', 0.0, 3.0, 0.01],
  ['A_adv', 0.0, 1.5, 0.01],
  ['theta', 0.0, 1.0, 0.01],
  ['alpha', 0.0, 0.5, 0.01],
  ['beta', 0.0, 0.5, 0.01],
  ['damping', 0.0, 0.02, 0.0005],
  ['floor', 0.0, 0.005, 0.00005],
  ['mass_decay', 0.0, 0.01, 0.0001],
  ['mass_release', 0.0, 0.01, 0.0001],
  ['D_u', 0.0, 0.3, 0.001],
  ['D_v', 0.0, 0.3, 0.001],
  ['F', 0.0, 0.08, 0.0005],
  ['k', 0.0, 0.08, 0.0005],
  ['poke', 0.0, 5.0, 0.1],
  ['poke_sigma', 1.0, 10.0, 0.1],
  ['wave_c2', 0.0, 2.0, 0.01],
  ['wave_damp', 0.0, 1.0, 0.01],
  ['wave_couple', 0.0, 0.05, 0.001],
  ['display_gamma', 0.5, 2.5, 0.01],
  ['bubble_thresh', 0.0, 1.0, 0.01],
  ['gravity_iters', 0, 50, 1]
];

function buildSliders() {
  slidersEl.innerHTML = '';
  sliderDefs.forEach(([name, min, max, step]) => {
    const wrap = document.createElement('div');
    wrap.className = 'slider';

    const label = document.createElement('span');
    label.textContent = name;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = params[name];

    const value = document.createElement('span');
    value.textContent = Number(params[name]).toFixed(4);

    input.addEventListener('input', () => {
      params[name] = input.type === 'range' ? parseFloat(input.value) : input.value;
      value.textContent = Number(params[name]).toFixed(4);
    });

    wrap.appendChild(label);
    wrap.appendChild(input);
    wrap.appendChild(value);
    slidersEl.appendChild(wrap);
  });
}

let u, v, m, wField, wDot, phi, phiNew;
let imgData, imgBuf;

function allocFields(n) {
  const N = n * n;
  u = new Float32Array(N);
  v = new Float32Array(N);
  m = new Float32Array(N);
  wField = new Float32Array(N);
  wDot = new Float32Array(N);
  phi = new Float32Array(N);
  phiNew = new Float32Array(N);
  imgData = ctx.createImageData(n, n);
  imgBuf = imgData.data;

  for (let i = 0; i < N; i++) {
    u[i] = 1.0;
    v[i] = 0.0;
    m[i] = 0.0;
    wField[i] = 0.0;
    wDot[i] = 0.0;
  }

  // noise seeds
  for (let i = 0; i < N; i++) {
    v[i] += Math.random() * 0.01;
  }
  for (let i = 0; i < 6; i++) {
    const cx = Math.floor(Math.random() * n);
    const cy = Math.floor(Math.random() * n);
    const sigma = 3 + Math.random() * 3;
    const amp = 0.12 + Math.random() * 0.1;
    addGaussian(v, n, cx, cy, sigma, amp);
  }
  for (let i = 0; i < N; i++) {
    v[i] = Math.min(1.0, Math.max(0.0, v[i]));
    u[i] = 1.0 - v[i];
  }
}

function idx(x, y, n) {
  x = (x + n) % n;
  y = (y + n) % n;
  return y * n + x;
}

function addGaussian(field, n, cx, cy, sigma, amp) {
  const r = Math.max(2, Math.floor(sigma * 3));
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const dist2 = dx * dx + dy * dy;
      const val = Math.exp(-dist2 / (2 * sigma * sigma)) * amp;
      const ii = idx(cx + dx, cy + dy, n);
      field[ii] += val;
    }
  }
}

function addRingImpulse(field, n, cx, cy, sigma, amp) {
  const r = Math.max(2, Math.floor(sigma * 3));
  const s2 = sigma * sigma;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const dist2 = dx * dx + dy * dy;
      const norm = dist2 / (s2 + 1e-6);
      const val = (norm - 1.0) * Math.exp(-0.5 * norm) * amp;
      const ii = idx(cx + dx, cy + dy, n);
      field[ii] += val;
    }
  }
}

function laplaceAt(field, n, x, y) {
  const c = field[idx(x, y, n)];
  const n1 = field[idx(x, y - 1, n)];
  const s1 = field[idx(x, y + 1, n)];
  const w1 = field[idx(x - 1, y, n)];
  const e1 = field[idx(x + 1, y, n)];
  const nw = field[idx(x - 1, y - 1, n)];
  const ne = field[idx(x + 1, y - 1, n)];
  const sw = field[idx(x - 1, y + 1, n)];
  const se = field[idx(x + 1, y + 1, n)];
  return 0.05 * (nw + ne + sw + se) + 0.2 * (n1 + s1 + w1 + e1) - 1.0 * c;
}

function solvePoissonJacobi(n, source, iters) {
  const N = n * n;
  for (let i = 0; i < N; i++) {
    phi[i] = 0.0;
  }
  for (let iter = 0; iter < iters; iter++) {
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const s = source[idx(x, y, n)];
        const n1 = phi[idx(x, y - 1, n)];
        const s1 = phi[idx(x, y + 1, n)];
        const w1 = phi[idx(x - 1, y, n)];
        const e1 = phi[idx(x + 1, y, n)];
        phiNew[idx(x, y, n)] = 0.25 * (n1 + s1 + w1 + e1 - s);
      }
    }
    const tmp = phi; phi = phiNew; phiNew = tmp;
  }
}

function step() {
  const n = size;
  const dt = params.dt;
  const N = n * n;

  // wave field
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = idx(x, y, n);
      const lapW = laplaceAt(wField, n, x, y);
      wDot[i] += (params.wave_c2 * lapW - params.wave_damp * wDot[i]) * dt;
    }
  }
  for (let i = 0; i < N; i++) {
    wField[i] += wDot[i] * dt;
  }

  // diffusion
  const lapV = new Float32Array(N);
  const lapU = new Float32Array(N);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = idx(x, y, n);
      lapV[i] = laplaceAt(v, n, x, y);
      lapU[i] = laplaceAt(u, n, x, y);
    }
  }
  for (let i = 0; i < N; i++) {
    v[i] += params.D_E * lapV[i] * dt;
  }

  // reaction
  for (let i = 0; i < N; i++) {
    const uvv = u[i] * v[i] * v[i];
    u[i] += (params.D_u * lapU[i] - uvv + params.F * (1.0 - u[i])) * dt;
    v[i] += (params.D_v * lapV[i] + uvv - (params.F + params.k) * v[i]) * dt;
    v[i] += params.wave_couple * wField[i];
    u[i] = Math.min(1.0, Math.max(0.0, u[i]));
    v[i] = Math.max(0.0, v[i]);
  }

  // freeze
  const dM = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (v[i] > params.theta) {
      const dm = params.alpha * v[i] * dt;
      v[i] -= dm;
      m[i] += dm;
      dM[i] = dm;
    }
  }

  if (params.beta > 0) {
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const i = idx(x, y, n);
        const pulse = laplaceAt(dM, n, x, y);
        if (pulse > 0) {
          v[i] += params.beta * pulse;
        }
      }
    }
  }

  // gravity via Poisson
  const meanM = m.reduce((a, b) => a + b, 0) / N;
  const source = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    source[i] = params.kappa * (m[i] - meanM);
  }
  solvePoissonJacobi(n, source, params.gravity_iters | 0);

  // advection
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = idx(x, y, n);
      const gx = -0.5 * (phi[idx(x + 1, y, n)] - phi[idx(x - 1, y, n)]);
      const gy = -0.5 * (phi[idx(x, y + 1, n)] - phi[idx(x, y - 1, n)]);
      const fx = v[i] * gx;
      const fy = v[i] * gy;
      const div = 0.5 * (fx - v[idx(x - 1, y, n)] * (-0.5 * (phi[idx(x, y, n)] - phi[idx(x - 2, y, n)])))
                + 0.5 * (fy - v[idx(x, y - 1, n)] * (-0.5 * (phi[idx(x, y, n)] - phi[idx(x, y - 2, n)])));
      v[i] -= params.A_adv * div * dt;
      if (v[i] < 0) v[i] = 0;
    }
  }

  // cleanup
  for (let i = 0; i < N; i++) {
    v[i] *= (1.0 - params.damping * dt);
    if (v[i] < params.floor) v[i] = 0;
    if (params.mass_decay > 0) {
      m[i] *= (1.0 - params.mass_decay * dt);
    }
    if (params.mass_release > 0) {
      const rel = params.mass_release * m[i] * dt;
      m[i] -= rel;
      v[i] += rel;
    }
  }
}

function render() {
  const n = size;
  const N = n * n;
  let maxVal = 1e-6;

  if (viewMode === 'energy') {
    for (let i = 0; i < N; i++) {
      if (v[i] > maxVal) maxVal = v[i];
    }
  } else if (viewMode === 'mass') {
    for (let i = 0; i < N; i++) {
      if (m[i] > maxVal) maxVal = m[i];
    }
  } else if (viewMode === 'potential') {
    for (let i = 0; i < N; i++) {
      const p = Math.abs(phi[i]);
      if (p > maxVal) maxVal = p;
    }
  }

  for (let i = 0; i < N; i++) {
    let val = 0;
    if (viewMode === 'energy') {
      val = v[i] / maxVal;
    } else if (viewMode === 'mass') {
      val = m[i] / maxVal;
    } else if (viewMode === 'potential') {
      val = Math.abs(phi[i]) / maxVal;
    } else {
      const vNorm = v[i] / maxVal;
      const mNorm = m[i] / maxVal;
      const r = Math.min(1, mNorm * 1.2);
      const g = Math.min(1, vNorm * 1.2);
      const b = Math.min(1, vNorm + 0.3 * mNorm);
      imgBuf[i * 4 + 0] = Math.floor(r * 255);
      imgBuf[i * 4 + 1] = Math.floor(g * 255);
      imgBuf[i * 4 + 2] = Math.floor(b * 255);
      imgBuf[i * 4 + 3] = 255;
      continue;
    }
    val = Math.pow(val, 1.0 / params.display_gamma);
    const c = Math.floor(Math.max(0, Math.min(1, val)) * 255);
    imgBuf[i * 4 + 0] = c;
    imgBuf[i * 4 + 1] = c;
    imgBuf[i * 4 + 2] = c;
    imgBuf[i * 4 + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
}

let lastTime = null;
let fpsEma = 0;
let frameCount = 0;

function loop(ts) {
  if (lastTime === null) lastTime = ts;
  const dtWall = (ts - lastTime) / 1000;
  lastTime = ts;
  const fps = dtWall > 0 ? 1 / dtWall : 0;
  fpsEma = fpsEma === 0 ? fps : fpsEma * 0.95 + fps * 0.05;

  if (running) {
    step();
    frameCount++;
  }
  render();
  monitor.textContent = `mode: ${clickMode}\nview: ${viewMode}\nframe: ${frameCount}\nFPS: ${fps.toFixed(1)} (EMA ${fpsEma.toFixed(1)})`;

  requestAnimationFrame(loop);
}

canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor(((e.clientX - rect.left) / rect.width) * size);
  const y = Math.floor(((e.clientY - rect.top) / rect.height) * size);
  if (!started) {
    started = true;
    running = true;
    runBtn.textContent = 'Pause';
  }
  if (clickMode === 'poke') {
    addRingImpulse(wDot, size, x, y, params.poke_sigma, params.poke);
  } else {
    addGaussian(v, size, x, y, params.poke_sigma, params.poke);
    for (let i = 0; i < size * size; i++) {
      v[i] = Math.min(1, Math.max(0, v[i]));
      u[i] = 1.0 - v[i];
    }
  }
});

runBtn.addEventListener('click', () => {
  running = !running;
  runBtn.textContent = running ? 'Pause' : 'Run';
});

resetBtn.addEventListener('click', () => {
  allocFields(size);
  started = false;
  running = false;
  frameCount = 0;
  runBtn.textContent = 'Run';
});

modeBtn.addEventListener('click', () => {
  clickMode = clickMode === 'poke' ? 'seed' : 'poke';
  modeBtn.textContent = clickMode === 'poke' ? 'Mode: Poke' : 'Mode: Seed';
});

viewBtn.addEventListener('click', () => {
  if (viewMode === 'energy') viewMode = 'mass';
  else if (viewMode === 'mass') viewMode = 'potential';
  else if (viewMode === 'potential') viewMode = 'composite';
  else viewMode = 'energy';
  viewBtn.textContent = `View: ${viewMode[0].toUpperCase()}${viewMode.slice(1)}`;
  viewTitle.textContent = `EMIT (${viewBtn.textContent.replace('View: ', '')})`;
});

saveBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = `emit_${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
});

sizeSel.addEventListener('change', () => {
  size = parseInt(sizeSel.value, 10);
  canvas.width = size;
  canvas.height = size;
  allocFields(size);
  started = false;
  running = false;
  frameCount = 0;
  runBtn.textContent = 'Run';
});

buildSliders();
allocFields(size);
requestAnimationFrame(loop);
