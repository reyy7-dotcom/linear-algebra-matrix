# ============================================================
# APLIKASI ALJABAR LINIER INTERAKTIF - Streamlit
# Fitur: Eigen, Eliminasi Gauss, Determinan & Invers,
#        Dekomposisi LU, Regresi Linear
# Jalankan: streamlit run app_matriks_uas.py
# ============================================================

import streamlit as st
import streamlit.components.v1
import numpy as np
import pandas as pd
import io

st.set_page_config(
    page_title="Aljabar Linier Interaktif",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-title { font-size: 1.9rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin: 0 0 6px 0; }
.hero-sub { font-size: 0.9rem; color: rgba(255,255,255,0.5); margin: 0; }

.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #60a5fa;
    margin: 28px 0 10px 0; display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: rgba(96,165,250,0.2); }

.info-card {
    background: rgba(96,165,250,0.06); border: 1px solid rgba(96,165,250,0.18);
    border-left: 3px solid #60a5fa; border-radius: 10px;
    padding: 14px 18px; font-size: 0.875rem; color: #e2e8f0;
    margin-bottom: 18px; line-height: 1.6;
}
.step-card {
    background: #0f172a; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;
}
.step-card .step-title { color: #60a5fa; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.result-box {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 1px solid rgba(139,92,246,0.3); border-radius: 12px;
    padding: 20px 24px; text-align: center; margin-top: 12px;
}
.result-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #34d399; }
.eigen-card {
    background: #0f172a; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 20px 24px; margin-bottom: 14px;
    position: relative; overflow: hidden;
}
.eigen-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #60a5fa, #a78bfa);
}
.eigen-value { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
.eigen-badge { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 2px 10px; border-radius: 20px; }
.badge-real { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-complex { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid rgba(255,255,255,0.07); }
[data-testid="stMetric"] { background: #0f172a; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigasi ──────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔢 Aljabar Linier")
    st.markdown("---")
    menu = st.radio("Pilih Fitur", [
        "🏠 Beranda",
        "📊 Analisis Eigen",
        "➗ Eliminasi Gauss",
        "🔲 Determinan & Invers",
        "🔺 Dekomposisi LU",
        "📈 Regresi Linear",
    ])
    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#475569'>Dibuat dengan NumPy + Streamlit</p>", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Aljabar Linier Interaktif</h1>
    <p class="hero-sub">Eigen · Eliminasi Gauss · Determinan & Invers · Dekomposisi LU · Regresi Linear</p>
</div>
""", unsafe_allow_html=True)

def format_kompleks(z, d=4):
    r, i = z.real, z.imag
    t = "+" if i >= 0 else "−"
    return f"{r:.{d}f} {t} {abs(i):.{d}f}i"

def format_matrix(mat, decimals=4):
    df = pd.DataFrame(
        np.round(mat, decimals),
        columns=[f"x{i+1}" for i in range(mat.shape[1])],
        index=[f"b{i+1}" for i in range(mat.shape[0])]
    )
    return df

# ════════════════════════════════════════════════════════════
# BERANDA
# ════════════════════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.markdown('<p class="section-label">Fitur Tersedia</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="info-card">
        <b>📊 Analisis Eigen</b><br>
        Buat matriks acak 100×100 dengan seed terkunci, hitung nilai eigen dan vektor eigen,
        serta buktikan Teorema Trace secara otomatis.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        <b>🔲 Determinan & Invers</b><br>
        Hitung determinan matriks n×n dan invers-nya (jika ada).
        Tampilkan matriks hasil dalam tabel interaktif.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        <b>📈 Regresi Linear</b><br>
        Input data titik (x, y) dan hitung persamaan garis regresi
        terbaik menggunakan metode least squares.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
        <b>➗ Eliminasi Gauss</b><br>
        Selesaikan sistem persamaan linear Ax = b menggunakan
        eliminasi Gauss dengan tampilan langkah-langkah detail.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        <b>🔺 Dekomposisi LU</b><br>
        Faktorkan matriks A menjadi L (lower triangular) dan
        U (upper triangular) dengan metode Doolittle.
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-label">Mulai</p>', unsafe_allow_html=True)
    st.info("Pilih fitur dari sidebar kiri untuk memulai.")

# ════════════════════════════════════════════════════════════
# ANALISIS EIGEN
# ════════════════════════════════════════════════════════════
elif menu == "📊 Analisis Eigen":
    st.markdown('<p class="section-label">Kontrol</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        seed_val = st.number_input("Random Seed", 0, 99999, 42)
    with col2:
        ukuran = st.selectbox("Ukuran Matriks (n×n)", [50, 100, 150, 200], index=1)
    with col3:
        rmin = st.number_input("Nilai Min", 1, 50, 1)
        rmax = st.number_input("Nilai Maks", 2, 100, 10)

    if st.button("▶ Jalankan Komputasi", type="primary"):
        with st.spinner("Menghitung..."):
            np.random.seed(seed_val)
            M = np.random.randint(rmin, rmax+1, size=(ukuran, ukuran))
            vals, vecs = np.linalg.eig(M)
            trace_m = int(np.trace(M))
            trace_e = vals.real.sum()
            buf = io.BytesIO()
            np.savetxt(buf, M, fmt="%d", delimiter="\t")

        st.markdown('<p class="section-label">Info Matriks</p>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Dimensi", f"{ukuran}×{ukuran}")
        c2.metric("Total Elemen", f"{ukuran*ukuran:,}")
        c3.metric("Rentang", f"{rmin}–{rmax}")
        c4.metric("Seed", str(seed_val))

        st.markdown('<p class="section-label">Preview Matriks</p>', unsafe_allow_html=True)
        df = pd.DataFrame(M, columns=[f"K{i+1}" for i in range(ukuran)], index=[f"B{i+1}" for i in range(ukuran)])
        st.dataframe(df, height=300, use_container_width=True)
        st.download_button("⬇️ Unduh matriks_uas.txt", buf.getvalue(), "matriks_uas.txt", "text/plain")

        st.markdown('<p class="section-label">3 Nilai Eigen Pertama</p>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for idx, col in enumerate([c1,c2,c3]):
            λ = vals[idx]
            col.metric(f"λ{idx+1}", f"{λ.real:.4f}", f"Im: {λ.imag:+.4f}" if abs(λ.imag)>1e-10 else "Murni Real")

        for idx in range(3):
            λ = vals[idx]
            is_c = abs(λ.imag) >= 1e-10
            badge_cls = "badge-complex" if is_c else "badge-real"
            badge_txt = "Kompleks" if is_c else "Real"
            badge_html = "<span class='eigen-badge " + badge_cls + "'> " + badge_txt + "</span>"
            v_prev = " &nbsp;|&nbsp; ".join([format_kompleks(vecs[j,idx]) for j in range(5)])
            idx1 = idx + 1
            eigen_str = format_kompleks(λ)
            card_html = "<div class='eigen-card'>"
            card_html += "<div style='font-size:.68rem;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px'>Nilai Eigen ke-" + str(idx1) + "</div>"
            card_html += "<div class='eigen-value'>" + eigen_str + "</div>"
            card_html += badge_html
            card_html += "<div style='margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.06);font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>5 Elemen Pertama Vektor Eigen v" + str(idx1) + "</div>"
            card_html += "<div style='font-family:JetBrains Mono,monospace;font-size:.78rem;color:#94a3b8'>" + v_prev + "</div></div>"
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown('<p class="section-label">Pembuktian Teorema Trace</p>', unsafe_allow_html=True)
        st.latex(r"\mathrm{tr}(A) = \sum_{i=1}^{n} \lambda_i")
        st.latex(r"\mathrm{tr}(A) = " + str(trace_m) + r"\quad \approx \quad \sum\lambda_i = " + f"{trace_e:.4f}")
        st.markdown(f"""<div class="result-box">
            <div style="font-size:.72rem;color:#a78bfa;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Verifikasi Trace</div>
            <div style="display:flex;justify-content:center;gap:40px">
                <div><div class="result-value">{trace_m}</div><div style="font-size:.8rem;color:rgba(255,255,255,.4)">tr(A)</div></div>
                <div style="font-size:2rem;color:#475569;align-self:center">=</div>
                <div><div class="result-value">{trace_e:.4f}</div><div style="font-size:.8rem;color:rgba(255,255,255,.4)">Σλᵢ</div></div>
            </div>
            <div style="margin-top:14px;font-family:'JetBrains Mono',monospace;color:#34d399">Δ = {abs(trace_m-trace_e):.2e} ✓</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# ELIMINASI GAUSS
# ════════════════════════════════════════════════════════════
elif menu == "➗ Eliminasi Gauss":
    st.markdown('<p class="section-label">Sistem Persamaan Linear Ax = b</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Atur ukuran matriks (Rows = Cols), klik <b>Set Matrix</b> untuk buat grid.
    Gunakan <b>Auto (+)</b> untuk isi angka positif acak, atau <b>Auto (+/-)</b> untuk campuran.
    Edit tiap sel manual, lalu klik <b>Calculate</b>.
    </div>""", unsafe_allow_html=True)

    gauss_html = """
<style>
.gauss-wrap { font-family: 'Inter', sans-serif; }
.gauss-controls { display:flex; align-items:center; gap:12px; margin-bottom:16px; flex-wrap:wrap; }
.gauss-controls label { color:#94a3b8; font-size:.85rem; font-weight:600; }
.gauss-controls input[type=number] {
    width:80px; padding:8px 10px; background:#1e293b; border:1px solid rgba(255,255,255,0.12);
    border-radius:8px; color:#f1f5f9; font-size:.9rem; text-align:center;
}
.gauss-controls input[type=number]:focus { outline:none; border-color:#60a5fa; }
.gauss-btn {
    padding:8px 18px; border:none; border-radius:8px; font-size:.82rem;
    font-weight:600; cursor:pointer; transition:all .15s;
}
.btn-set   { background:#3b82f6; color:#fff; }
.btn-auto1 { background:#7c3aed; color:#fff; }
.btn-auto2 { background:#6d28d9; color:#fff; }
.btn-clear { background:#374151; color:#e5e7eb; }
.btn-calc  { background:#059669; color:#fff; font-size:.9rem; padding:10px 28px; }
.gauss-btn:hover { opacity:.85; transform:translateY(-1px); }

.matrix-scroll {
    overflow:auto; max-height:420px; max-width:100%;
    border:1px solid rgba(255,255,255,0.08); border-radius:10px;
    background:#0f172a; padding:12px; margin-bottom:14px;
}
.matrix-title {
    color:#60a5fa; font-size:.72rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;
}
.matrix-grid { display:inline-block; }
.matrix-row  { display:flex; gap:4px; margin-bottom:4px; align-items:center; }
.row-label   { color:#475569; font-size:.7rem; width:28px; text-align:right; padding-right:6px; flex-shrink:0; }
.cell-input  {
    width:52px; height:36px; text-align:center; font-size:.8rem; font-weight:600;
    background:#1e293b; border:1px solid rgba(255,255,255,0.08);
    border-radius:6px; color:#f1f5f9; padding:0 4px;
    transition:border-color .15s;
}
.cell-input:focus { outline:none; border-color:#60a5fa; background:#1e3a5f; }
.cell-input.b-col { background:#1a1a3e; border-color:rgba(139,92,246,0.4); color:#c4b5fd; }
.cell-input.b-col:focus { border-color:#a78bfa; }
.sep-line { width:2px; height:36px; background:rgba(139,92,246,0.4); margin:0 4px; flex-shrink:0; }

.result-section { margin-top:16px; }
.result-title { color:#60a5fa; font-size:.72rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px; }
.sol-grid { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
.sol-card {
    background:#0f172a; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:10px 14px; min-width:90px; text-align:center;
}
.sol-var { font-size:.68rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.sol-val { font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:700; color:#34d399; }
.verif-box {
    background:linear-gradient(135deg,#0f172a,#1e1b4b);
    border:1px solid rgba(139,92,246,.3); border-radius:12px;
    padding:16px 20px; text-align:center; margin-top:12px;
}
.verif-status { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; }
.status-ok  { color:#34d399; }
.status-err { color:#f87171; }
.error-msg  { color:#f87171; background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.3); border-radius:8px; padding:10px 14px; font-size:.85rem; margin-top:10px; }
</style>

<div class="gauss-wrap">
  <div class="gauss-controls">
    <label>Rows:</label>
    <input type="number" id="inp-rows" value="4" min="2" max="100">
    <label>Cols:</label>
    <input type="number" id="inp-cols" value="4" min="2" max="100" readonly style="opacity:.5">
    <button class="gauss-btn btn-set"   onclick="setMatrix()">Set Matrix</button>
    <button class="gauss-btn btn-auto1" onclick="autoFill(true)">⚡ Auto (+/-)</button>
    <button class="gauss-btn btn-auto2" onclick="autoFill(false)">⚡ Auto (+)</button>
    <button class="gauss-btn btn-clear" onclick="clearMatrix()">Clear</button>
  </div>

  <div class="matrix-scroll" id="matrix-container">
    <div class="matrix-title">INPUT MATRIX [A | b] — KLIK SEL UNTUK EDIT</div>
    <div class="matrix-grid" id="matrix-grid"></div>
  </div>

  <div style="text-align:left;margin-bottom:12px">
    <button class="gauss-btn btn-calc" onclick="calculate()">▶ Calculate</button>
  </div>

  <div class="result-section" id="result-section" style="display:none">
    <div class="result-title">SOLUSI x</div>
    <div class="sol-grid" id="sol-grid"></div>
    <div class="verif-box" id="verif-box"></div>
  </div>
</div>

<script>
let N = 4;

function setMatrix() {
    const r = parseInt(document.getElementById('inp-rows').value);
    if (r < 2 || r > 100) { alert('Ukuran harus antara 2–100'); return; }
    N = r;
    document.getElementById('inp-cols').value = N;
    buildGrid();
    document.getElementById('result-section').style.display = 'none';
}

function buildGrid(values) {
    const grid = document.getElementById('matrix-grid');
    grid.innerHTML = '';
    // Header row
    const hdr = document.createElement('div');
    hdr.className = 'matrix-row';
    hdr.innerHTML = '<div class="row-label"></div>';
    for (let j = 0; j < N; j++) {
        const h = document.createElement('div');
        h.style.cssText = 'width:52px;text-align:center;font-size:.68rem;color:#64748b;font-weight:700;padding-bottom:4px';
        h.textContent = 'x' + (j+1);
        hdr.appendChild(h);
    }
    hdr.innerHTML += '<div class="sep-line" style="height:20px;margin-bottom:0"></div>';
    const bh = document.createElement('div');
    bh.style.cssText = 'width:52px;text-align:center;font-size:.68rem;color:#a78bfa;font-weight:700;padding-bottom:4px';
    bh.textContent = 'b';
    hdr.appendChild(bh);
    grid.appendChild(hdr);

    // Data rows
    for (let i = 0; i < N; i++) {
        const row = document.createElement('div');
        row.className = 'matrix-row';
        const lbl = document.createElement('div');
        lbl.className = 'row-label';
        lbl.textContent = 'P' + (i+1);
        row.appendChild(lbl);
        for (let j = 0; j < N; j++) {
            const inp = document.createElement('input');
            inp.type = 'number'; inp.className = 'cell-input';
            inp.id = 'c_' + i + '_' + j;
            inp.value = values ? (values[i][j] || 0) : 0;
            row.appendChild(inp);
        }
        const sep = document.createElement('div'); sep.className = 'sep-line';
        row.appendChild(sep);
        const b = document.createElement('input');
        b.type = 'number'; b.className = 'cell-input b-col';
        b.id = 'b_' + i; b.value = values ? (values[i][N] || 0) : 0;
        row.appendChild(b);
        grid.appendChild(row);
    }
}

function autoFill(allowNeg) {
    for (let i = 0; i < N; i++) {
        let rowSum = 0;
        for (let j = 0; j < N; j++) {
            if (i === j) continue;
            const v = allowNeg ? (Math.floor(Math.random()*19)-9) : (Math.floor(Math.random()*9)+1);
            document.getElementById('c_'+i+'_'+j).value = v;
            rowSum += Math.abs(v);
        }
        // Diagonal dominan
        document.getElementById('c_'+i+'_'+i).value = rowSum + Math.floor(Math.random()*5) + 1;
        document.getElementById('b_'+i).value = allowNeg ? (Math.floor(Math.random()*19)-9) : (Math.floor(Math.random()*9)+1);
    }
    document.getElementById('result-section').style.display = 'none';
}

function clearMatrix() {
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) document.getElementById('c_'+i+'_'+j).value = 0;
        document.getElementById('b_'+i).value = 0;
    }
    document.getElementById('result-section').style.display = 'none';
}

function calculate() {
    // Baca matriks augmented
    let M = [];
    for (let i = 0; i < N; i++) {
        let row = [];
        for (let j = 0; j < N; j++) row.push(parseFloat(document.getElementById('c_'+i+'_'+j).value) || 0);
        row.push(parseFloat(document.getElementById('b_'+i).value) || 0);
        M.push(row);
    }
    const A_orig = M.map(r => r.slice(0, N));
    const b_orig = M.map(r => r[N]);

    // Eliminasi Gauss dengan partial pivoting
    for (let col = 0; col < N; col++) {
        // Cari pivot terbesar
        let maxRow = col;
        for (let r = col+1; r < N; r++) if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
        [M[col], M[maxRow]] = [M[maxRow], M[col]];
        if (Math.abs(M[col][col]) < 1e-12) {
            showError('Matriks singular — tidak ada solusi unik. Coba ubah nilai.');
            return;
        }
        for (let row = col+1; row < N; row++) {
            const f = M[row][col] / M[col][col];
            for (let k = col; k <= N; k++) M[row][k] -= f * M[col][k];
        }
    }

    // Substitusi mundur
    let x = new Array(N).fill(0);
    for (let i = N-1; i >= 0; i--) {
        x[i] = M[i][N];
        for (let j = i+1; j < N; j++) x[i] -= M[i][j] * x[j];
        x[i] /= M[i][i];
    }

    // Verifikasi
    let maxErr = 0;
    for (let i = 0; i < N; i++) {
        let ax = 0;
        for (let j = 0; j < N; j++) ax += A_orig[i][j] * x[j];
        maxErr = Math.max(maxErr, Math.abs(ax - b_orig[i]));
    }

    // Tampilkan solusi
    const solGrid = document.getElementById('sol-grid');
    solGrid.innerHTML = '';
    const showAll = N <= 50;
    const limit = showAll ? N : 20;
    for (let i = 0; i < limit; i++) {
        const card = document.createElement('div');
        card.className = 'sol-card';
        card.innerHTML = '<div class="sol-var">x' + (i+1) + '</div><div class="sol-val">' + x[i].toFixed(4) + '</div>';
        solGrid.appendChild(card);
    }
    if (!showAll) {
        const more = document.createElement('div');
        more.style.cssText = 'color:#64748b;font-size:.8rem;align-self:center;padding:10px';
        more.textContent = '... dan ' + (N-20) + ' variabel lainnya';
        solGrid.appendChild(more);
    }

    const valid = maxErr < 1e-6;
    document.getElementById('verif-box').innerHTML =
        '<div style="font-size:.7rem;color:#a78bfa;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">Verifikasi Ax = b</div>' +
        '<div class="verif-status ' + (valid?'status-ok':'status-err') + '">' + (valid?'✓ VALID':'✗ GAGAL') + '</div>' +
        '<div style="margin-top:8px;color:#94a3b8;font-size:.82rem">Sistem ' + N + '×' + N + ' · Maks |Ax−b| = ' + maxErr.toExponential(2) + (valid?' · Solusi akurat ✓':'') + '</div>';

    document.getElementById('result-section').style.display = 'block';
}

function showError(msg) {
    document.getElementById('result-section').style.display = 'block';
    document.getElementById('sol-grid').innerHTML = '<div class="error-msg">⚠️ ' + msg + '</div>';
    document.getElementById('verif-box').innerHTML = '';
}

// Init
buildGrid();
</script>
"""

    st.components.v1.html(gauss_html, height=780, scrolling=True)

# ════════════════════════════════════════════════════════════
# DETERMINAN & INVERS
# ════════════════════════════════════════════════════════════
elif menu == "🔲 Determinan & Invers":
    st.markdown('<p class="section-label">Input Matriks</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Masukkan matriks persegi n×n. Pisahkan angka dengan spasi, baris baru untuk baris berikutnya.
    </div>""", unsafe_allow_html=True)

    default_mat = "4 7 2 1\n1 3 1 2\n2 5 3 1\n3 6 4 8"
    raw = st.text_area("Matriks A", value=default_mat, height=130)

    if st.button("▶ Hitung Determinan & Invers", type="primary"):
        try:
            rows = [[float(x) for x in r.split()] for r in raw.strip().split('\n') if r.strip()]
            A = np.array(rows)
            n = A.shape[0]
            if A.shape[0] != A.shape[1]:
                st.error("Matriks harus persegi (n×n)!")
            else:
                det = np.linalg.det(A)

                st.markdown('<p class="section-label">Matriks Input</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(A,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                st.markdown('<p class="section-label">Determinan</p>', unsafe_allow_html=True)
                st.latex(r"\det(A) = " + f"{det:.6f}")
                st.markdown(f"""<div class="result-box">
                    <div style="font-size:.72rem;color:#a78bfa;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">det(A)</div>
                    <div class="result-value">{det:.6f}</div>
                    <div style="margin-top:8px;font-size:.82rem;color:{'#34d399' if abs(det)>1e-10 else '#f87171'}">
                    {"✓ Matriks non-singular — invers ada" if abs(det)>1e-10 else "✗ Matriks singular — invers tidak ada"}
                    </div>
                </div>""", unsafe_allow_html=True)

                if abs(det) > 1e-10:
                    inv = np.linalg.inv(A)
                    st.markdown('<p class="section-label">Matriks Invers A⁻¹</p>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(np.round(inv,6), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                    # Verifikasi A × A⁻¹ = I
                    I_check = np.round(A @ inv, 4)
                    st.markdown('<p class="section-label">Verifikasi: A × A⁻¹ = I</p>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(I_check, columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                    st.latex(r"A \times A^{-1} = I")
                    if np.allclose(A @ inv, np.eye(n)):
                        st.success("✓ Terverifikasi — A × A⁻¹ = I")

        except Exception as e:
            st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════
# DEKOMPOSISI LU
# ════════════════════════════════════════════════════════════
elif menu == "🔺 Dekomposisi LU":
    st.markdown('<p class="section-label">Faktorisasi LU (Doolittle)</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Dekomposisi LU memfaktorkan matriks A menjadi A = P·L·U dimana:
    <b>P</b> = matriks permutasi, <b>L</b> = lower triangular, <b>U</b> = upper triangular.
    </div>""", unsafe_allow_html=True)

    default_lu = "2 1 1\n4 3 3\n8 7 9"
    raw = st.text_area("Matriks A", value=default_lu, height=110)

    if st.button("▶ Dekomposisi LU", type="primary"):
        try:
            from scipy import linalg as la
            rows = [[float(x) for x in r.split()] for r in raw.strip().split('\n') if r.strip()]
            A = np.array(rows)
            n = A.shape[0]

            if A.shape[0] != A.shape[1]:
                st.error("Matriks harus persegi!")
            else:
                import scipy.linalg
                P, L, U = scipy.linalg.lu(A)

                st.markdown('<p class="section-label">Matriks A (Input)</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(A,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                st.latex(r"A = P \cdot L \cdot U")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Matriks P (Permutasi)**")
                    st.dataframe(pd.DataFrame(np.round(P,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                with c2:
                    st.markdown("**Matriks L (Lower)**")
                    st.dataframe(pd.DataFrame(np.round(L,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                with c3:
                    st.markdown("**Matriks U (Upper)**")
                    st.dataframe(pd.DataFrame(np.round(U,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                # Verifikasi
                PLU = P @ L @ U
                st.markdown('<p class="section-label">Verifikasi: P·L·U = A</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(PLU,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                if np.allclose(PLU, A):
                    st.success("✓ Terverifikasi — P·L·U = A")

        except ImportError:
            # Fallback tanpa scipy
            try:
                rows = [[float(x) for x in r.split()] for r in raw.strip().split('\n') if r.strip()]
                A = np.array(rows, dtype=float)
                n = A.shape[0]
                L = np.eye(n)
                U = A.copy()
                for k in range(n-1):
                    for i in range(k+1, n):
                        if abs(U[k,k]) < 1e-12:
                            st.error("Pivot nol — coba gunakan matriks lain.")
                            st.stop()
                        factor = U[i,k] / U[k,k]
                        L[i,k] = factor
                        U[i] -= factor * U[k]

                st.markdown('<p class="section-label">Matriks L (Lower Triangular)</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(L,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                st.markdown('<p class="section-label">Matriks U (Upper Triangular)</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(U,4), columns=[f"K{i+1}" for i in range(n)], index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                if np.allclose(L @ U, A):
                    st.success("✓ Terverifikasi — L·U = A")
            except Exception as e:
                st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════
# REGRESI LINEAR
# ════════════════════════════════════════════════════════════
elif menu == "📈 Regresi Linear":
    st.markdown('<p class="section-label">Regresi Linear (Least Squares)</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Masukkan data titik (x, y) — satu titik per baris, pisahkan x dan y dengan spasi.
    Metode least squares akan mencari garis terbaik: <b>y = mx + c</b>
    </div>""", unsafe_allow_html=True)

    default_data = "1 2.1\n2 3.9\n3 6.2\n4 7.8\n5 10.1\n6 11.9\n7 14.2\n8 16.1"
    raw = st.text_area("Data (x y)", value=default_data, height=180)

    if st.button("▶ Hitung Regresi Linear", type="primary"):
        try:
            pts = [[float(v) for v in r.split()] for r in raw.strip().split('\n') if r.strip()]
            data = np.array(pts)
            x, y = data[:,0], data[:,1]
            n = len(x)

            # Susun matriks normal equations: A^T A x = A^T b
            A = np.column_stack([x, np.ones(n)])
            ATA = A.T @ A
            ATb = A.T @ y
            coeffs = np.linalg.solve(ATA, ATb)
            m, c = coeffs

            y_pred = m * x + c
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot

            st.markdown('<p class="section-label">Persamaan Regresi</p>', unsafe_allow_html=True)
            st.latex(f"\\hat{{y}} = {m:.4f}x + ({c:.4f})")
            st.latex(f"R^2 = {r2:.4f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Gradien (m)", f"{m:.4f}")
            c2.metric("Intercept (c)", f"{c:.4f}")
            c3.metric("R² (kecocokan)", f"{r2:.4f}")

            st.markdown('<p class="section-label">Matriks Normal Equations AᵀA</p>', unsafe_allow_html=True)
            st.latex(r"(A^T A)\mathbf{x} = A^T \mathbf{b}")
            st.dataframe(pd.DataFrame(np.round(ATA,4), columns=["Σx²","Σx"], index=["Σx²","Σx"]), use_container_width=True)

            st.markdown('<p class="section-label">Tabel Data & Prediksi</p>', unsafe_allow_html=True)
            df_res = pd.DataFrame({
                "x": x, "y (aktual)": y,
                "ŷ (prediksi)": np.round(y_pred,4),
                "Residual (y−ŷ)": np.round(y-y_pred,4)
            })
            st.dataframe(df_res, use_container_width=True)

            st.markdown(f"""<div class="result-box">
                <div style="font-size:.72rem;color:#a78bfa;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">Hasil Regresi</div>
                <div class="result-value">ŷ = {m:.4f}x + ({c:.4f})</div>
                <div style="margin-top:8px;color:#94a3b8;font-size:.85rem">R² = {r2:.4f} — {'Sangat baik ✓' if r2>0.9 else 'Cukup baik' if r2>0.7 else 'Lemah'}</div>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='font-size:.75rem;color:#475569;text-align:center'>Aljabar Linier Interaktif · NumPy + Streamlit · Semua komputasi dijalankan secara lokal</p>", unsafe_allow_html=True)
