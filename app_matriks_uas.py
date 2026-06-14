# ============================================================
# APLIKASI ALJABAR LINIER INTERAKTIF - Streamlit
# Fitur: Eigen, Eliminasi Gauss, Determinan & Invers,
#        Dekomposisi LU, Regresi Linear
# SEMUA FITUR MENAMPILKAN LANGKAH-LANGKAH PENGERJAAN
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

/* Step-by-step styles */
.step-timeline {
    border-left: 2px solid rgba(96,165,250,0.3);
    margin: 12px 0 12px 12px;
    padding-left: 20px;
}
.step-item {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    position: relative;
}
.step-item::before {
    content: '';
    position: absolute;
    left: -27px;
    top: 18px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #60a5fa;
    border: 2px solid #0f172a;
}
.step-num {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 6px;
}
.step-op {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #fbbf24;
    margin-bottom: 8px;
    font-weight: 600;
}
.step-explain {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 10px;
    line-height: 1.5;
}
.mini-matrix {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: #1e293b;
    border-radius: 6px;
    padding: 8px 12px;
    display: inline-block;
    color: #e2e8f0;
}
.highlight-row { color: #60a5fa !important; font-weight: 700; }
.highlight-pivot { color: #f87171 !important; font-weight: 700; }
.highlight-result { color: #34d399 !important; font-weight: 700; }
.formula-box {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #c4b5fd;
    margin: 8px 0;
}
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

def mat_to_str(mat, decimals=3, highlight_row=None, highlight_col=None):
    """Render matriks sebagai HTML mini dengan highlight baris/kolom tertentu."""
    rows_html = []
    for i, row in enumerate(mat):
        cells = []
        for j, val in enumerate(row):
            cls = ""
            if highlight_row is not None and i == highlight_row:
                cls = " class='highlight-row'"
            elif highlight_col is not None and j == highlight_col:
                cls = " class='highlight-pivot'"
            cells.append(f"<span{cls}>{val:.{decimals}f}</span>")
        rows_html.append("  [" + "  ".join(cells) + "]")
    return "<br>".join(rows_html)

def render_step(step_num, operation, explanation, matrix=None, extra_html=""):
    html = f"""
    <div class="step-item">
        <div class="step-num">Langkah {step_num}</div>
        <div class="step-op">{operation}</div>
        <div class="step-explain">{explanation}</div>
        {f'<div class="mini-matrix">{mat_to_str(matrix)}</div>' if matrix is not None else ""}
        {extra_html}
    </div>
    """
    return html

# ════════════════════════════════════════════════════════════
# BERANDA
# ════════════════════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.markdown('<p class="section-label">Fitur Tersedia</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="info-card">
        <b>📊 Analisis Eigen</b><br>
        Buat matriks acak, hitung nilai eigen & vektor eigen, serta buktikan Teorema Trace.
        Dilengkapi <b>langkah-langkah</b> perhitungan karakteristik polinomial.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        <b>🔲 Determinan & Invers</b><br>
        Hitung determinan dan invers matriks n×n dengan <b>langkah ekspansi kofaktor</b>
        dan eliminasi bertahap yang ditampilkan secara detail.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        <b>📈 Regresi Linear</b><br>
        Hitung regresi dengan <b>langkah pembentukan normal equations</b>,
        perhitungan AᵀA, dan penyelesaian sistem.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
        <b>➗ Eliminasi Gauss</b><br>
        Selesaikan sistem Ax = b dengan <b>setiap operasi baris ditampilkan</b>:
        pivot, pertukaran baris, eliminasi, dan substitusi mundur.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        <b>🔺 Dekomposisi LU</b><br>
        Faktorkan A = P·L·U dengan <b>langkah pembentukan multiplier</b>,
        eliminasi per kolom, dan verifikasi akhir.
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-label">Mulai</p>', unsafe_allow_html=True)
    st.info("Pilih fitur dari sidebar kiri untuk memulai. Semua fitur menampilkan langkah-langkah pengerjaan lengkap.")

# ════════════════════════════════════════════════════════════
# ANALISIS EIGEN
# ════════════════════════════════════════════════════════════
elif menu == "📊 Analisis Eigen":
    st.markdown('<p class="section-label">Kontrol</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        seed_val = st.number_input("Random Seed", 0, 99999, 42)
    with col2:
        ukuran = st.selectbox("Ukuran Matriks (n×n)", [3, 4, 5, 50, 100, 150, 200], index=0)
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
        st.dataframe(df, height=min(300, ukuran*40+50), use_container_width=True)
        st.download_button("⬇️ Unduh matriks_uas.txt", buf.getvalue(), "matriks_uas.txt", "text/plain")

        # ── LANGKAH-LANGKAH EIGEN ──
        st.markdown('<p class="section-label">📋 Langkah-Langkah Perhitungan Eigen</p>', unsafe_allow_html=True)

        with st.expander("🔍 Lihat Langkah-Langkah Pengerjaan", expanded=True):
            steps_html = '<div class="step-timeline">'

            # Langkah 1: Definisi masalah
            steps_html += f"""
            <div class="step-item">
                <div class="step-num">Langkah 1</div>
                <div class="step-op">Definisi Masalah Eigen</div>
                <div class="step-explain">
                    Nilai eigen λ dan vektor eigen v dari matriks A memenuhi persamaan:<br>
                    <b>A·v = λ·v</b><br>
                    Atau equivalen: <b>(A − λI)·v = 0</b><br>
                    Agar solusi non-trivial ada, maka: <b>det(A − λI) = 0</b>
                </div>
                <div class="formula-box">Persamaan Karakteristik: det(A − λI) = 0</div>
            </div>
            """

            # Langkah 2: Trace & Det
            steps_html += f"""
            <div class="step-item">
                <div class="step-num">Langkah 2</div>
                <div class="step-op">Hitung Trace dan Determinan Matriks</div>
                <div class="step-explain">
                    Sebelum mencari nilai eigen, kita cek properti dasar matriks:
                </div>
                <div class="formula-box">
                    tr(A) = jumlah diagonal utama = {trace_m}<br>
                    (Teorema: tr(A) = Σλᵢ — jumlah semua nilai eigen)
                </div>
            </div>
            """

            # Langkah 3: Khusus matriks kecil, tunjukkan (A - λI) secara simbolik
            if ukuran <= 5:
                diag_str = " + ".join([f"a{i+1}{i+1}" for i in range(ukuran)])
                steps_html += f"""
                <div class="step-item">
                    <div class="step-num">Langkah 3</div>
                    <div class="step-op">Bentuk Matriks (A − λI)</div>
                    <div class="step-explain">
                        Kurangi setiap elemen diagonal dengan λ. Matriks A berukuran {ukuran}×{ukuran}:
                    </div>
                    <div class="formula-box">
                        Diagonal utama A = [{", ".join([str(M[i,i]) for i in range(ukuran)])}]<br>
                        Setelah dikurangi λ  → [{", ".join([f"{M[i,i]}−λ" for i in range(ukuran)])}]
                    </div>
                </div>
                """
                step_start = 4
            else:
                step_start = 3

            # Langkah berikut: Komputasi numerik
            steps_html += f"""
            <div class="step-item">
                <div class="step-num">Langkah {step_start}</div>
                <div class="step-op">Selesaikan det(A − λI) = 0 Secara Numerik</div>
                <div class="step-explain">
                    Untuk matriks {ukuran}×{ukuran}, menyelesaikan polinomial karakteristik secara manual
                    sangat kompleks. Digunakan algoritma numerik <b>QR Decomposition Iteration</b>
                    (numpy.linalg.eig) yang bekerja sebagai berikut:<br><br>
                    1. Reduksi matriks ke bentuk Hessenberg atas<br>
                    2. Iterasi QR: A = QR → A_baru = RQ (hingga konvergen)<br>
                    3. Nilai eigen muncul di diagonal saat matriks mendekati segitiga atas
                </div>
                <div class="formula-box">
                    Iterasi: Aₖ = QₖRₖ → Aₖ₊₁ = RₖQₖ<br>
                    Konvergen ke matriks segitiga atas (Schur form)
                </div>
            </div>
            """

            steps_html += f"""
            <div class="step-item">
                <div class="step-num">Langkah {step_start+1}</div>
                <div class="step-op">Hasil: {ukuran} Nilai Eigen Ditemukan</div>
                <div class="step-explain">
                    Setelah konvergen, diperoleh {ukuran} nilai eigen (termasuk kemungkinan kompleks).
                    Jumlah nilai eigen selalu sama dengan dimensi matriks (n={ukuran}).
                </div>
                <div class="formula-box">
                    λ₁ = {format_kompleks(vals[0])}<br>
                    λ₂ = {format_kompleks(vals[1])}<br>
                    {"λ₃ = " + format_kompleks(vals[2]) + "<br>" if ukuran >= 3 else ""}
                    {"... (total " + str(ukuran) + " nilai eigen)" if ukuran > 3 else ""}
                </div>
            </div>
            """

            steps_html += f"""
            <div class="step-item">
                <div class="step-num">Langkah {step_start+2}</div>
                <div class="step-op">Hitung Vektor Eigen untuk Setiap λ</div>
                <div class="step-explain">
                    Untuk setiap nilai eigen λᵢ, selesaikan sistem: <b>(A − λᵢI)·vᵢ = 0</b><br>
                    menggunakan eliminasi Gauss pada matriks yang sudah disingularkan.
                    Vektor eigen adalah solusi null space dari (A − λᵢI).
                </div>
                <div class="formula-box">
                    (A − λ₁I)·v₁ = 0  →  v₁ = [{", ".join([f"{vecs[j,0].real:.3f}" for j in range(min(3,ukuran))])}{"..." if ukuran>3 else ""}]<br>
                    Vektor eigen dinormalisasi: ||v|| = 1
                </div>
            </div>
            """

            steps_html += f"""
            <div class="step-item">
                <div class="step-num">Langkah {step_start+3}</div>
                <div class="step-op">Verifikasi: Cek Teorema Trace</div>
                <div class="step-explain">
                    Teorema menyatakan: <b>tr(A) = Σλᵢ</b><br>
                    Ini digunakan sebagai verifikasi cepat kebenaran nilai eigen.
                </div>
                <div class="formula-box">
                    tr(A) = {trace_m}<br>
                    Σλᵢ (bagian real) = {trace_e:.4f}<br>
                    Selisih Δ = {abs(trace_m - trace_e):.2e} {"✓ Sangat akurat" if abs(trace_m-trace_e) < 0.01 else "⚠ Ada pembulatan"}
                </div>
            </div>
            """

            steps_html += '</div>'
            st.markdown(steps_html, unsafe_allow_html=True)

        # Hasil eigen
        st.markdown('<p class="section-label">3 Nilai Eigen Pertama</p>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for idx, col in enumerate([c1,c2,c3]):
            λ = vals[idx]
            col.metric(f"λ{idx+1}", f"{λ.real:.4f}", f"Im: {λ.imag:+.4f}" if abs(λ.imag)>1e-10 else "Murni Real")

        for idx in range(min(3, ukuran)):
            λ = vals[idx]
            is_c = abs(λ.imag) >= 1e-10
            badge_cls = "badge-complex" if is_c else "badge-real"
            badge_txt = "Kompleks" if is_c else "Real"
            badge_html = "<span class='eigen-badge " + badge_cls + "'> " + badge_txt + "</span>"
            v_prev = " &nbsp;|&nbsp; ".join([format_kompleks(vecs[j,idx]) for j in range(min(5,ukuran))])
            idx1 = idx + 1
            eigen_str = format_kompleks(λ)
            # Verifikasi Av = λv
            Av = M @ vecs[:, idx]
            lv = λ * vecs[:, idx]
            err = np.max(np.abs(Av - lv))
            card_html = "<div class='eigen-card'>"
            card_html += "<div style='font-size:.68rem;color:#60a5fa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px'>Nilai Eigen ke-" + str(idx1) + "</div>"
            card_html += "<div class='eigen-value'>" + eigen_str + "</div>"
            card_html += badge_html
            card_html += "<div style='margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.06);font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>5 Elemen Pertama Vektor Eigen v" + str(idx1) + "</div>"
            card_html += "<div style='font-family:JetBrains Mono,monospace;font-size:.78rem;color:#94a3b8'>" + v_prev + "</div>"
            card_html += f"<div style='margin-top:8px;font-size:.72rem;color:{'#34d399' if err<1e-8 else '#fbbf24'}'>✓ Verifikasi A·v = λ·v: error = {err:.2e}</div>"
            card_html += "</div>"
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
    Edit tiap sel manual, lalu klik <b>Calculate</b>. Setiap langkah operasi baris akan ditampilkan.
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
.sol-grid { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px; max-height:400px; overflow-y:auto; padding:4px; }
.sol-card {
    background:#0f172a; border:1px solid rgba(255,255,255,0.07);
    border-radius:8px; padding:6px 10px; min-width:70px; text-align:center;
}
.sol-var { font-size:.62rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:2px; }
.sol-val { font-family:'JetBrains Mono',monospace; font-size:.82rem; font-weight:700; color:#34d399; }
.verif-box {
    background:linear-gradient(135deg,#0f172a,#1e1b4b);
    border:1px solid rgba(139,92,246,.3); border-radius:12px;
    padding:16px 20px; text-align:center; margin-top:12px;
}
.verif-status { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; }
.status-ok  { color:#34d399; }
.status-err { color:#f87171; }
.error-msg  { color:#f87171; background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.3); border-radius:8px; padding:10px 14px; font-size:.85rem; margin-top:10px; }

/* Log styles */
.log-section { margin-top:18px; }
.log-header  { color:#f87171; font-size:1rem; font-weight:700; margin-bottom:12px; border-left:3px solid #f87171; padding-left:10px; }
.log-block   { margin-bottom:18px; }
.log-title   { color:#fbbf24; font-size:.78rem; font-weight:600; margin-bottom:8px; }
.log-table-wrap { overflow-x:auto; border-radius:8px; }
.log-table   { border-collapse:collapse; font-family:"JetBrains Mono",monospace; font-size:.78rem; min-width:100%; }
.log-table th { background:#1e293b; color:#94a3b8; padding:6px 10px; text-align:center; border:1px solid rgba(255,255,255,0.07); font-weight:600; }
.log-table th.b-head { color:#a78bfa; }
.log-table td { padding:5px 10px; text-align:center; border:1px solid rgba(255,255,255,0.05); color:#e2e8f0; }
.log-table td.row-lbl { color:#475569; font-size:.7rem; background:#0f172a; }
.log-table td.b-cell  { color:#c4b5fd; background:rgba(139,92,246,0.08); }
.log-table tr.hl-row td { background:rgba(96,165,250,0.1); color:#93c5fd; }
.log-table tr.pivot-row td { background:rgba(248,113,113,0.1); color:#fca5a5; }

/* Step timeline */
.step-timeline { border-left:2px solid rgba(96,165,250,0.3); margin:12px 0 12px 12px; padding-left:20px; }
.step-item-log {
    background:#0f172a; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:12px 16px; margin-bottom:10px; position:relative;
}
.step-item-log::before {
    content:''; position:absolute; left:-27px; top:16px;
    width:10px; height:10px; border-radius:50%;
    background:#60a5fa; border:2px solid #0f172a;
}
.step-num-log { font-size:.62rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#60a5fa; margin-bottom:4px; }
.step-op-log  { font-family:'JetBrains Mono',monospace; font-size:.83rem; color:#fbbf24; font-weight:600; margin-bottom:6px; }
.step-desc-log { font-size:.78rem; color:#94a3b8; margin-bottom:8px; }
.step-formula { font-family:'JetBrains Mono',monospace; font-size:.75rem; background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.2); border-radius:6px; padding:6px 10px; color:#c4b5fd; margin:6px 0; }

/* Back sub */
.backsub-section { margin-top:14px; }
.backsub-title { color:#34d399; font-size:.72rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px; }

.chk-log { display:flex; align-items:center; gap:8px; margin-bottom:14px; cursor:pointer; font-size:.85rem; color:#94a3b8; user-select:none; }
.chk-log input { width:16px; height:16px; accent-color:#f87171; cursor:pointer; }
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

  <label class="chk-log">
    <input type="checkbox" id="chk-log" checked>
    Tampilkan langkah-langkah pengerjaan (Solution Log)
  </label>
  <div style="text-align:left;margin-bottom:12px">
    <button class="gauss-btn btn-calc" onclick="calculate()">▶ Calculate</button>
  </div>

  <div class="log-section" id="solution-log" style="display:none">
    <div class="log-header">📋 Langkah-Langkah Eliminasi Gauss</div>
    <div id="log-content"></div>
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
    document.getElementById('solution-log').style.display = 'none';
}

function buildGrid(values) {
    const grid = document.getElementById('matrix-grid');
    grid.innerHTML = '';
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

function matToHTML(mat, n, pivotRow, pivotCol, changedRow) {
    let html = '<table class="log-table"><thead><tr><th></th>';
    for (let j = 0; j < n; j++) html += '<th>x'+(j+1)+'</th>';
    html += '<th class="b-head">b</th></tr></thead><tbody>';
    for (let i = 0; i < n; i++) {
        const isPivot = (i === pivotRow);
        const isChanged = (i === changedRow);
        let rowClass = isPivot ? ' class="pivot-row"' : (isChanged ? ' class="hl-row"' : '');
        html += '<tr'+rowClass+'><td class="row-lbl">P'+(i+1)+'</td>';
        for (let j = 0; j <= n; j++) {
            const isB = j === n;
            const isPivotCell = (i === pivotRow && j === pivotCol);
            let tdStyle = isB ? ' class="b-cell"' : '';
            let val = mat[i][j].toFixed(3);
            if (isPivotCell) val = '<b style="color:#f87171">'+val+'</b>';
            html += '<td'+tdStyle+'>' + val + '</td>';
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    return html;
}

function calculate() {
    let M = [];
    for (let i = 0; i < N; i++) {
        let row = [];
        for (let j = 0; j < N; j++) row.push(parseFloat(document.getElementById('c_'+i+'_'+j).value) || 0);
        row.push(parseFloat(document.getElementById('b_'+i).value) || 0);
        M.push(row);
    }
    const A_orig = M.map(r => r.slice(0, N));
    const b_orig = M.map(r => r[N]);

    const showLog = document.getElementById('chk-log').checked;
    let logHTML = '';

    if (showLog) {
        logHTML += '<div class="step-timeline">';

        // Langkah 0: Matriks awal
        logHTML += '<div class="step-item-log">';
        logHTML += '<div class="step-num-log">Langkah Awal</div>';
        logHTML += '<div class="step-op-log">Matriks Augmentasi [A|b]</div>';
        logHTML += '<div class="step-desc-log">Tulis sistem persamaan sebagai matriks augmentasi. Setiap baris = satu persamaan.</div>';
        logHTML += '<div class="log-table-wrap">' + matToHTML(M, N, -1, -1, -1) + '</div>';
        logHTML += '</div>';
    }

    let stepNum = 1;

    // Eliminasi Gauss dengan partial pivoting
    for (let col = 0; col < N; col++) {
        // Cari pivot terbesar
        let maxRow = col;
        for (let r = col+1; r < N; r++) if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;

        if (maxRow !== col) {
            [M[col], M[maxRow]] = [M[maxRow], M[col]];
            if (showLog && N <= 12) {
                logHTML += '<div class="step-item-log">';
                logHTML += '<div class="step-num-log">Langkah ' + stepNum + '</div>';
                logHTML += '<div class="step-op-log">Tukar Baris: P'+(col+1)+' ↔ P'+(maxRow+1)+'</div>';
                logHTML += '<div class="step-desc-log">Pilih pivot terbesar di kolom '+(col+1)+' (partial pivoting) untuk stabilitas numerik. Nilai pivot terbesar = '+M[col][col].toFixed(3)+'</div>';
                logHTML += '<div class="step-formula">Tukar P'+(col+1)+' dan P'+(maxRow+1)+'</div>';
                logHTML += '<div class="log-table-wrap">' + matToHTML(M, N, col, col, -1) + '</div>';
                logHTML += '</div>';
                stepNum++;
            }
        }

        if (Math.abs(M[col][col]) < 1e-12) {
            showError('Matriks singular — tidak ada solusi unik. Coba ubah nilai.');
            return;
        }

        // Eliminasi ke bawah
        for (let row = col+1; row < N; row++) {
            const f = M[row][col] / M[col][col];
            if (Math.abs(f) < 1e-15) continue;
            for (let k = col; k <= N; k++) M[row][k] -= f * M[col][k];

            if (showLog && N <= 12) {
                logHTML += '<div class="step-item-log">';
                logHTML += '<div class="step-num-log">Langkah ' + stepNum + '</div>';
                logHTML += '<div class="step-op-log">Eliminasi: P'+(row+1)+' = P'+(row+1)+' − ('+f.toFixed(4)+') × P'+(col+1)+'</div>';
                logHTML += '<div class="step-desc-log">Faktor pengali m = '+f.toFixed(4)+' = M['+(row+1)+']['+(col+1)+'] / M['+(col+1)+']['+(col+1)+']. Kurangkan kelipatan baris pivot dari baris ini agar elemen kolom '+(col+1)+' menjadi 0.</div>';
                logHTML += '<div class="step-formula">m = '+M[row][col+0].toFixed(0)+' / '+M[col][col].toFixed(3)+' sebelum digunakan = '+f.toFixed(4)+'<br>P'+(row+1)+' ← P'+(row+1)+' − '+f.toFixed(4)+' × P'+(col+1)+'</div>';
                logHTML += '<div class="log-table-wrap">' + matToHTML(M, N, col, col, row) + '</div>';
                logHTML += '</div>';
                stepNum++;
            }
        }
    }

    if (showLog) {
        // Matriks segitiga atas
        logHTML += '<div class="step-item-log">';
        logHTML += '<div class="step-num-log">Hasil Eliminasi Maju</div>';
        logHTML += '<div class="step-op-log">Matriks Segitiga Atas (Upper Triangular)</div>';
        logHTML += '<div class="step-desc-log">Semua elemen di bawah diagonal utama sudah menjadi 0. Siap untuk substitusi mundur.</div>';
        logHTML += '<div class="log-table-wrap">' + matToHTML(M, N, -1, -1, -1) + '</div>';
        logHTML += '</div>';

        if (N > 12) {
            logHTML += '<div style="color:#fbbf24;font-size:.8rem;padding:10px;background:rgba(251,191,36,.08);border-radius:8px;margin-top:8px">ℹ️ Detail langkah disembunyikan untuk sistem '+N+'×'+N+' (terlalu banyak langkah). Hanya tampilan matriks awal & akhir.</div>';
        }
    }

    // Substitusi mundur dengan langkah-langkah
    let x = new Array(N).fill(0);
    let backSubHTML = '';
    if (showLog) {
        backSubHTML = '<div class="backsub-section"><div class="backsub-title">🔁 Substitusi Mundur (Back Substitution)</div><div class="step-timeline">';
    }

    for (let i = N-1; i >= 0; i--) {
        x[i] = M[i][N];
        let formula = `x${i+1} = (${M[i][N].toFixed(3)}`;
        for (let j = i+1; j < N; j++) {
            x[i] -= M[i][j] * x[j];
            formula += ` − (${M[i][j].toFixed(3)}) × x${j+1}`;
        }
        x[i] /= M[i][i];
        formula += `) / ${M[i][i].toFixed(3)} = ${x[i].toFixed(4)}`;

        if (showLog) {
            backSubHTML += '<div class="step-item-log">';
            backSubHTML += '<div class="step-num-log">Substitusi x'+(i+1)+'</div>';
            backSubHTML += '<div class="step-op-log">Hitung x'+(i+1)+' dari baris P'+(i+1)+'</div>';
            backSubHTML += `<div class="step-formula">${formula}</div>`;
            backSubHTML += '</div>';
        }
    }

    if (showLog) {
        backSubHTML += '</div></div>';
        logHTML += backSubHTML;
        logHTML += '</div>'; // tutup step-timeline
    }

    // Render log
    const logEl = document.getElementById('solution-log');
    if (showLog) {
        logEl.style.display = 'block';
        document.getElementById('log-content').innerHTML = logHTML;
    } else {
        logEl.style.display = 'none';
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
    for (let i = 0; i < N; i++) {
        const card = document.createElement('div');
        card.className = 'sol-card';
        card.innerHTML = '<div class="sol-var">x' + (i+1) + '</div><div class="sol-val">' + x[i].toFixed(4) + '</div>';
        solGrid.appendChild(card);
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

buildGrid();
</script>
"""
    st.components.v1.html(gauss_html, height=900, scrolling=True)


# ════════════════════════════════════════════════════════════
# DETERMINAN & INVERS
# ════════════════════════════════════════════════════════════
elif menu == "🔲 Determinan & Invers":
    st.markdown('<p class="section-label">Input Matriks</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Masukkan matriks persegi n×n. Pisahkan angka dengan spasi, baris baru untuk baris berikutnya.
    Langkah-langkah perhitungan determinan dan invers akan ditampilkan secara detail.
    </div>""", unsafe_allow_html=True)

    default_mat = "4 7 2 1\n1 3 1 2\n2 5 3 1\n3 6 4 8"
    raw = st.text_area("Matriks A", value=default_mat, height=130)

    show_steps = st.checkbox("Tampilkan langkah-langkah pengerjaan", value=True)

    if st.button("▶ Hitung Determinan & Invers", type="primary"):
        try:
            rows_data = [[float(x) for x in r.split()] for r in raw.strip().split('\n') if r.strip()]
            A = np.array(rows_data)
            n = A.shape[0]
            if A.shape[0] != A.shape[1]:
                st.error("Matriks harus persegi (n×n)!")
            else:
                det = np.linalg.det(A)

                st.markdown('<p class="section-label">Matriks Input A</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(A,4),
                    columns=[f"K{i+1}" for i in range(n)],
                    index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                # ── LANGKAH DETERMINAN ──
                if show_steps:
                    st.markdown('<p class="section-label">📋 Langkah Perhitungan Determinan</p>', unsafe_allow_html=True)

                    with st.expander("🔍 Lihat Langkah-Langkah Determinan", expanded=True):
                        steps_html = '<div class="step-timeline">'

                        # Langkah 1: Penjelasan metode
                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah 1</div>
                            <div class="step-op">Pilih Metode Perhitungan</div>
                            <div class="step-explain">
                                Untuk matriks {n}×{n}, digunakan metode <b>Eliminasi Gauss (Row Reduction)</b>:<br>
                                det(A) = produk elemen diagonal matriks segitiga atas × faktor pertukaran baris.
                            </div>
                            <div class="formula-box">det(A) = (−1)^k × U[1,1] × U[2,2] × ... × U[n,n]<br>
k = jumlah pertukaran baris, U = matriks segitiga atas hasil eliminasi</div>
                        </div>
                        """

                        # Eliminasi dengan tracking pivot
                        U = A.copy().astype(float)
                        sign = 1
                        pivot_log = []

                        for col in range(n):
                            maxRow = col
                            for r in range(col+1, n):
                                if abs(U[r, col]) > abs(U[maxRow, col]):
                                    maxRow = r
                            if maxRow != col:
                                U[[col, maxRow]] = U[[maxRow, col]]
                                sign *= -1
                                pivot_log.append(('swap', col, maxRow, U.copy()))

                            if abs(U[col, col]) < 1e-12:
                                break

                            for row in range(col+1, n):
                                if abs(U[row, col]) < 1e-15:
                                    continue
                                f = U[row, col] / U[col, col]
                                U[row] -= f * U[col]
                                pivot_log.append(('elim', col, row, f, U.copy()))

                        # Langkah 2: Matriks awal
                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah 2</div>
                            <div class="step-op">Matriks Awal A</div>
                            <div class="step-explain">Titik mulai eliminasi. Kita akan ubah matriks ini ke bentuk segitiga atas.</div>
                            <div class="mini-matrix">{mat_to_str(A)}</div>
                        </div>
                        """

                        # Langkah 3+: Setiap operasi eliminasi (max 10 langkah agar tidak terlalu panjang)
                        step_shown = 3
                        ops_shown = 0
                        for op in pivot_log:
                            if ops_shown >= 12:
                                steps_html += f'<div style="color:#fbbf24;font-size:.8rem;padding:8px;background:rgba(251,191,36,.08);border-radius:6px">... ({len(pivot_log) - ops_shown} langkah eliminasi lainnya disembunyikan)</div>'
                                break
                            if op[0] == 'swap':
                                _, c, mr, mat_after = op
                                steps_html += f"""
                                <div class="step-item">
                                    <div class="step-num">Langkah {step_shown}</div>
                                    <div class="step-op">Tukar Baris B{c+1} ↔ B{mr+1} (Partial Pivoting)</div>
                                    <div class="step-explain">Cari pivot terbesar di kolom {c+1}. Pertukaran baris mengubah tanda determinan (sign × −1).</div>
                                    <div class="formula-box">det berubah tanda: sign = {sign}</div>
                                    <div class="mini-matrix">{mat_to_str(mat_after, highlight_row=c)}</div>
                                </div>
                                """
                            elif op[0] == 'elim':
                                _, c, row, f, mat_after = op
                                steps_html += f"""
                                <div class="step-item">
                                    <div class="step-num">Langkah {step_shown}</div>
                                    <div class="step-op">B{row+1} = B{row+1} − ({f:.4f}) × B{c+1}</div>
                                    <div class="step-explain">
                                        Faktor m = U[{row+1},{c+1}] / U[{c+1},{c+1}] = {f:.4f}<br>
                                        Operasi baris elementer ini tidak mengubah nilai determinan.
                                    </div>
                                    <div class="mini-matrix">{mat_to_str(mat_after, highlight_row=row)}</div>
                                </div>
                                """
                            step_shown += 1
                            ops_shown += 1

                        # Diagonal pivot
                        diag = [U[i,i] for i in range(n)]
                        diag_str = " × ".join([f"{d:.4f}" for d in diag])
                        det_from_diag = 1.0
                        for d in diag:
                            det_from_diag *= d
                        det_from_diag *= sign

                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah {step_shown}</div>
                            <div class="step-op">Baca Diagonal Matriks Segitiga Atas U</div>
                            <div class="step-explain">
                                Setelah eliminasi selesai, nilai diagonal utama U adalah pivot-pivot hasil eliminasi.
                            </div>
                            <div class="formula-box">
                                Diagonal U = [{", ".join([f"{d:.4f}" for d in diag])}]<br>
                                Produk diagonal = {diag_str} = {det_from_diag:.6f}
                            </div>
                        </div>
                        """

                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah {step_shown+1}</div>
                            <div class="step-op">Hitung det(A) = sign × produk diagonal</div>
                            <div class="step-explain">
                                Faktor tanda karena pertukaran baris: sign = {sign} (jumlah swap = {'genap ✓' if sign==1 else 'ganjil'})<br>
                                det(A) = {sign} × {det_from_diag/sign:.6f} = {det:.6f}
                            </div>
                            <div class="formula-box">det(A) = {det:.6f}</div>
                        </div>
                        """

                        steps_html += '</div>'
                        st.markdown(steps_html, unsafe_allow_html=True)

                # Tampilkan determinan
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

                    # ── LANGKAH INVERS ──
                    if show_steps:
                        st.markdown('<p class="section-label">📋 Langkah Perhitungan Invers (Gauss-Jordan)</p>', unsafe_allow_html=True)

                        with st.expander("🔍 Lihat Langkah-Langkah Invers", expanded=True):
                            steps_html2 = '<div class="step-timeline">'

                            steps_html2 += f"""
                            <div class="step-item">
                                <div class="step-num">Langkah 1</div>
                                <div class="step-op">Bentuk Matriks Augmentasi [A | I]</div>
                                <div class="step-explain">
                                    Metode Gauss-Jordan: gabungkan matriks A dengan matriks identitas I.
                                    Transformasikan sisi kiri menjadi I, maka sisi kanan menjadi A⁻¹.
                                </div>
                                <div class="formula-box">[A | I] → [I | A⁻¹]</div>
                            </div>
                            """

                            # Gauss-Jordan step by step
                            AI = np.hstack([A.copy().astype(float), np.eye(n)])
                            gj_step = 2

                            for col in range(n):
                                # Pivoting
                                maxRow = col
                                for r in range(col+1, n):
                                    if abs(AI[r, col]) > abs(AI[maxRow, col]):
                                        maxRow = r
                                if maxRow != col:
                                    AI[[col, maxRow]] = AI[[maxRow, col]]
                                    if gj_step <= 15:
                                        steps_html2 += f"""
                                        <div class="step-item">
                                            <div class="step-num">Langkah {gj_step}</div>
                                            <div class="step-op">Tukar Baris B{col+1} ↔ B{maxRow+1}</div>
                                            <div class="step-explain">Pastikan pivot terbesar di posisi ({col+1},{col+1}).</div>
                                        </div>
                                        """
                                        gj_step += 1

                                # Normalisasi pivot
                                pivot = AI[col, col]
                                AI[col] /= pivot
                                if gj_step <= 15:
                                    steps_html2 += f"""
                                    <div class="step-item">
                                        <div class="step-num">Langkah {gj_step}</div>
                                        <div class="step-op">Normalisasi B{col+1}: B{col+1} ÷ {pivot:.4f}</div>
                                        <div class="step-explain">
                                            Bagi seluruh baris {col+1} dengan pivot = {pivot:.4f} agar diagonal menjadi 1.
                                        </div>
                                        <div class="formula-box">B{col+1} ← B{col+1} / {pivot:.4f}</div>
                                    </div>
                                    """
                                    gj_step += 1

                                # Eliminasi ke atas DAN bawah
                                for row in range(n):
                                    if row == col or abs(AI[row, col]) < 1e-15:
                                        continue
                                    f = AI[row, col]
                                    AI[row] -= f * AI[col]
                                    if gj_step <= 20:
                                        direction = "bawah" if row > col else "atas"
                                        steps_html2 += f"""
                                        <div class="step-item">
                                            <div class="step-num">Langkah {gj_step}</div>
                                            <div class="step-op">Eliminasi ke {direction}: B{row+1} = B{row+1} − ({f:.4f}) × B{col+1}</div>
                                            <div class="step-explain">Buat elemen kolom {col+1} di baris {row+1} menjadi 0.</div>
                                            <div class="formula-box">B{row+1} ← B{row+1} − {f:.4f} × B{col+1}</div>
                                        </div>
                                        """
                                        gj_step += 1

                            steps_html2 += f"""
                            <div class="step-item">
                                <div class="step-num">Langkah {gj_step}</div>
                                <div class="step-op">Hasil: Sisi kiri menjadi I → Sisi kanan adalah A⁻¹</div>
                                <div class="step-explain">
                                    Transformasi selesai. Matriks identitas di sisi kiri membuktikan proses berhasil.
                                    Sisi kanan adalah matriks invers A⁻¹.
                                </div>
                                <div class="formula-box">[A | I] → [I | A⁻¹] ✓</div>
                            </div>
                            """

                            steps_html2 += '</div>'
                            st.markdown(steps_html2, unsafe_allow_html=True)

                    st.markdown('<p class="section-label">Matriks Invers A⁻¹</p>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(np.round(inv,6),
                        columns=[f"K{i+1}" for i in range(n)],
                        index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                    # Verifikasi A × A⁻¹ = I
                    I_check = np.round(A @ inv, 4)
                    st.markdown('<p class="section-label">Verifikasi: A × A⁻¹ = I</p>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(I_check,
                        columns=[f"K{i+1}" for i in range(n)],
                        index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
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
    Setiap langkah pembentukan multiplier dan eliminasi akan ditampilkan.
    </div>""", unsafe_allow_html=True)

    default_lu = "2 1 1\n4 3 3\n8 7 9"
    raw = st.text_area("Matriks A", value=default_lu, height=110)
    show_steps_lu = st.checkbox("Tampilkan langkah-langkah pengerjaan LU", value=True)

    if st.button("▶ Dekomposisi LU", type="primary"):
        try:
            rows_data = [[float(x) for x in r.split()] for r in raw.strip().split('\n') if r.strip()]
            A = np.array(rows_data)
            n = A.shape[0]

            if A.shape[0] != A.shape[1]:
                st.error("Matriks harus persegi!")
            else:
                st.markdown('<p class="section-label">Matriks A (Input)</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(A,4),
                    columns=[f"K{i+1}" for i in range(n)],
                    index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                # ── LANGKAH-LANGKAH LU ──
                if show_steps_lu:
                    st.markdown('<p class="section-label">📋 Langkah-Langkah Dekomposisi LU (Doolittle)</p>', unsafe_allow_html=True)

                    with st.expander("🔍 Lihat Langkah-Langkah Pengerjaan LU", expanded=True):
                        steps_html = '<div class="step-timeline">'

                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah 1</div>
                            <div class="step-op">Tujuan: Faktorisasi A = L × U</div>
                            <div class="step-explain">
                                Metode Doolittle mencari L (lower triangular, diagonal = 1) dan U (upper triangular) sehingga A = L·U.<br>
                                Formula: <b>u_ij = a_ij − Σ(l_ik × u_kj)</b> untuk U<br>
                                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>l_ij = (a_ij − Σ(l_ik × u_kj)) / u_jj</b> untuk L
                            </div>
                            <div class="formula-box">
                                L = matriks segitiga bawah (diagonal = 1)<br>
                                U = matriks segitiga atas<br>
                                A = L × U
                            </div>
                        </div>
                        """

                        # Hitung LU manual dengan tracking
                        L_manual = np.eye(n)
                        U_manual = np.zeros((n, n))
                        A_work = A.copy().astype(float)
                        step_lu = 2

                        for k in range(n):
                            # Hitung baris U ke-k
                            for j in range(k, n):
                                sum_val = sum(L_manual[k, s] * U_manual[s, j] for s in range(k))
                                U_manual[k, j] = A_work[k, j] - sum_val

                            # Tulis langkah U
                            u_row_vals = [f"U[{k+1},{j+1}] = {A_work[k,j]:.3f} − {sum(L_manual[k,s]*U_manual[s,j] for s in range(k)):.3f} = {U_manual[k,j]:.4f}" for j in range(k, min(k+3, n))]
                            steps_html += f"""
                            <div class="step-item">
                                <div class="step-num">Langkah {step_lu}</div>
                                <div class="step-op">Hitung Baris U ke-{k+1} (U[{k+1},:])</div>
                                <div class="step-explain">Gunakan rumus: U[{k+1},j] = A[{k+1},j] − Σ(L[{k+1},s]×U[s,j]) untuk s = 1..{k}</div>
                                <div class="formula-box">{"<br>".join(u_row_vals)}{"<br>..." if n > k+3 else ""}</div>
                            </div>
                            """
                            step_lu += 1

                            # Hitung kolom L ke-k
                            for i in range(k+1, n):
                                sum_val = sum(L_manual[i, s] * U_manual[s, k] for s in range(k))
                                if abs(U_manual[k, k]) < 1e-12:
                                    st.error(f"Pivot nol di posisi ({k+1},{k+1}). Gunakan matriks lain atau gunakan scipy untuk dekomposisi dengan pivoting.")
                                    st.stop()
                                L_manual[i, k] = (A_work[i, k] - sum_val) / U_manual[k, k]

                            if k < n-1:
                                l_col_vals = [f"L[{i+1},{k+1}] = ({A_work[i,k]:.3f} − ...) / U[{k+1},{k+1}] = {L_manual[i,k]:.4f}" for i in range(k+1, min(k+4, n))]
                                steps_html += f"""
                                <div class="step-item">
                                    <div class="step-num">Langkah {step_lu}</div>
                                    <div class="step-op">Hitung Kolom L ke-{k+1} (L[:,{k+1}])</div>
                                    <div class="step-explain">Gunakan rumus: L[i,{k+1}] = (A[i,{k+1}] − Σ(L[i,s]×U[s,{k+1}])) / U[{k+1},{k+1}]</div>
                                    <div class="formula-box">{"<br>".join(l_col_vals)}{"<br>..." if n > k+4 else ""}</div>
                                </div>
                                """
                                step_lu += 1

                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah {step_lu}</div>
                            <div class="step-op">Matriks L dan U Selesai</div>
                            <div class="step-explain">
                                Semua elemen L dan U telah dihitung. Verifikasi: L × U harus menghasilkan A kembali.
                            </div>
                            <div class="formula-box">
                                L (diagonal = 1, segitiga bawah) ✓<br>
                                U (segitiga atas) ✓<br>
                                L × U = A ? → Cek verifikasi di bawah
                            </div>
                        </div>
                        """

                        steps_html += '</div>'
                        st.markdown(steps_html, unsafe_allow_html=True)

                # Coba scipy, fallback ke manual
                try:
                    import scipy.linalg
                    P, L, U = scipy.linalg.lu(A)
                    use_scipy = True
                except ImportError:
                    L, U = L_manual, U_manual
                    P = np.eye(n)
                    use_scipy = False

                st.latex(r"A = P \cdot L \cdot U" if use_scipy else r"A = L \cdot U")

                c1, c2, c3 = st.columns(3) if use_scipy else st.columns(2)
                if use_scipy:
                    with c1:
                        st.markdown("**Matriks P (Permutasi)**")
                        st.dataframe(pd.DataFrame(np.round(P,4),
                            columns=[f"K{i+1}" for i in range(n)],
                            index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                    with c2:
                        st.markdown("**Matriks L (Lower)**")
                        st.dataframe(pd.DataFrame(np.round(L,4),
                            columns=[f"K{i+1}" for i in range(n)],
                            index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                    with c3:
                        st.markdown("**Matriks U (Upper)**")
                        st.dataframe(pd.DataFrame(np.round(U,4),
                            columns=[f"K{i+1}" for i in range(n)],
                            index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                else:
                    with c1:
                        st.markdown("**Matriks L (Lower)**")
                        st.dataframe(pd.DataFrame(np.round(L,4),
                            columns=[f"K{i+1}" for i in range(n)],
                            index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                    with c2:
                        st.markdown("**Matriks U (Upper)**")
                        st.dataframe(pd.DataFrame(np.round(U,4),
                            columns=[f"K{i+1}" for i in range(n)],
                            index=[f"B{i+1}" for i in range(n)]), use_container_width=True)

                # Verifikasi
                PLU = P @ L @ U if use_scipy else L @ U
                st.markdown('<p class="section-label">Verifikasi: P·L·U = A</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(np.round(PLU,4),
                    columns=[f"K{i+1}" for i in range(n)],
                    index=[f"B{i+1}" for i in range(n)]), use_container_width=True)
                if np.allclose(PLU, A):
                    st.success("✓ Terverifikasi — P·L·U = A")

        except Exception as e:
            st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════
# REGRESI LINEAR
# ════════════════════════════════════════════════════════════
elif menu == "📈 Regresi Linear":
    st.markdown('<p class="section-label">Regresi Linear (Least Squares)</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Masukkan data titik (x, y) — satu titik per baris, pisahkan x dan y dengan spasi.
    Metode least squares akan mencari garis terbaik: <b>y = mx + c</b>.
    Langkah-langkah pembentukan normal equations akan ditampilkan.
    </div>""", unsafe_allow_html=True)

    default_data = "1 2.1\n2 3.9\n3 6.2\n4 7.8\n5 10.1\n6 11.9\n7 14.2\n8 16.1"
    raw = st.text_area("Data (x y)", value=default_data, height=180)
    show_steps_reg = st.checkbox("Tampilkan langkah-langkah pengerjaan regresi", value=True)

    if st.button("▶ Hitung Regresi Linear", type="primary"):
        try:
            pts = [[float(v) for v in r.split()] for r in raw.strip().split('\n') if r.strip()]
            data = np.array(pts)
            x, y = data[:,0], data[:,1]
            n_pts = len(x)

            A = np.column_stack([x, np.ones(n_pts)])
            ATA = A.T @ A
            ATb = A.T @ y
            coeffs = np.linalg.solve(ATA, ATb)
            m, c = coeffs

            y_pred = m * x + c
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot

            # ── LANGKAH REGRESI ──
            if show_steps_reg:
                st.markdown('<p class="section-label">📋 Langkah-Langkah Regresi Least Squares</p>', unsafe_allow_html=True)

                with st.expander("🔍 Lihat Langkah-Langkah Pengerjaan", expanded=True):
                    steps_html = '<div class="step-timeline">'

                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah 1</div>
                        <div class="step-op">Bentuk Matriks Desain A</div>
                        <div class="step-explain">
                            Sistem over-determined (lebih banyak titik dari parameter). Bentuk matriks A berukuran {n_pts}×2
                            dengan kolom pertama = x dan kolom kedua = 1 (konstanta).
                        </div>
                        <div class="formula-box">
                            A = [x₁  1]   = [{x[0]:.1f}  1]<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x₂  1]   &nbsp;&nbsp;[{x[1]:.1f}  1]<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x₃  1]   &nbsp;&nbsp;[{x[2]:.1f}  1]<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[...  ]   &nbsp;&nbsp;[...  ]
                        </div>
                    </div>
                    """

                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah 2</div>
                        <div class="step-op">Bentuk Normal Equations: AᵀA · θ = Aᵀb</div>
                        <div class="step-explain">
                            Untuk meminimalkan ||Ax − b||², turunkan dan samakan ke nol.
                            Hasilnya adalah sistem normal equations.
                        </div>
                        <div class="formula-box">
                            (AᵀA) · [m, c]ᵀ = Aᵀy<br>
                            AᵀA memiliki dimensi 2×2
                        </div>
                    </div>
                    """

                    # Hitung elemen ATA secara eksplisit
                    sum_x  = np.sum(x)
                    sum_x2 = np.sum(x**2)
                    sum_y  = np.sum(y)
                    sum_xy = np.sum(x*y)

                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah 3</div>
                        <div class="step-op">Hitung Komponen AᵀA dan Aᵀy</div>
                        <div class="step-explain">
                            Hitung jumlah-jumlah yang diperlukan dari {n_pts} titik data:
                        </div>
                        <div class="formula-box">
                            n = {n_pts}<br>
                            Σx  = {sum_x:.4f}<br>
                            Σx² = {sum_x2:.4f}<br>
                            Σy  = {sum_y:.4f}<br>
                            Σxy = {sum_xy:.4f}
                        </div>
                    </div>
                    """

                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah 4</div>
                        <div class="step-op">Susun Matriks Normal Equations</div>
                        <div class="step-explain">
                            Masukkan nilai-nilai tersebut ke dalam sistem persamaan:
                        </div>
                        <div class="formula-box">
                            [Σx²   Σx ] [m]   [Σxy]<br>
                            [Σx    n  ] [c] = [Σy ]<br><br>
                            [{sum_x2:.4f}  {sum_x:.4f}] [m]   [{sum_xy:.4f}]<br>
                            [{sum_x:.4f}   {n_pts:.0f}    ] [c] = [{sum_y:.4f}]
                        </div>
                    </div>
                    """

                    # Selesaikan secara manual
                    det_ata = sum_x2 * n_pts - sum_x * sum_x
                    m_manual = (n_pts * sum_xy - sum_x * sum_y) / det_ata
                    c_manual = (sum_x2 * sum_y - sum_x * sum_xy) / det_ata

                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah 5</div>
                        <div class="step-op">Selesaikan Sistem 2×2 dengan Cramer's Rule</div>
                        <div class="step-explain">
                            Untuk sistem 2×2, gunakan rumus langsung (invers matriks 2×2):
                        </div>
                        <div class="formula-box">
                            det(AᵀA) = {sum_x2:.4f} × {n_pts} − {sum_x:.4f} × {sum_x:.4f} = {det_ata:.4f}<br><br>
                            m = (n·Σxy − Σx·Σy) / det = ({n_pts}×{sum_xy:.4f} − {sum_x:.4f}×{sum_y:.4f}) / {det_ata:.4f}<br>
                            m = {m_manual:.4f}<br><br>
                            c = (Σx²·Σy − Σx·Σxy) / det = ({sum_x2:.4f}×{sum_y:.4f} − {sum_x:.4f}×{sum_xy:.4f}) / {det_ata:.4f}<br>
                            c = {c_manual:.4f}
                        </div>
                    </div>
                    """

                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah 6</div>
                        <div class="step-op">Hitung R² (Koefisien Determinasi)</div>
                        <div class="step-explain">
                            R² mengukur seberapa baik garis cocok dengan data (0 = buruk, 1 = sempurna).
                        </div>
                        <div class="formula-box">
                            SS_res = Σ(yᵢ − ŷᵢ)² = {ss_res:.4f}<br>
                            SS_tot = Σ(yᵢ − ȳ)² = {ss_tot:.4f}  (ȳ = {np.mean(y):.4f})<br>
                            R² = 1 − SS_res/SS_tot = 1 − {ss_res:.4f}/{ss_tot:.4f} = {r2:.4f}
                        </div>
                    </div>
                    """

                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah 7</div>
                        <div class="step-op">Persamaan Regresi Final</div>
                        <div class="step-explain">Garis regresi terbaik yang meminimalkan jumlah kuadrat residual:</div>
                        <div class="formula-box">ŷ = {m:.4f}x + ({c:.4f})   dengan R² = {r2:.4f}</div>
                    </div>
                    """

                    steps_html += '</div>'
                    st.markdown(steps_html, unsafe_allow_html=True)

            # Tampilkan hasil
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
