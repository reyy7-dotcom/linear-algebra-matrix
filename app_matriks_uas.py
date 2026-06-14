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

/* Tabel matriks (untuk Eliminasi Gauss & lainnya) */
.matrix-table-wrap {
    overflow-x: auto;
    margin: 10px 0;
}
.matrix-table {
    border-collapse: collapse;
    margin: 0 auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.matrix-table td {
    border: 1px solid rgba(255,255,255,0.08);
    padding: 6px 14px;
    text-align: center;
    color: #e2e8f0;
    background: #1e293b;
    min-width: 46px;
}
.matrix-table td.divider {
    border-left: 2px dashed #a78bfa;
}
.matrix-table td.pivot {
    color: #f87171;
    font-weight: 700;
    background: rgba(248,113,113,0.08);
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


def mat_table(mat, decimals=4, divider_index=None, highlight_row=None, highlight_col=None):
    """
    Render matriks sebagai tabel HTML penuh (mirip renderTable di versi JS),
    dengan opsi garis pemisah kolom (untuk matriks augmentasi [A|b])
    dan highlight baris/kolom pivot.
    """
    html = '<div class="matrix-table-wrap"><table class="matrix-table"><tbody>'
    for i, row in enumerate(mat):
        html += "<tr>"
        for j, val in enumerate(row):
            classes = []
            if divider_index is not None and j == divider_index:
                classes.append("divider")
            if (highlight_row is not None and i == highlight_row) or \
               (highlight_col is not None and j == highlight_col):
                classes.append("pivot")
            cls_attr = f' class="{" ".join(classes)}"' if classes else ""
            html += f'<td{cls_attr}>{val:.{decimals}f}</td>'
        html += "</tr>"
    html += "</tbody></table></div>"
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
            M = np.random.randint(rmin, rmax + 1, size=(ukuran, ukuran))
            vals, vecs = np.linalg.eig(M)
            trace_m = int(np.trace(M))
            trace_e = vals.real.sum()
            buf = io.BytesIO()
            np.savetxt(buf, M, fmt="%d", delimiter="\t")

        st.markdown('<p class="section-label">Info Matriks</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dimensi", f"{ukuran}×{ukuran}")
        c2.metric("Total Elemen", f"{ukuran*ukuran:,}")
        c3.metric("Rentang", f"{rmin}–{rmax}")
        c4.metric("Seed", str(seed_val))

        st.markdown('<p class="section-label">Preview Matriks</p>', unsafe_allow_html=True)
        df = pd.DataFrame(M, columns=[f"K{i+1}" for i in range(ukuran)], index=[f"B{i+1}" for i in range(ukuran)])
        st.dataframe(df, height=min(300, ukuran * 40 + 50), use_container_width=True)
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
        c1, c2, c3 = st.columns(3)
        for idx, col in enumerate([c1, c2, c3]):
            λ = vals[idx]
            col.metric(f"λ{idx+1}", f"{λ.real:.4f}", f"Im: {λ.imag:+.4f}" if abs(λ.imag) > 1e-10 else "Murni Real")

        for idx in range(min(3, ukuran)):
            λ = vals[idx]
            is_c = abs(λ.imag) >= 1e-10
            badge_cls = "badge-complex" if is_c else "badge-real"
            badge_txt = "Kompleks" if is_c else "Real"
            badge_html = "<span class='eigen-badge " + badge_cls + "'> " + badge_txt + "</span>"
            v_prev = " &nbsp;|&nbsp; ".join([format_kompleks(vecs[j, idx]) for j in range(min(5, ukuran))])
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
                <div style="font-size:2rem;color:#475569;align-self:center">≈</div>
                <div><div class="result-value">{trace_e:.4f}</div><div style="font-size:.8rem;color:rgba(255,255,255,.4)">Σλᵢ</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# ELIMINASI GAUSS
# ════════════════════════════════════════════════════════════
elif menu == "➗ Eliminasi Gauss":
    st.markdown('<p class="section-label">Kontrol</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Jumlah Variabel (n×n)", 2, 8, 3)
    with col2:
        show_steps = st.checkbox("Tampilkan Langkah-Langkah Pengerjaan", value=True)

    st.markdown('<p class="section-label">Input Matriks Augmentasi [A | b]</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Masukkan koefisien matriks A (kiri) dan konstanta b (kanan, dipisahkan garis putus-putus ungu).
    Sistem akan diselesaikan dengan metode <b>Eliminasi Gauss-Jordan</b> hingga
    didapat bentuk <b>Reduced Row Echelon Form (RREF)</b>.
    </div>""", unsafe_allow_html=True)

    A_rows = []
    for i in range(n):
        cols_in = st.columns(n + 1)
        row = []
        for j in range(n):
            val = cols_in[j].number_input(
                f"a{i+1}{j+1}", value=1.0 if i == j else 0.0,
                key=f"gauss_a_{i}_{j}", label_visibility="collapsed"
            )
            row.append(val)
        b_val = cols_in[n].number_input(
            f"b{i+1}", value=0.0, key=f"gauss_b_{i}", label_visibility="collapsed"
        )
        row.append(b_val)
        A_rows.append(row)

    if st.button("▶ Selesaikan Sistem", type="primary"):
        mat = np.array(A_rows, dtype=float)
        r, c = mat.shape  # r = n, c = n+1
        A_orig = mat[:, :n].copy()
        b_orig = mat[:, n].copy()

        st.markdown('<p class="section-label">📋 Langkah-Langkah Eliminasi Gauss-Jordan</p>', unsafe_allow_html=True)

        with st.expander("🔍 Lihat Langkah-Langkah Pengerjaan", expanded=True):
            steps_html = '<div class="step-timeline">'

            # Langkah 1: Matriks Augmentasi Awal
            steps_html += f"""
            <div class="step-item">
                <div class="step-num">Langkah 1</div>
                <div class="step-op">Bentuk Matriks Augmentasi [A | b]</div>
                <div class="step-explain">
                    Sistem persamaan linear Ax = b ditulis ulang sebagai matriks augmentasi,
                    di mana kolom terakhir (dipisahkan garis ungu) berisi konstanta b
                    dari sisi kanan setiap persamaan.
                </div>
                {mat_table(mat, divider_index=n)}
            </div>
            """

            step_num = 2
            lead = 0

            for r_target in range(r):
                if c <= lead:
                    break
                i = r_target

                # Cari baris dengan elemen tidak nol pada kolom 'lead'
                while mat[i][lead] == 0:
                    i += 1
                    if i == r:
                        i = r_target
                        lead += 1
                        if lead == c:
                            break
                if lead == c:
                    break

                # Tukar baris jika perlu
                if i != r_target:
                    mat[[r_target, i]] = mat[[i, r_target]]
                    if show_steps:
                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah {step_num}</div>
                            <div class="step-op">Tukar Baris B{r_target+1} ↔ B{i+1}</div>
                            <div class="step-explain">
                                Elemen pada posisi (baris {r_target+1}, kolom {lead+1}) bernilai 0,
                                sehingga baris {r_target+1} ditukar dengan baris {i+1} yang memiliki
                                elemen tidak nol pada kolom tersebut, agar bisa dijadikan pivot.
                            </div>
                            {mat_table(mat, divider_index=n, highlight_row=r_target)}
                        </div>
                        """
                        step_num += 1

                # Bagi baris dengan pivot agar menjadi 1
                pivot = mat[r_target][lead]
                if pivot != 1 and pivot != 0:
                    mat[r_target] = mat[r_target] / pivot
                    if show_steps:
                        steps_html += f"""
                        <div class="step-item">
                            <div class="step-num">Langkah {step_num}</div>
                            <div class="step-op">Bagi Baris B{r_target+1} dengan {pivot:.4f} (Leading 1)</div>
                            <div class="step-explain">
                                Operasi baris: <b>B{r_target+1} → B{r_target+1} ÷ {pivot:.4f}</b><br>
                                Tujuannya membuat elemen pivot pada posisi (baris {r_target+1}, kolom {lead+1})
                                bernilai 1, sebagai syarat bentuk eselon baris tereduksi (RREF).
                            </div>
                            {mat_table(mat, divider_index=n, highlight_row=r_target)}
                        </div>
                        """
                        step_num += 1

                # Eliminasi kolom 'lead' pada baris lain
                eliminated_rows = []
                for k in range(r):
                    if k != r_target:
                        factor = mat[k][lead]
                        if factor != 0:
                            mat[k] = mat[k] - factor * mat[r_target]
                            eliminated_rows.append((k, factor))

                if show_steps and eliminated_rows:
                    ops_list = "<br>".join([
                        f"B{k+1} → B{k+1} − ({fac:.4f}) × B{r_target+1}"
                        for k, fac in eliminated_rows
                    ])
                    steps_html += f"""
                    <div class="step-item">
                        <div class="step-num">Langkah {step_num}</div>
                        <div class="step-op">Eliminasi Kolom {lead+1} pada Baris Lain</div>
                        <div class="step-explain">
                            Agar kolom {lead+1} hanya memiliki satu elemen tidak nol (pada baris pivot
                            B{r_target+1}), elemen pada baris lain di kolom tersebut dieliminasi
                            menjadi 0 dengan operasi berikut:
                        </div>
                        <div class="formula-box">{ops_list}</div>
                        {mat_table(mat, divider_index=n, highlight_col=lead)}
                    </div>
                    """
                    step_num += 1

                lead += 1

            # Langkah hasil akhir
            if show_steps:
                steps_html += f"""
                <div class="step-item">
                    <div class="step-num">Langkah {step_num}</div>
                    <div class="step-op">Hasil Akhir: Reduced Row Echelon Form (RREF)</div>
                    <div class="step-explain">
                        Setelah seluruh proses eliminasi selesai, matriks berbentuk RREF
                        dengan diagonal utama bernilai 1 dan elemen lain pada kolom A bernilai 0.
                        Kolom terakhir (b) memuat nilai solusi xᵢ untuk setiap variabel.
                    </div>
                    {mat_table(mat, divider_index=n)}
                </div>
                """

            steps_html += '</div>'
            st.markdown(steps_html, unsafe_allow_html=True)

            if not show_steps:
                st.markdown('<p class="section-label">Hasil Akhir (RREF)</p>', unsafe_allow_html=True)
                st.markdown(mat_table(mat, divider_index=n), unsafe_allow_html=True)

        # ── Solusi ──
        st.markdown('<p class="section-label">Solusi Sistem (X)</p>', unsafe_allow_html=True)
        sol_html = "<div class='result-box'><div style='display:flex;justify-content:center;gap:30px;flex-wrap:wrap'>"
        for i in range(n):
            sol_html += f"""<div>
                <div class="result-value">{mat[i][n]:.4f}</div>
                <div style="font-size:.8rem;color:rgba(255,255,255,.4)">x<sub>{i+1}</sub></div>
            </div>"""
        sol_html += "</div></div>"
        st.markdown(sol_html, unsafe_allow_html=True)

        # ── Verifikasi A·x = b ──
        x_sol = mat[:, n]
        residual = np.max(np.abs(A_orig @ x_sol - b_orig))
        st.markdown('<p class="section-label">Verifikasi Hasil</p>', unsafe_allow_html=True)
        st.latex(r"A \cdot x = b")
        st.markdown(f"""<div class="info-card">
        Substitusi solusi x ke dalam sistem semula untuk memeriksa kebenaran:
        residual maksimum |A·x − b| = <b>{residual:.2e}</b>
        {'✓ Solusi akurat (RREF konsisten dengan sistem)' if residual < 1e-8 else '⚠ Ada pembulatan numerik atau sistem mungkin singular/tak konsisten'}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# DETERMINAN & INVERS  (placeholder - belum diimplementasikan)
# ════════════════════════════════════════════════════════════
elif menu == "🔲 Determinan & Invers":
    st.markdown('<p class="section-label">Determinan & Invers</p>', unsafe_allow_html=True)
    st.info("Fitur ini belum diimplementasikan. Silakan tambahkan logika perhitungan determinan dan invers dengan langkah-langkah ekspansi kofaktor di sini.")

# ════════════════════════════════════════════════════════════
# DEKOMPOSISI LU  (placeholder - belum diimplementasikan)
# ════════════════════════════════════════════════════════════
elif menu == "🔺 Dekomposisi LU":
    st.markdown('<p class="section-label">Dekomposisi LU</p>', unsafe_allow_html=True)
    st.info("Fitur ini belum diimplementasikan. Silakan tambahkan logika dekomposisi A = P·L·U dengan langkah-langkah pembentukan multiplier di sini.")

# ════════════════════════════════════════════════════════════
# REGRESI LINEAR  (placeholder - belum diimplementasikan)
# ════════════════════════════════════════════════════════════
elif menu == "📈 Regresi Linear":
    st.markdown('<p class="section-label">Regresi Linear</p>', unsafe_allow_html=True)
    st.info("Fitur ini belum diimplementasikan. Silakan tambahkan logika regresi dengan langkah-langkah normal equations (AᵀAx = Aᵀb) di sini.")
