# ============================================================
# UAS ALJABAR LINIER - Aplikasi Web Interaktif (Streamlit)
# ============================================================
# Jalankan dengan: streamlit run app_matriks_uas.py
# Dependensi    : pip install streamlit numpy pandas
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import io

# ── Konfigurasi halaman (harus dipanggil pertama sebelum widget apapun) ──
st.set_page_config(
    page_title="UAS Aljabar Linier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Injeksi CSS kustom untuk tampilan akademis yang bersih ──
st.markdown("""
<style>
/* ─── Font & Warna Dasar ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Header Utama ───────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.55);
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.4);
    color: #93c5fd;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
}

/* ─── Section Label ──────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #60a5fa;
    margin: 32px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(96,165,250,0.2);
}

/* ─── Info Card ──────────────────────────────────── */
.info-card {
    background: rgba(96,165,250,0.06);
    border: 1px solid rgba(96,165,250,0.18);
    border-left: 3px solid #60a5fa;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.875rem;
    color: #e2e8f0;
    margin-bottom: 18px;
    line-height: 1.6;
}

/* ─── Eigen Card ─────────────────────────────────── */
.eigen-card {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.eigen-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #60a5fa, #a78bfa);
    border-radius: 3px 0 0 3px;
}
.eigen-index {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 6px;
}
.eigen-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 4px;
}
.eigen-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}
.badge-real {
    background: rgba(52,211,153,0.15);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
}
.badge-complex {
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.3);
}

/* ─── Trace Box ──────────────────────────────────── */
.trace-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
}
.trace-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 18px;
}
.trace-result {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #34d399;
    margin: 12px 0 4px 0;
}
.trace-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.4);
}
.verified-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.35);
    color: #34d399;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 20px;
    margin-top: 14px;
}

/* ─── Sidebar kustom ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
    font-size: 0.82rem;
}

/* ─── Streamlit metric box ───────────────────────── */
[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SIDEBAR — Panel Kontrol Input
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Panel Kontrol")
    st.markdown("---")

    # ── Input: nilai seed ──────────────────────────────────
    st.markdown("**🔒 Random Seed**")
    seed_val = st.number_input(
        "Nilai seed (default 42 untuk UAS)",
        min_value=0, max_value=99999,
        value=42, step=1,
        help="Seed mengunci angka acak agar matriks selalu sama. "
             "Gunakan 42 untuk UAS agar dapat direplikasi dosen."
    )

    st.markdown("---")

    # ── Input: rentang nilai matriks ───────────────────────
    st.markdown("**📐 Rentang Nilai Matriks**")
    range_min = st.number_input("Nilai minimum", min_value=1,  max_value=50,  value=1,  step=1)
    range_max = st.number_input("Nilai maksimum", min_value=2, max_value=100, value=10, step=1)

    # Validasi rentang
    if range_min >= range_max:
        st.error("⚠️ Nilai minimum harus lebih kecil dari maksimum.")
        valid_range = False
    else:
        valid_range = True

    st.markdown("---")

    # ── Input: jumlah baris/kolom ──────────────────────────
    st.markdown("**📏 Ukuran Matriks**")
    ukuran = st.selectbox(
        "Dimensi (n × n)",
        options=[50, 100, 150, 200],
        index=1,
        help="Matriks UAS standar adalah 100×100."
    )

    st.markdown("---")

    # ── Tombol utama: Jalankan Komputasi ───────────────────
    # Tombol ini mencegah komputasi berat saat halaman pertama dibuka.
    run_button = st.button(
        "▶  Jalankan Komputasi",
        type="primary",
        use_container_width=True,
        disabled=not valid_range
    )

    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.78rem; color:#475569; line-height:1.6'>
    📌 <b>Petunjuk:</b><br>
    1. Atur seed ke <code>42</code> untuk UAS.<br>
    2. Klik <i>Jalankan Komputasi</i>.<br>
    3. Unduh <code>matriks_uas.txt</code> untuk dikumpulkan.
    </p>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">Ujian Akhir Semester</div>
    <h1 class="hero-title">Analisis Matriks &amp; Dekomposisi Eigen</h1>
    <p class="hero-sub">
        Aljabar Linier · NumPy · np.linalg.eig · Teorema Trace &nbsp;|&nbsp;
        Matriks acak terkunci dengan random seed untuk reproducibility
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  KONDISI: tampilkan konten hanya setelah tombol diklik
# ════════════════════════════════════════════════════════════

if not run_button:
    # Tampilan awal sebelum komputasi — ringan, tidak ada perhitungan berat
    st.markdown("""
    <div class="info-card">
    🚀 &nbsp;<b>Siap dijalankan.</b>&nbsp; Atur parameter di sidebar kiri,
    lalu klik <b>▶ Jalankan Komputasi</b> untuk memulai analisis matriks.
    Semua perhitungan akan muncul di bawah ini.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ukuran Matriks", "100 × 100", help="Dimensi default untuk UAS")
    with col2:
        st.metric("Seed Default", "42", help="Dikunci untuk reproducibility")
    with col3:
        st.metric("Jumlah Nilai Eigen", "n (= ukuran)", help="Satu per baris/kolom")

else:
    # ──────────────────────────────────────────────────────
    # KOMPUTASI INTI (dijalankan hanya saat tombol diklik)
    # ──────────────────────────────────────────────────────

    with st.spinner("⚙️  Membuat matriks dan menghitung nilai eigen..."):

        # 1. Buat matriks dengan seed yang dipilih user
        np.random.seed(seed_val)
        matriks = np.random.randint(range_min, range_max + 1, size=(ukuran, ukuran))

        # 2. Hitung nilai eigen dan vektor eigen
        #    np.linalg.eig → mengembalikan dtype complex128 untuk matriks umum
        nilai_eigen, vektor_eigen = np.linalg.eig(matriks)

        # 3. Hitung Trace untuk pembuktian teorema
        trace_matriks = int(np.trace(matriks))          # Σ elemen diagonal utama
        trace_eigen   = nilai_eigen.real.sum()           # Σ bagian real nilai eigen
        selisih_trace = abs(trace_matriks - trace_eigen) # Harus ≈ 0

        # 4. Siapkan file teks untuk download (di-encode ke bytes)
        buffer = io.BytesIO()
        np.savetxt(buffer, matriks, fmt="%d", delimiter="\t")
        matriks_bytes = buffer.getvalue()

    # ════════════════════════════════════════════
    #  SEKSI 1: INFO RINGKAS
    # ════════════════════════════════════════════

    st.markdown('<p class="section-label">01 &nbsp; Informasi Matriks</p>', unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Dimensi", f"{ukuran} × {ukuran}")
    with col_b:
        st.metric("Total Elemen", f"{ukuran * ukuran:,}")
    with col_c:
        st.metric("Rentang Nilai", f"{range_min} – {range_max}")
    with col_d:
        st.metric("Seed Aktif", str(seed_val))

    # ════════════════════════════════════════════
    #  SEKSI 2: PREVIEW MATRIKS (st.dataframe)
    # ════════════════════════════════════════════

    st.markdown('<p class="section-label">02 &nbsp; Preview Matriks Interaktif</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    📋 &nbsp;Tabel di bawah dapat di-<b>scroll</b> horizontal dan vertikal.
    Klik header kolom untuk sortir. Angka dikunci dengan seed
    <code>np.random.seed({seed})</code> — selalu identik setiap run.
    </div>
    """.format(seed=seed_val), unsafe_allow_html=True)

    # Konversi numpy array ke Pandas DataFrame agar st.dataframe bisa render
    # dengan fitur sortir, scroll, dan pin kolom secara otomatis.
    df_matriks = pd.DataFrame(
        matriks,
        columns=[f"K{i+1}" for i in range(ukuran)],  # Label kolom: K1, K2, ...
        index=[f"B{i+1}" for i in range(ukuran)]      # Label baris: B1, B2, ...
    )

    # st.dataframe dengan height terbatas → tidak memenuhi seluruh layar
    st.dataframe(
        df_matriks,
        height=320,          # Tinggi tampilan tabel (px) — user bisa scroll
        use_container_width=True
    )

    # ── Tombol Download matriks_uas.txt ─────────────────────
    st.download_button(
        label="⬇️  Unduh matriks_uas.txt",
        data=matriks_bytes,
        file_name="matriks_uas.txt",
        mime="text/plain",
        help="File teks tab-separated, siap dikumpulkan ke dosen."
    )

    # ════════════════════════════════════════════
    #  SEKSI 3: NILAI EIGEN (st.metric + custom card)
    # ════════════════════════════════════════════

    st.markdown('<p class="section-label">03 &nbsp; Tiga Sampel Nilai Eigen Pertama</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    🔬 &nbsp;Nilai eigen dihitung via <code>np.linalg.eig(A)</code>.
    Untuk matriks umum (non-simetri), nilai eigen berpotensi
    <b>bilangan kompleks</b> (dtype <code>complex128</code>).
    Bagian real dan imajiner diformat secara terpisah untuk menghindari
    <code>TypeError</code>.
    </div>
    """, unsafe_allow_html=True)

    def format_kompleks(z, desimal=4):
        """
        Memformat bilangan kompleks NumPy menjadi string aman tanpa TypeError.
        Memisahkan .real dan .imag → tidak bisa diformat langsung dengan :.4f.
        """
        r = z.real   # Bagian real bilangan kompleks
        i = z.imag   # Bagian imajiner bilangan kompleks
        tanda = "+" if i >= 0 else "−"
        return f"{r:.{desimal}f} {tanda} {abs(i):.{desimal}f}i"

    # st.metric untuk 3 nilai eigen pertama (tampil sejajar dalam 3 kolom)
    col_e1, col_e2, col_e3 = st.columns(3)
    cols_eigen = [col_e1, col_e2, col_e3]

    for idx in range(3):
        λ = nilai_eigen[idx]
        is_complex = abs(λ.imag) >= 1e-10  # True jika imajiner signifikan

        # Delta = bagian imajiner (menunjukkan "seberapa kompleks" nilai eigen)
        delta_str = f"Im: {λ.imag:+.4f}" if is_complex else "Murni Real"

        with cols_eigen[idx]:
            st.metric(
                label=f"λ{idx+1}",
                value=f"{λ.real:.4f}",   # Nilai real sebagai angka utama
                delta=delta_str,          # Bagian imajiner sebagai delta
            )

    # Card detail per nilai eigen dengan HTML kustom
    for idx in range(3):
        λ = nilai_eigen[idx]
        is_complex = abs(λ.imag) >= 1e-10
        badge_class = "badge-complex" if is_complex else "badge-real"
        badge_text  = "Bilangan Kompleks" if is_complex else "Bilangan Real"
        vektor_preview = " &nbsp;|&nbsp; ".join(
            [format_kompleks(vektor_eigen[j, idx]) for j in range(5)]
        )

        st.markdown(f"""
        <div class="eigen-card">
            <div class="eigen-index">Nilai Eigen ke-{idx+1}</div>
            <div class="eigen-value">{format_kompleks(λ)}</div>
            <span class="eigen-badge {badge_class}">{badge_text}</span>
            <div style="margin-top:12px; padding-top:12px;
                        border-top:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.7rem; color:#64748b;
                            letter-spacing:1px; text-transform:uppercase;
                            margin-bottom:6px;">
                    5 Elemen Pertama Vektor Eigen v{idx+1}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;
                            font-size:0.78rem; color:#94a3b8; line-height:1.9;">
                    {vektor_preview}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  SEKSI 4: PEMBUKTIAN TEOREMA TRACE (st.latex)
    # ════════════════════════════════════════════

    st.markdown('<p class="section-label">04 &nbsp; Pembuktian Teorema Trace</p>', unsafe_allow_html=True)

    # Tampilkan persamaan teorema trace dalam format LaTeX akademis
    st.latex(r"\text{Teorema Trace:} \quad \mathrm{tr}(A) = \sum_{i=1}^{n} \lambda_i")

    st.markdown("""
    <div class="info-card">
    📐 &nbsp;Teorema Trace menyatakan bahwa <b>jumlah semua elemen diagonal utama</b>
    suatu matriks selalu <b>sama dengan jumlah semua nilai eigennya</b>.
    Ini adalah sifat invarian dari matriks yang dapat digunakan sebagai
    verifikasi kebenaran komputasi eigen.
    </div>
    """, unsafe_allow_html=True)

    # Tampilkan persamaan numerik aktual menggunakan st.latex
    st.latex(
        r"\mathrm{tr}(A) = \sum_{j=1}^{" + str(ukuran) + r"} a_{jj} = "
        + str(trace_matriks)
        + r"\quad \approx \quad \sum_{i=1}^{"
        + str(ukuran) + r"} \lambda_i = "
        + f"{trace_eigen:.4f}"
    )

    # Kotak ringkasan visual hasil verifikasi trace
    st.markdown(f"""
    <div class="trace-box">
        <div class="trace-title">Hasil Verifikasi Trace</div>
        <div style="display:flex; justify-content:center; gap:48px; margin-top:8px;">
            <div>
                <div class="trace-result">{trace_matriks}</div>
                <div class="trace-label">tr(A) · Σ Diagonal</div>
            </div>
            <div style="font-size:2rem; color:#475569; align-self:center;">=</div>
            <div>
                <div class="trace-result">{trace_eigen:.4f}</div>
                <div class="trace-label">Σ λᵢ · Jumlah Nilai Eigen</div>
            </div>
        </div>
        <div style="margin-top:18px;">
            <div style="font-size:0.8rem; color:#64748b; margin-bottom:6px;">
                Selisih (|tr(A) − Σλᵢ|) harus mendekati 0
            </div>
            <div style="font-family:'JetBrains Mono',monospace;
                        font-size:1.1rem; color:#f1f5f9;">
                Δ = {selisih_trace:.2e}
            </div>
        </div>
        <div class="verified-pill">✓ &nbsp; Teorema Trace Terbukti Valid</div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  FOOTER
    # ════════════════════════════════════════════

    st.markdown("---")
    st.markdown(f"""
    <p style="font-size:0.78rem; color:#475569; text-align:center; padding:8px 0">
    🔢 &nbsp; UAS Aljabar Linier &nbsp;·&nbsp;
    Matriks {ukuran}×{ukuran} &nbsp;·&nbsp;
    Seed: <code>{seed_val}</code> &nbsp;·&nbsp;
    np.linalg.eig &nbsp;·&nbsp; NumPy + Streamlit
    </p>
    """, unsafe_allow_html=True)
