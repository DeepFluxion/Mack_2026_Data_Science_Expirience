# ══════════════════════════════════════════════════════════════════════════════
#  MLClassifier — Classificação Supervisionada com Streamlit
#  Execução: streamlit run app.py
# ══════════════════════════════════════════════════════════════════════════════

import io, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from pandas.api.types import is_numeric_dtype

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MLClassifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "figure.dpi": 110,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "df_raw": None, "filename": "", "feature_cols": [], "target_col": None,
    "test_size": 0.2, "use_scaler": True, "X_scaled": None, "df_work": None,
    "y": None, "y_raw": None, "X_train": None, "X_test": None,
    "y_train": None, "y_test": None, "X_train_svm": None, "y_train_svm": None,
    "feature_names": [], "classes": [], "neg_label": "0", "pos_label": "1",
    "scale_pos_weight": 1.0,
    "pca_full": None, "pca_2d": None, "X_pca": None,
    "var_ratio": None, "cum_var": None, "pc1_var": 0.0, "pc2_var": 0.0,
    "K_range": None, "inertias": [], "silhouettes": [],
    "best_k": 3, "k_final": None, "kmeans": None, "cluster_labels": None,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

for _k in ("models", "results", "roc_curves", "pr_curves", "figs"):
    if _k not in st.session_state:
        st.session_state[_k] = {}

ss = st.session_state

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def reset_all():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    for k in ("models", "results", "roc_curves", "pr_curves", "figs"):
        st.session_state[k] = {}


def reset_models():
    for k in ("models", "results", "roc_curves", "pr_curves", "figs"):
        st.session_state[k] = {}
    for k in ("K_range", "inertias", "silhouettes", "k_final", "kmeans", "cluster_labels"):
        st.session_state[k] = _DEFAULTS.get(k)
    st.session_state["best_k"] = 3
    for k in ("pca_full", "pca_2d", "X_pca", "var_ratio", "cum_var"):
        st.session_state[k] = None
    st.session_state["pc1_var"] = 0.0
    st.session_state["pc2_var"] = 0.0


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def detect_sep(raw: bytes) -> str:
    sample = raw[:8192].decode("utf-8", errors="ignore")
    counts = {s: sample.count(s) for s in [",", ";", "\t", "|"]}
    return max(counts, key=counts.get)


def detect_id_cols(df: pd.DataFrame) -> list:
    out = []
    for col in df.columns:
        low = col.lower().strip()
        if low in ("id","index","idx","codigo","code","pk","uid","nr","no"):
            out.append(col)
        elif is_numeric_dtype(df[col]) and df[col].nunique() == len(df):
            out.append(col)
    return out


def pipeline_ready() -> bool:
    return ss.X_train is not None and ss.y_train is not None


def run_model(clf, clf_name: str, use_svm: bool = False) -> None:
    """Train, evaluate, store results + figure bytes. Does NOT display."""
    X_tr = ss.X_train_svm if (use_svm and ss.X_train_svm is not None) else ss.X_train
    y_tr = ss.y_train_svm if (use_svm and ss.y_train_svm is not None) else ss.y_train

    with st.spinner(f"Treinando {clf_name}..."):
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        t_fit = round(time.time() - t0, 2)

    y_pred = clf.predict(ss.X_test)
    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(ss.X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        sc = clf.decision_function(ss.X_test)
        y_prob = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)

    acc  = accuracy_score(ss.y_test, y_pred)
    prec = precision_score(ss.y_test, y_pred, zero_division=0)
    rec  = recall_score(ss.y_test, y_pred, zero_division=0)
    f1   = f1_score(ss.y_test, y_pred, zero_division=0)

    fpr_a = tpr_a = roc_v = None
    prec_c = rec_c = ap_v = None
    if y_prob is not None:
        fpr_a, tpr_a, _ = roc_curve(ss.y_test, y_prob)
        roc_v = auc(fpr_a, tpr_a)
        prec_c, rec_c, _ = precision_recall_curve(ss.y_test, y_prob)
        ap_v = average_precision_score(ss.y_test, y_prob)

    ss.models[clf_name] = clf
    ss.results[clf_name] = {
        "Acurácia":    round(acc,  4),
        "Precisão(+)": round(prec, 4),
        "Recall(+)":   round(rec,  4),
        "F1(+)":       round(f1,   4),
        "AUC-ROC":     round(roc_v, 4) if roc_v is not None else None,
        "AUC-PR":      round(ap_v,  4) if ap_v  is not None else None,
        "Tempo(s)":    t_fit,
        "_report":     classification_report(
                            ss.y_test, y_pred,
                            target_names=[ss.neg_label, ss.pos_label]),
    }
    if fpr_a is not None:  ss.roc_curves[clf_name] = (fpr_a, tpr_a, roc_v)
    if prec_c is not None: ss.pr_curves[clf_name]  = (prec_c, rec_c, ap_v)

    # ── Figura 2×2 ────────────────────────────────────────────────────────────
    fn  = ss.feature_names
    TOP = min(12, len(fn))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Resultados: {clf_name}", fontsize=13, fontweight="bold")

    # [0,0] Matriz de Confusão
    ax = axes[0, 0]
    cm = confusion_matrix(ss.y_test, y_pred)
    xl = [f"{ss.neg_label} (0)", f"{ss.pos_label} (1)"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=xl, yticklabels=xl)
    ax.set_title("Matriz de Confusão", fontweight="bold")
    ax.set_ylabel("Real"); ax.set_xlabel("Previsto")
    ax.text(0.5, -0.15,
            f"TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}",
            ha="center", transform=ax.transAxes, fontsize=8, color="gray")

    # [0,1] Curva ROC
    ax = axes[0, 1]
    if fpr_a is not None:
        ax.plot(fpr_a, tpr_a, color="darkorange", lw=2.5, label=f"AUC-ROC={roc_v:.3f}")
        ax.plot([0,1],[0,1], "k--", lw=1.5, label="Baseline")
        ax.fill_between(fpr_a, tpr_a, alpha=0.1, color="darkorange")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14, color="gray")
    ax.set_title("Curva ROC", fontweight="bold")

    # [1,0] Curva Precision-Recall
    ax = axes[1, 0]
    if prec_c is not None:
        bl = ss.y_test.mean()
        ax.plot(rec_c, prec_c, color="navy", lw=2.5, label=f"AUC-PR={ap_v:.3f}")
        ax.axhline(bl, color="gray", ls="--", lw=1.5, label=f"Baseline={bl:.3f}")
        ax.fill_between(rec_c, prec_c, alpha=0.1, color="navy")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precisão"); ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14, color="gray")
    ax.set_title("Curva Precision-Recall", fontweight="bold")

    # [1,1] Feature Importance / Coeficientes / Report
    ax = axes[1, 1]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        idx = np.argsort(imp)[-TOP:]
        ax.barh(range(len(idx)), imp[idx], color="steelblue", alpha=0.85)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([fn[i] for i in idx], fontsize=7)
        ax.set_xlabel("Importância (Gini)")
        ax.set_title(f"Top {len(idx)} Features", fontweight="bold")
    elif hasattr(clf, "coef_"):
        coef = clf.coef_.ravel()
        idx  = np.argsort(np.abs(coef))[-TOP:]
        cols = ["#e53935" if coef[i] > 0 else "#42A5F5" for i in idx]
        ax.barh(range(len(idx)), coef[idx], color=cols, alpha=0.85)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([fn[i] for i in idx], fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Coeficiente")
        ax.set_title(f"Top {len(idx)} Coeficientes", fontweight="bold")
        ax.text(0.98, 0.01, "🔴 positivo  🔵 negativo",
                transform=ax.transAxes, ha="right", fontsize=7, color="gray")
    else:
        rep = ss.results[clf_name]["_report"]
        ax.text(0.04, 0.97, rep, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", family="monospace")
        ax.axis("off")
        ax.set_title("Relatório de Classificação", fontweight="bold")

    plt.tight_layout()
    ss.figs[clf_name] = fig_to_bytes(fig)
    plt.close(fig)

    st.success(f"✅ **{clf_name}** treinado em {t_fit}s — resultados abaixo.")


def show_results(clf_name: str):
    """Display stored metrics + figure for a trained model."""
    if clf_name not in ss.results:
        return
    res = ss.results[clf_name]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Acurácia",    f"{res['Acurácia']:.3f}")
    c2.metric("Precisão(+)", f"{res['Precisão(+)']:.3f}")
    c3.metric("Recall(+)",   f"{res['Recall(+)']:.3f}")
    c4.metric("F1(+)",       f"{res['F1(+)']:.3f}")
    roc = res.get("AUC-ROC"); apr = res.get("AUC-PR")
    c5.metric("AUC-ROC", f"{roc:.3f}" if roc else "N/A")
    c6.metric("AUC-PR",  f"{apr:.3f}" if apr else "N/A")

    if clf_name in ss.figs:
        st.image(ss.figs[clf_name], use_container_width=True)
        st.download_button(
            "⬇️ Baixar figura PNG",
            data=ss.figs[clf_name],
            file_name=f"clf_{clf_name.lower().replace(' ','_')}.png",
            mime="image/png",
            key=f"dl_fig_{clf_name}",
        )

    with st.expander("📋 Relatório de Classificação completo"):
        st.text(res.get("_report", "Não disponível"))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 MLClassifier")
        st.markdown("*Classificação Supervisionada*")
        st.divider()

        st.markdown("### ⚙️ Pipeline")
        checks = {
            "Upload": ss.df_raw is not None,
            "Dados preparados": pipeline_ready(),
            "PCA calculado": ss.pca_2d is not None,
            "K-Means treinado": ss.cluster_labels is not None,
        }
        for label, ok in checks.items():
            st.markdown(f"{'✅' if ok else '⬜'} {label}")

        st.divider()

        if ss.df_raw is not None:
            st.markdown("### 📄 Dataset")
            st.markdown(f"`{ss.filename}`")
            c1, c2 = st.columns(2)
            c1.metric("Linhas", f"{len(ss.df_raw):,}")
            c2.metric("Features", str(len(ss.feature_cols)))
            if ss.target_col:
                st.markdown(f"**Target:** `{ss.target_col}`")
            if ss.classes:
                st.markdown(f"**`{ss.classes[0]}`→0  `{ss.classes[1]}`→1**")
            if pipeline_ready():
                c1.metric("Treino", f"{len(ss.X_train):,}")
                c2.metric("Teste",  f"{len(ss.X_test):,}")
            st.divider()

        if ss.results:
            st.markdown("### ✅ Modelos treinados")
            for name, res in ss.results.items():
                ap = res.get("AUC-PR")
                st.markdown(f"• **{name}**: {f'AUC-PR={ap:.3f}' if ap else 'OK'}")
            st.divider()

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Reset tudo", use_container_width=True):
                reset_all(); st.rerun()
        with col_r2:
            if ss.results:
                if st.button("🗑️ Reset modelos", use_container_width=True):
                    reset_models(); st.rerun()

        st.divider()
        st.caption("MBA Data Science | Ferramenta Didática")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — UPLOAD & DADOS
# ══════════════════════════════════════════════════════════════════════════════
def tab_upload():
    st.markdown("## 📁 Upload & Preparação dos Dados")
    st.divider()

    c_file, c_opts = st.columns([3, 1])
    with c_opts:
        st.markdown("#### ⚙️ Leitura")
        sep_ch = st.radio("Separador", ["Auto", ",", ";", "TAB"], key="up_sep")
        enc_ch = st.radio("Encoding",  ["UTF-8", "Latin-1"], key="up_enc")

    with c_file:
        uploaded = st.file_uploader("CSV com cabeçalho", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.info("📂 Aguardando upload. Requisitos: cabeçalho na 1ª linha · ≥ 2 colunas · target binário.")
        return

    raw = uploaded.read()
    enc = "utf-8" if enc_ch == "UTF-8" else "latin-1"
    sep_map = {"Auto": detect_sep(raw), ",": ",", ";": ";", "TAB": "\t"}
    sep = sep_map[sep_ch]

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
    except Exception as e:
        st.error(f"❌ Erro: {e}"); return

    if ss.filename != uploaded.name:
        reset_all()
    ss.df_raw = df; ss.filename = uploaded.name

    st.success(f"✅ **{uploaded.name}** · `{df.shape[0]:,} × {df.shape[1]}` · sep=`{repr(sep)}`")
    st.dataframe(df.head(5), use_container_width=True)
    st.divider()

    id_cols  = detect_id_cols(df)
    all_cols = [c for c in df.columns if c not in id_cols]

    c_feat, c_tgt = st.columns([2, 1])
    with c_feat:
        st.markdown("### 🔢 Features")
        if id_cols:
            st.caption(f"⚠️ Prováveis IDs ignorados: `{'`, `'.join(id_cols[:4])}`")
        def_feats = [c for c in (ss.feature_cols or all_cols[:16]) if c in all_cols]
        feat_cols = st.multiselect("Selecione as features", all_cols, default=def_feats, key="up_feat")

    with c_tgt:
        st.markdown("### 🎯 Target")
        opts = ["(nenhuma)"] + list(df.columns)
        cur  = ss.target_col if ss.target_col in df.columns else "(nenhuma)"
        tgt  = st.selectbox("Coluna alvo (binária)", opts, index=opts.index(cur), key="up_tgt")
        tgt  = None if tgt == "(nenhuma)" else tgt
        if tgt:
            uv = df[tgt].nunique()
            st.info(f"{uv} valores únicos: `{list(df[tgt].unique()[:5])}`")
            if uv != 2:
                st.error("❌ Target deve ter exatamente 2 classes!"); tgt = None

    st.divider()
    st.markdown("### 🔧 Configuração do Pipeline")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Percentual de Teste (%)", 10, 40,
                               int((ss.test_size or 0.2) * 100), 5, key="up_ts") / 100
    with c2:
        use_scaler = st.checkbox("StandardScaler", value=bool(ss.use_scaler), key="up_sc",
                                  help="Padroniza: média=0, desvio=1. Obrigatório para SVM, KNN, MLP.")
    with c3:
        st.info("**Dica:** Features categóricas → OrdinalEncoder automático")

    st.divider()
    c_btn, c_msg = st.columns([1, 3])
    with c_btn:
        apply = st.button("✅ Preparar Dados", type="primary", use_container_width=True, key="up_apply")

    if apply:
        feat_cols = [c for c in feat_cols if c != tgt]
        if len(feat_cols) < 2 or not tgt:
            st.error("❌ Selecione pelo menos 2 features e 1 target binário."); return

        df_clean = df[feat_cols + [tgt]].dropna()
        X_df = df_clean[feat_cols]
        y_series = df_clean[tgt]

        le = LabelEncoder()
        y  = le.fit_transform(y_series)
        if len(le.classes_) != 2:
            st.error("❌ Target deve ter exatamente 2 classes."); return

        num_f = [c for c in feat_cols if is_numeric_dtype(X_df[c])]
        cat_f = [c for c in feat_cols if not is_numeric_dtype(X_df[c])]

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts = []
        if num_f: parts.append(X_df[num_f].values.astype(float))
        if cat_f: parts.append(enc.fit_transform(X_df[cat_f]))
        X_comb = np.column_stack(parts) if len(parts) > 1 else (parts[0] if parts else np.zeros((len(df_clean), 1)))
        feat_names = num_f + cat_f

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_comb) if use_scaler else X_comb

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y)

        N_SVM = min(10000, len(X_train))
        X_tr_svm, y_tr_svm = resample(X_train, y_train, n_samples=N_SVM, stratify=y_train, random_state=42)

        ss.feature_cols    = feat_cols
        ss.target_col      = tgt
        ss.test_size       = test_size
        ss.use_scaler      = use_scaler
        ss.X_scaled        = X_scaled
        ss.df_work         = df_clean[feat_cols].reset_index(drop=True)
        ss.y               = y
        ss.y_raw           = y_series.values
        ss.X_train         = X_train;  ss.X_test  = X_test
        ss.y_train         = y_train;  ss.y_test  = y_test
        ss.X_train_svm     = X_tr_svm; ss.y_train_svm = y_tr_svm
        ss.feature_names   = feat_names
        ss.classes         = list(le.classes_)
        ss.neg_label       = le.classes_[0]
        ss.pos_label       = le.classes_[1]
        ss.scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        reset_models()

        with c_msg:
            st.success(f"✅ **{len(df_clean):,} obs × {len(feat_cols)} features** prontas!")
            st.markdown(f"Treino: **{len(X_train):,}** · Teste: **{len(X_test):,}** · "
                        f"Classes: `{le.classes_[0]}`=0 / `{le.classes_[1]}`=1")
            pct_pos = y_train.mean() * 100
            if pct_pos < 35 or pct_pos > 65:
                st.warning(f"⚠️ Desbalanceamento: {pct_pos:.1f}% positivos. "
                           "Modelos com `class_weight='balanced'` são recomendados.")

    if pipeline_ready():
        st.divider()
        st.markdown("### 📊 Distribuição do Target")
        vc = pd.Series(ss.y_raw).value_counts()
        fig_b, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        cores = ["#42A5F5", "#FF5722"]
        axes[0].bar(vc.index.astype(str), vc.values, color=cores[:len(vc)], alpha=0.88, edgecolor="white")
        for i, (l, c) in enumerate(zip(vc.index.astype(str), vc.values)):
            axes[0].text(i, c + len(ss.y_raw)*0.008, f"{c:,}\n({c/len(ss.y_raw)*100:.1f}%)",
                         ha="center", fontsize=10, fontweight="bold")
        axes[0].set_title("Contagem por Classe", fontweight="bold"); axes[0].set_xlabel(ss.target_col)
        axes[1].pie(vc.values, labels=vc.index.astype(str), colors=cores[:len(vc)],
                    autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Proporção", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_b); plt.close(fig_b)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — PCA & CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════
def tab_pca():
    st.markdown("## 📐 PCA & Clusterização K-Means")

    with st.expander("📖 Teoria — PCA & K-Means", expanded=False):
        st.markdown("""
**PCA (Análise de Componentes Principais)** busca as **direções de máxima variância** nos dados
multidimensionais, comprimindo-os em componentes ortogonais (CPs). O **Biplot** combina a
projeção das observações (pontos) e as setas dos *loadings* — indicando o quanto cada variável
original contribui para cada CP.

**K-Means** divide os dados em K grupos via centróides iterativos:
1. Inicializa K centróides aleatórios
2. Atribui cada ponto ao centróide mais próximo
3. Recalcula centróides como média do grupo
4. Repete até convergência

Dois critérios para escolha de K:
- **Método do Cotovelo:** busca o "joelho" na curva de inércia
- **Silhouette Score:** mede coesão vs. separação dos clusters (−1 a +1)

> 💡 K-Means é **não supervisionado** — não usa o target. Se os clusters coincidirem
> com as classes reais, existe estrutura genuína nos dados.
        """)

    if not pipeline_ready():
        st.warning("⬅️ Configure os dados na aba **Upload & Dados** primeiro.")
        return

    X = ss.X_scaled

    if ss.pca_full is None:
        with st.spinner("Calculando PCA..."):
            pf = PCA(); pf.fit(X)
            p2 = PCA(n_components=min(2, X.shape[1])); Xp = p2.fit_transform(X)
            ss.pca_full  = pf; ss.pca_2d = p2; ss.X_pca = Xp
            ss.var_ratio = pf.explained_variance_ratio_
            ss.cum_var   = np.cumsum(pf.explained_variance_ratio_)
            ss.pc1_var   = float(pf.explained_variance_ratio_[0]) * 100
            ss.pc2_var   = float(pf.explained_variance_ratio_[1]) * 100 if len(pf.explained_variance_ratio_) > 1 else 0.0

    vr = ss.var_ratio; cv = ss.cum_var
    p1v = ss.pc1_var; p2v = ss.pc2_var; Xp = ss.X_pca
    n_f = len(ss.feature_names)
    n85 = int(np.argmax(cv >= 0.85) + 1) if any(cv >= 0.85) else n_f

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PC1 explica", f"{p1v:.1f}%"); c2.metric("PC2 explica", f"{p2v:.1f}%")
    c3.metric("CPs para ≥85%", str(n85));    c4.metric("Total 2D", f"{p1v+p2v:.1f}%")
    st.divider()

    # ── Scree + Variância Acumulada ────────────────────────────────────────────
    st.markdown("### 1️⃣ Variância Explicada")
    x_tks = list(range(1, n_f + 1)); xlbls = [f"PC{i}" for i in x_tks]

    fig1, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    bars = ax.bar(x_tks, vr * 100, color="steelblue", alpha=0.85, edgecolor="white")
    ax.plot(x_tks, vr * 100, "o-", color="navy", lw=2, markersize=5)
    ax.set_xlabel("Componente"); ax.set_ylabel("Variância (%)"); ax.set_title("Scree Plot", fontweight="bold")
    ax.set_xticks(x_tks); ax.set_xticklabels(xlbls, rotation=45, fontsize=8)
    for b, v in zip(bars, vr):
        if v * 100 > 2:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v*100:.1f}%",
                    ha="center", fontsize=7, fontweight="bold")

    ax = axes[1]
    ax.plot(x_tks, cv * 100, "s-", color="darkorange", lw=2.5, markersize=7)
    ax.fill_between(x_tks, cv * 100, alpha=0.12, color="darkorange")
    for th, col, lbl in [(80,"red","80%"), (90,"green","90%")]:
        ax.axhline(th, color=col, ls="--", alpha=0.65, lw=1.5, label=f"Limiar {lbl}")
    ax.axvline(n85, color="purple", ls=":", lw=2, label=f"PC{n85}→≥85%")
    ax.set_xlabel("Nº de Componentes"); ax.set_ylabel("Var. Acumulada (%)")
    ax.set_title("Variância Acumulada", fontweight="bold")
    ax.set_xticks(x_tks); ax.set_xticklabels(xlbls, rotation=45, fontsize=8)
    ax.set_ylim([0, 110]); ax.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)
    st.divider()

    # ── Biplot ─────────────────────────────────────────────────────────────────
    st.markdown("### 2️⃣ Biplot PCA")
    c_sl, c_col = st.columns([1, 1])
    with c_sl:
        arrow_scale = st.slider("Escala das setas", 0.5, 5.0, 3.0, 0.5, key="pca_arr")
    with c_col:
        c_by = st.radio("Colorir por", ["Target (y)", "Sem cor"], horizontal=True, key="pca_col")

    y_int = pal_y = cls_plot = None
    if c_by == "Target (y)" and ss.y_raw is not None:
        cls_plot = ss.classes
        mp = {c: i for i, c in enumerate(cls_plot)}
        y_int = np.array([mp[v] for v in ss.y_raw])
        pal_y = sns.color_palette("tab10", n_colors=2)

    n_sc = len(Xp)
    if n_sc > 8000:
        idx = np.random.default_rng(42).choice(n_sc, 8000, replace=False)
        Xp_p = Xp[idx]; yi_p = y_int[idx] if y_int is not None else None
    else:
        Xp_p = Xp; yi_p = y_int

    fig2, ax = plt.subplots(figsize=(11, 7))
    if yi_p is not None:
        for i, cls in enumerate(cls_plot):
            mask = yi_p == i
            ax.scatter(Xp_p[mask,0], Xp_p[mask,1], color=pal_y[i],
                       label=str(cls), alpha=0.25, s=8, rasterized=True)
        ax.legend(title=ss.target_col, fontsize=9, markerscale=3)
    else:
        ax.scatter(Xp_p[:,0], Xp_p[:,1], color="steelblue", alpha=0.2, s=8, rasterized=True)

    ldg = ss.pca_2d.components_.T
    for i, var in enumerate(ss.feature_names):
        lx = ldg[i, 0] * arrow_scale; ly = ldg[i, 1] * arrow_scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.8, mutation_scale=15))
        tx = lx*1.13 if abs(lx) > 0.15 else lx + 0.15*(1 if lx >= 0 else -1)
        ty = ly*1.13 if abs(ly) > 0.15 else ly + 0.12*(1 if ly >= 0 else -1)
        ax.text(tx, ty, var, fontsize=8.5, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.78, ec="lightgray"))

    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlabel(f"PC1 ({p1v:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({p2v:.1f}%)", fontsize=11)
    ax.set_title("Biplot PCA — Pontos: observações | Setas: loadings das features", fontsize=12, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    st.divider()

    # ── K-Means ─────────────────────────────────────────────────────────────
    st.markdown("### 3️⃣ K-Means Clustering")

    c_sl2, c_btn2, _ = st.columns([2, 1, 2])
    with c_sl2:
        k_max = st.slider("K máximo", 5, 15, 10, key="km_kmax")
    with c_btn2:
        st.write("")
        calc_km = st.button("🔄 Calcular Elbow + Silhouette", key="btn_km_calc", use_container_width=True)

    if calc_km:
        K_range = range(2, k_max + 1); ine = []; sils = []
        pb = st.progress(0, text="Calculando...")
        n_sil = min(5000, len(X))
        for i, k in enumerate(K_range):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X); ine.append(km.inertia_)
            sils.append(silhouette_score(X, km.labels_, sample_size=n_sil, random_state=42))
            pb.progress((i+1)/len(K_range), text=f"K={k}...")
        pb.progress(1.0, text="✅ Concluído!")
        ss.K_range = K_range; ss.inertias = ine; ss.silhouettes = sils
        ss.best_k = list(K_range)[int(np.argmax(sils))]
        ss.k_final = None; ss.kmeans = None; ss.cluster_labels = None

    if ss.inertias:
        K_list = list(ss.K_range)
        fig3, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        ax.plot(K_list, ss.inertias, "o-", color="steelblue", lw=2.5, markersize=9)
        ax.fill_between(K_list, ss.inertias, alpha=0.1, color="steelblue")
        ax.set_xlabel("K"); ax.set_ylabel("Inércia")
        ax.set_title("Curva do Cotovelo", fontweight="bold"); ax.set_xticks(K_list)

        colors_s = ["#e53935" if k == ss.best_k else "#90CAF9" for k in K_list]
        ax = axes[1]
        bars = ax.bar(K_list, ss.silhouettes, color=colors_s, alpha=0.9, edgecolor="white", width=0.6)
        for b, s in zip(bars, ss.silhouettes):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{s:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
        ax.set_title(f"Silhouette — Melhor K={ss.best_k}", fontweight="bold"); ax.set_xticks(K_list)
        plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

        st.info(f"💡 Silhouette sugere K = **{ss.best_k}**. Confirme abaixo.")
        c_ki, c_tr2, _ = st.columns([1, 1, 2])
        with c_ki:
            k_ch = st.number_input("K final", 2, int(k_max), int(ss.k_final or ss.best_k), key="km_kf")
        with c_tr2:
            st.write("")
            trn_km = st.button("🚀 Treinar K-Means", type="primary", key="btn_km_train", use_container_width=True)

        if trn_km:
            with st.spinner("Treinando K-Means..."):
                km2 = KMeans(n_clusters=int(k_ch), random_state=42, n_init=20)
                lbl = km2.fit_predict(X)
            ss.k_final = int(k_ch); ss.kmeans = km2; ss.cluster_labels = lbl
            st.success(f"✅ K-Means treinado com K={k_ch}!")

        if ss.cluster_labels is not None:
            lbl = ss.cluster_labels; k_f = ss.k_final
            pal_cl = sns.color_palette("tab10", n_colors=k_f)
            cnts = pd.Series(lbl).value_counts().sort_index()
            col_d = st.columns(min(k_f, 6))
            for k in range(k_f):
                cnt = cnts.get(k, 0)
                with col_d[k % len(col_d)]:
                    st.metric(f"Cluster {k}", f"{cnt:,}", f"{cnt/len(lbl)*100:.1f}%")

            ncols = 2 if ss.target_col and y_int is not None else 1
            fig4, axes = plt.subplots(1, ncols, figsize=(8*ncols, 5.5))
            if ncols == 1: axes = [axes]

            n_s2 = len(Xp)
            if n_s2 > 8000:
                idx2 = np.random.default_rng(42).choice(n_s2, 8000, replace=False)
                Xp2 = Xp[idx2]; cl2 = lbl[idx2]
                yi2 = y_int[idx2] if y_int is not None else None
            else:
                Xp2 = Xp; cl2 = lbl; yi2 = y_int

            ax = axes[0]
            for k in range(k_f):
                mask = cl2 == k
                ax.scatter(Xp2[mask,0], Xp2[mask,1], color=pal_cl[k],
                           label=f"Cluster {k}", alpha=0.4, s=10, rasterized=True)
            ax.set_xlabel(f"PC1 ({p1v:.1f}%)"); ax.set_ylabel(f"PC2 ({p2v:.1f}%)")
            ax.set_title(f"K-Means ({k_f} Clusters)\nSem usar target", fontweight="bold", fontsize=11)
            ax.legend(title="Cluster", fontsize=8, markerscale=3)
            ax.axhline(0, color="gray", lw=0.4, ls="--"); ax.axvline(0, color="gray", lw=0.4, ls="--")

            if ncols == 2 and yi2 is not None:
                ax = axes[1]
                for i, cls in enumerate(cls_plot or ss.classes):
                    mask = yi2 == i
                    ax.scatter(Xp2[mask,0], Xp2[mask,1], color=pal_y[i],
                               label=str(cls), alpha=0.28, s=10, rasterized=True)
                ax.set_xlabel(f"PC1 ({p1v:.1f}%)"); ax.set_ylabel(f"PC2 ({p2v:.1f}%)")
                ax.set_title(f"Target: {ss.target_col}\nClasse real", fontweight="bold", fontsize=11)
                ax.legend(title=ss.target_col, fontsize=8, markerscale=3)
                ax.axhline(0, color="gray", lw=0.4, ls="--"); ax.axvline(0, color="gray", lw=0.4, ls="--")
                plt.suptitle("Clusters (sem target) vs. Classe real — alinhamento sugere estrutura nos dados",
                             fontsize=9, y=1.01)
            plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)
    elif not calc_km:
        st.info("👆 Clique em **Calcular Elbow + Silhouette** para determinar o K ideal.")


# ══════════════════════════════════════════════════════════════════════════════
# ABAS DOS ALGORITMOS — funções auxiliares de teoria
# ══════════════════════════════════════════════════════════════════════════════
def _header_check(name):
    if not pipeline_ready():
        st.warning("⬅️ Configure os dados na aba **Upload & Dados** primeiro.")
        return False
    if name in ss.results:
        st.info(f"💡 **{name}** já treinado. Clique no botão para retreinar com novos parâmetros.")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — REGRESSÃO LOGÍSTICA
# ══════════════════════════════════════════════════════════════════════════════
def tab_logistic():
    st.markdown("## 📊 Regressão Logística")
    with st.expander("📖 Teoria — Regressão Logística", expanded=False):
        st.markdown("""
A **Regressão Logística** modela a probabilidade de pertencer à classe positiva via função sigmoide:
```
P(y=1|X) = 1 / (1 + e^(-z))    z = β₀ + β₁x₁ + ... + βₙxₙ
```
Os coeficientes β são estimados maximizando a **log-verossimilhança**. Regularização L2 (padrão)
penaliza coeficientes grandes evitando overfitting.

**Vantagens:** Interpretável · Rápido · Probabilidades calibradas · Excelente baseline  
**Desvantagens:** Assume linearidade entre features e log-odds · Não captura interações complexas
        """)
    if not _header_check("Logistic Regression"): return
    st.markdown("### ⚙️ Hiperparâmetros")
    c1, c2, c3 = st.columns(3)
    with c1: C = st.number_input("C (inversão da regularização)", 0.01, 100.0, 1.0, key="lr_C")
    with c2: pen = st.selectbox("Penalty", ["l2", "l1"], key="lr_pen")
    with c3: mi  = st.number_input("Max Iter", 100, 5000, 1000, step=100, key="lr_mi")
    cw = st.checkbox("class_weight='balanced'", True, key="lr_cw")
    if st.button("🚀 Treinar Regressão Logística", type="primary", key="btn_lr"):
        slv = "saga" if pen == "l1" else "lbfgs"
        clf = LogisticRegression(C=C, penalty=pen, max_iter=int(mi),
                                  class_weight="balanced" if cw else None,
                                  solver=slv, random_state=42)
        run_model(clf, "Logistic Regression")
    if "Logistic Regression" in ss.results:
        show_results("Logistic Regression")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 4 — KNN
# ══════════════════════════════════════════════════════════════════════════════
def tab_knn():
    st.markdown("## 🔵 K-Nearest Neighbors (KNN)")
    with st.expander("📖 Teoria — KNN", expanded=False):
        st.markdown("""
O KNN é **lazy learning**: não constrói modelo interno — memoriza os dados de treino.
Para classificar um novo ponto, encontra os K vizinhos mais próximos (distância euclidiana)
e retorna a **classe majoritária**:
- **K pequeno:** fronteira irregular → risco de overfitting
- **K grande:** fronteira suave → pode subajustar

**Vantagens:** Simples · Não-paramétrico · Multiclasse natural  
**Desvantagens:** Lento na predição O(n·d) · Sensível à escala (padronização obrigatória) · Sem class_weight
        """)
    if not _header_check("KNN"): return
    st.markdown("### ⚙️ Hiperparâmetros")
    c1, c2 = st.columns(2)
    with c1: K = st.slider("K (nº de vizinhos)", 1, 51, 7, 2, key="knn_k")
    with c2: wts = st.selectbox("Pesos", ["distance", "uniform"], key="knn_w",
                                  help="'distance': vizinhos próximos têm mais influência")
    if st.button("🚀 Treinar KNN", type="primary", key="btn_knn"):
        clf = KNeighborsClassifier(n_neighbors=K, weights=wts, n_jobs=-1)
        run_model(clf, "KNN")
    if "KNN" in ss.results:
        show_results("KNN")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 5 — SVM LINEAR
# ══════════════════════════════════════════════════════════════════════════════
def tab_svm_linear():
    st.markdown("## ⚡ SVM Linear")
    with st.expander("📖 Teoria — SVM Linear", expanded=False):
        st.markdown("""
A SVM constrói um **hiperplano que maximiza a margem** entre as classes. Apenas os
**vetores de suporte** (pontos na fronteira) determinam o hiperplano:
```
Margem = 2 / ||w||    (a maximizar)
```
O parâmetro **C** controla o trade-off entre margem e erros de classificação.

**Vantagens:** Eficaz em alta dimensionalidade · Robusto a outliers  
**Desvantagens:** Lento para grandes datasets O(n²~n³) · **Sub-amostra de 10k usada aqui**
        """)
    if not _header_check("SVM Linear"): return
    n_svm = len(ss.X_train_svm) if ss.X_train_svm is not None else 0
    st.info(f"⚠️ SVM usa sub-amostra de **{n_svm:,}** registros para performance.")
    c1, c2 = st.columns(2)
    with c1: C = st.number_input("C", 0.001, 100.0, 1.0, key="svml_C")
    with c2: cw = st.checkbox("class_weight='balanced'", True, key="svml_cw")
    if st.button("🚀 Treinar SVM Linear", type="primary", key="btn_svml"):
        clf = SVC(kernel="linear", C=C, class_weight="balanced" if cw else None,
                  probability=True, random_state=42)
        run_model(clf, "SVM Linear", use_svm=True)
    if "SVM Linear" in ss.results:
        show_results("SVM Linear")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 6 — SVM RBF
# ══════════════════════════════════════════════════════════════════════════════
def tab_svm_rbf():
    st.markdown("## 🌀 SVM RBF")
    with st.expander("📖 Teoria — SVM RBF", expanded=False):
        st.markdown("""
O kernel **RBF (Radial Basis Function)** projeta implicitamente os dados em dimensão infinita,
permitindo fronteiras não-lineares:
```
K(xᵢ, xⱼ) = exp(−γ ||xᵢ − xⱼ||²)
```
**γ alto:** influência localizada → fronteira complexa → overfitting  
**γ baixo:** influência ampla → fronteira suave → underfitting

**Vantagens:** Captura fronteiras não-lineares complexas  
**Desvantagens:** Sem interpretabilidade · **Sub-amostra de 10k usada aqui**
        """)
    if not _header_check("SVM RBF"): return
    n_svm = len(ss.X_train_svm) if ss.X_train_svm is not None else 0
    st.info(f"⚠️ SVM usa sub-amostra de **{n_svm:,}** registros para performance.")
    c1, c2, c3 = st.columns(3)
    with c1: C   = st.number_input("C", 0.01, 100.0, 1.0, key="svmr_C")
    with c2: gam = st.selectbox("Gamma", ["scale", "auto"], key="svmr_g")
    with c3: cw  = st.checkbox("class_weight='balanced'", True, key="svmr_cw")
    if st.button("🚀 Treinar SVM RBF", type="primary", key="btn_svmr"):
        clf = SVC(kernel="rbf", C=C, gamma=gam,
                  class_weight="balanced" if cw else None,
                  probability=True, random_state=42)
        run_model(clf, "SVM RBF", use_svm=True)
    if "SVM RBF" in ss.results:
        show_results("SVM RBF")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 7 — ÁRVORE DE DECISÃO
# ══════════════════════════════════════════════════════════════════════════════
def tab_decision_tree():
    st.markdown("## 🌳 Árvore de Decisão")
    with st.expander("📖 Teoria — Árvore de Decisão", expanded=False):
        st.markdown("""
Aprende regras **if-then-else** recursivas. Em cada nó, escolhe a feature e limiar que
minimiza a **impureza de Gini**:
```
Gini(t) = 1 − Σₖ p(k|t)²    (0 = nó puro)
```
**Vantagens:** Altamente interpretável · Visualizável · Não precisa de padronização  
**Desvantagens:** Alta variância · Propenso a overfitting sem poda
        """)
    if not _header_check("Decision Tree"): return
    st.markdown("### ⚙️ Hiperparâmetros")
    c1, c2, c3 = st.columns(3)
    with c1: md  = st.slider("max_depth", 2, 20, 8, key="dt_md")
    with c2: msl = st.slider("min_samples_leaf", 5, 100, 20, key="dt_msl")
    with c3: cw  = st.checkbox("class_weight='balanced'", True, key="dt_cw")

    if st.button("🚀 Treinar Árvore de Decisão", type="primary", key="btn_dt"):
        clf = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl,
                                      class_weight="balanced" if cw else None,
                                      random_state=42)
        run_model(clf, "Decision Tree")

    if "Decision Tree" in ss.results:
        show_results("Decision Tree")
        if "Decision Tree" in ss.models:
            with st.expander("🌳 Visualizar estrutura da Árvore (3 níveis)"):
                clf_dt = ss.models["Decision Tree"]
                fig_t, ax_t = plt.subplots(figsize=(20, 6))
                plot_tree(clf_dt, max_depth=3, feature_names=ss.feature_names,
                          class_names=[ss.neg_label, ss.pos_label],
                          filled=True, fontsize=7, ax=ax_t)
                ax_t.set_title("Árvore de Decisão — 3 primeiros níveis", fontweight="bold")
                plt.tight_layout(); st.pyplot(fig_t); plt.close(fig_t)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 8 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
def tab_random_forest():
    st.markdown("## 🌲 Random Forest")
    with st.expander("📖 Teoria — Random Forest", expanded=False):
        st.markdown("""
Ensemble de árvores com dois mecanismos de aleatoriedade:
1. **Bootstrap sampling:** cada árvore treina em amostra com reposição (~63% dos dados)
2. **Feature subsampling:** apenas √p features consideradas em cada divisão

A **previsão final** é o voto majoritário das N árvores.
A **importância das features** = redução média de impureza Gini em todas as árvores.

**Vantagens:** Robusto · Paralelo · Feature importance · Out-of-bag error  
**Desvantagens:** Menos interpretável que árvore simples · Mais memória
        """)
    if not _header_check("Random Forest"): return
    st.markdown("### ⚙️ Hiperparâmetros")
    c1, c2, c3, c4 = st.columns(4)
    with c1: n_est = st.slider("n_estimators", 50, 500, 200, 50, key="rf_ne")
    with c2: md    = st.slider("max_depth", 3, 20, 10, key="rf_md")
    with c3: mf    = st.selectbox("max_features", ["sqrt", "log2"], key="rf_mf")
    with c4: cw    = st.checkbox("class_weight='balanced'", True, key="rf_cw")
    if st.button("🚀 Treinar Random Forest", type="primary", key="btn_rf"):
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=md, max_features=mf,
                                      class_weight="balanced" if cw else None,
                                      n_jobs=-1, random_state=42)
        run_model(clf, "Random Forest")
    if "Random Forest" in ss.results:
        show_results("Random Forest")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 9 — MLP
# ══════════════════════════════════════════════════════════════════════════════
def tab_mlp():
    st.markdown("## 🧠 MLP — Multi-Layer Perceptron")
    with st.expander("📖 Teoria — MLP", expanded=False):
        st.markdown("""
Rede neural com camadas ocultas. Cada neurônio aplica:
```
h⁽ˡ⁾ = σ(W⁽ˡ⁾ · h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)    σ = ReLU ou tanh
```
Treinado por **backpropagation + gradient descent** minimizando a Binary Cross-Entropy.

**Vantagens:** Aprende relações não-lineares complexas · Teorem. aprox. universal  
**Desvantagens:** Muitos hiperparâmetros · Sensível à escala · Sem class_weight nativo · "Caixa preta"
        """)
    if not _header_check("MLP"): return
    st.markdown("### ⚙️ Hiperparâmetros")
    c1, c2, c3, c4 = st.columns(4)
    with c1: h1 = st.number_input("Neurônios L1", 16, 512, 128, 16, key="mlp_h1")
    with c2: h2 = st.number_input("Neurônios L2", 0, 256, 64, 16, key="mlp_h2")
    with c3: alp = st.number_input("Alpha (L2 reg)", 1e-5, 1.0, 1e-3, format="%.5f", key="mlp_a")
    with c4: mi  = st.slider("Max Iter", 100, 1000, 300, 50, key="mlp_mi")
    act = st.selectbox("Ativação", ["relu", "tanh"], key="mlp_act")
    if st.button("🚀 Treinar MLP", type="primary", key="btn_mlp"):
        hls = (int(h1), int(h2)) if h2 > 0 else (int(h1),)
        clf = MLPClassifier(hidden_layer_sizes=hls, activation=act, alpha=alp,
                            max_iter=int(mi), early_stopping=True,
                            validation_fraction=0.1, random_state=42)
        run_model(clf, "MLP")

    if "MLP" in ss.results:
        show_results("MLP")
        if "MLP" in ss.models and hasattr(ss.models["MLP"], "loss_curve_"):
            with st.expander("📉 Curva de Aprendizado (Loss)"):
                clf_m = ss.models["MLP"]
                fig_lc, ax_lc = plt.subplots(figsize=(8, 3.5))
                ax_lc.plot(clf_m.loss_curve_, color="navy", lw=2, label="Treino")
                ax_lc.set_xlabel("Época"); ax_lc.set_ylabel("Loss")
                ax_lc.set_title("MLP — Curva de Aprendizado", fontweight="bold")
                ax_lc.legend(); ax_lc.grid(alpha=0.4)
                plt.tight_layout(); st.pyplot(fig_lc); plt.close(fig_lc)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 10 — ADABOOST
# ══════════════════════════════════════════════════════════════════════════════
def tab_adaboost():
    st.markdown("## 🚀 AdaBoost")
    with st.expander("📖 Teoria — AdaBoost", expanded=False):
        st.markdown("""
Combina **aprendizes fracos** (stumps) sequencialmente, aumentando o peso de amostras
incorretamente classificadas a cada iteração:
```
F(x) = sign(Σₜ αₜhₜ(x))    αₜ = ½ ln((1−εₜ)/εₜ)
```
Aprendizes com menor erro recebem maior peso α na combinação final.

**Vantagens:** Melhora muito sobre um único aprendiz fraco · Feature importance  
**Desvantagens:** Sensível a outliers (recebem pesos muito altos) · Sequencial (não paralelizável)
        """)
    if not _header_check("AdaBoost"): return
    c1, c2, c3 = st.columns(3)
    with c1: n_est = st.slider("n_estimators", 50, 500, 200, 50, key="ada_ne")
    with c2: lr    = st.number_input("learning_rate", 0.01, 2.0, 0.5, key="ada_lr")
    with c3: md    = st.slider("Base max_depth", 1, 5, 2, key="ada_md")
    if st.button("🚀 Treinar AdaBoost", type="primary", key="btn_ada"):
        base = DecisionTreeClassifier(max_depth=int(md), class_weight="balanced")
        clf  = AdaBoostClassifier(estimator=base, n_estimators=n_est,
                                   learning_rate=lr, random_state=42)
        run_model(clf, "AdaBoost")
    if "AdaBoost" in ss.results:
        show_results("AdaBoost")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 11 — NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════════
def tab_naive_bayes():
    st.markdown("## 🎲 Naive Bayes (Gaussian)")
    with st.expander("📖 Teoria — Naive Bayes", expanded=False):
        st.markdown("""
Aplica o **Teorema de Bayes** com suposição de independência condicional entre features:
```
P(y|x₁,...,xₙ) ∝ P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```
**GaussianNB** assume que cada feature em cada classe segue distribuição Normal:
```
P(xⱼ|y=k) = (1/√(2πσ²ₖⱼ)) × exp(−(xⱼ−μₖⱼ)² / 2σ²ₖⱼ)
```
**Vantagens:** Extremamente rápido · Funciona bem com poucos dados · Alta dimensionalidade  
**Desvantagens:** Suposição de independência raramente válida · Sem class_weight nativo
        """)
    if not _header_check("Naive Bayes"): return
    vs = st.number_input("var_smoothing (estabilidade numérica)",
                          1e-12, 1e-3, 1e-9, format="%.2e", key="nb_vs")
    if st.button("🚀 Treinar Naive Bayes", type="primary", key="btn_nb"):
        clf = GaussianNB(var_smoothing=vs)
        run_model(clf, "Naive Bayes")
    if "Naive Bayes" in ss.results:
        show_results("Naive Bayes")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 12 — XGBOOST
# ══════════════════════════════════════════════════════════════════════════════
def tab_xgboost():
    st.markdown("## ⚡ XGBoost — Extreme Gradient Boosting")
    with st.expander("📖 Teoria — XGBoost", expanded=False):
        st.markdown("""
**Gradient Boosting** com otimizações extremas. Minimiza uma função objetivo com regularização:
```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)    Ω(f) = γT + ½λ||w||²
```
Cada iteração adiciona uma árvore que corrige os **resíduos** da iteração anterior.

| Algoritmo | Analogia |
|---|---|
| Random Forest | Painel de entrevistadores votando |
| AdaBoost | Cada entrevistador aprende com o anterior |
| XGBoost | Boosting com regularização + otimizações extremas |

**Vantagens:** Estado da arte · Regularização nativa · Feature importance · Muito rápido  
**Desvantagens:** Muitos hiperparâmetros · Pode overfit sem regularização adequada
        """)
    if not XGB_OK:
        st.error("❌ XGBoost não instalado. Execute: `pip install xgboost`")
        return
    if not _header_check("XGBoost"): return

    spw = round(ss.scale_pos_weight, 2)
    st.info(f"ℹ️ scale_pos_weight automático = **{spw}** (compensa desbalanceamento: neg/pos)")

    c1, c2, c3, c4 = st.columns(4)
    with c1: n_est = st.slider("n_estimators", 50, 500, 300, 50, key="xgb_ne")
    with c2: md    = st.slider("max_depth", 3, 12, 6, key="xgb_md")
    with c3: lr    = st.number_input("learning_rate", 0.01, 0.5, 0.1, key="xgb_lr")
    with c4: sub   = st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key="xgb_sub")
    col = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05, key="xgb_col")
    use_spw = st.checkbox(f"Usar scale_pos_weight={spw}", True, key="xgb_spw")

    if st.button("🚀 Treinar XGBoost", type="primary", key="btn_xgb"):
        clf = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=md, learning_rate=lr,
            subsample=sub, colsample_bytree=col,
            scale_pos_weight=spw if use_spw else 1.0,
            use_label_encoder=False, eval_metric="logloss",
            n_jobs=-1, random_state=42,
        )
        run_model(clf, "XGBoost")
    if "XGBoost" in ss.results:
        show_results("XGBoost")


# ══════════════════════════════════════════════════════════════════════════════
# ABA 13 — COMPARAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def tab_comparison():
    st.markdown("## 📊 Comparação dos Algoritmos")

    if not ss.results:
        st.info("Nenhum modelo treinado ainda. Execute os algoritmos nas abas anteriores.")
        return

    # ── Tabela Comparativa ─────────────────────────────────────────────────────
    st.markdown("### 📋 Tabela Comparativa")
    df_res = (pd.DataFrame({k: {c: v for c, v in v.items() if not c.startswith("_")}
                             for k, v in ss.results.items()})
              .T.reset_index().rename(columns={"index": "Modelo"}))

    num_cols_res = [c for c in df_res.columns if c not in ("Modelo",)]
    for col in num_cols_res:
        df_res[col] = pd.to_numeric(df_res[col], errors="coerce")

    sort_col = "AUC-PR" if "AUC-PR" in df_res.columns else "F1(+)"
    df_res = df_res.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    df_res.index = df_res.index + 1

    def hl_max(s):
        is_max = s == s.max()
        return ["background-color:#c8e6c9; font-weight:bold" if v else "" for v in is_max]

    metric_cols = [c for c in ["Acurácia","Precisão(+)","Recall(+)","F1(+)","AUC-ROC","AUC-PR"] if c in df_res.columns]
    styled = (df_res.style
              .apply(hl_max, subset=metric_cols)
              .format({c: "{:.4f}" for c in metric_cols})
              .set_caption("Verde = melhor valor por coluna · Ordenado por AUC-PR"))
    st.dataframe(styled, use_container_width=True)

    csv_res = df_res.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar tabela CSV", data=csv_res, file_name="comparacao_modelos.csv",
                       mime="text/csv")
    st.divider()

    # ── Curvas ROC ─────────────────────────────────────────────────────────────
    if ss.roc_curves:
        st.markdown("### 📈 Curvas ROC — Todos os Modelos")
        palette = sns.color_palette("tab10", n_colors=len(ss.roc_curves))
        roc_sorted = sorted(ss.roc_curves.items(), key=lambda x: -x[1][2])
        fig_roc, ax = plt.subplots(figsize=(10, 6))
        for i, (name, (fpr, tpr, auc_v)) in enumerate(roc_sorted):
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_v:.3f})", color=palette[i])
        ax.plot([0,1],[0,1], "k--", lw=1.5, label="Baseline (random)")
        ax.set_xlabel("FPR — Taxa de Falsos Positivos", fontsize=11)
        ax.set_ylabel("TPR — Taxa de Verdadeiros Positivos", fontsize=11)
        ax.set_title("Curvas ROC — Comparativo de Modelos", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1)); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_roc)
        roc_bytes = fig_to_bytes(fig_roc); plt.close(fig_roc)
        st.download_button("⬇️ Baixar ROC PNG", data=roc_bytes, file_name="roc_comparison.png", mime="image/png")
        st.divider()

    # ── Curvas PR ──────────────────────────────────────────────────────────────
    if ss.pr_curves:
        st.markdown("### 📈 Curvas Precision-Recall — Todos os Modelos")
        bl_pr = ss.y_test.mean() if ss.y_test is not None else 0.5
        pr_sorted = sorted(ss.pr_curves.items(), key=lambda x: -x[1][2])
        fig_pr, ax = plt.subplots(figsize=(10, 6))
        for i, (name, (prec, rec, ap)) in enumerate(pr_sorted):
            ax.plot(rec, prec, lw=2, label=f"{name} (AUC-PR={ap:.3f})", color=palette[i])
        ax.axhline(bl_pr, color="black", ls="--", lw=1.5, label=f"Baseline={bl_pr:.3f}")
        ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precisão", fontsize=11)
        ax.set_title("Curvas Precision-Recall — Comparativo\n(mais informativa para dados desbalanceados)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1)); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_pr)
        pr_bytes = fig_to_bytes(fig_pr); plt.close(fig_pr)
        st.download_button("⬇️ Baixar PR PNG", data=pr_bytes, file_name="pr_comparison.png", mime="image/png")
        st.divider()

    # ── Bar Charts por Métrica ──────────────────────────────────────────────────
    st.markdown("### 📊 Ranking por Métrica")
    avail_metrics = [c for c in ["AUC-PR","AUC-ROC","F1(+)","Recall(+)","Precisão(+)","Acurácia"]
                     if c in df_res.columns and df_res[c].notna().any()]

    n_m = len(avail_metrics)
    ncols_m = min(n_m, 3)
    nrows_m = (n_m + ncols_m - 1) // ncols_m
    fig_bars, axes_b = plt.subplots(nrows_m, ncols_m, figsize=(5*ncols_m, 4*nrows_m))
    axes_b = np.array(axes_b).ravel() if n_m > 1 else [axes_b]

    for i, metric in enumerate(avail_metrics):
        ax = axes_b[i]
        sub = df_res[["Modelo", metric]].dropna().sort_values(metric, ascending=True)
        colors_bar = sns.color_palette("RdYlGn", n_colors=len(sub))
        ax.barh(sub["Modelo"], sub[metric], color=colors_bar, alpha=0.88, edgecolor="white")
        for idx2, (_, row) in enumerate(sub.iterrows()):
            ax.text(row[metric] + sub[metric].max()*0.01, idx2,
                    f"{row[metric]:.3f}", va="center", fontsize=8, fontweight="bold")
        ax.set_title(metric, fontweight="bold", fontsize=11)
        ax.set_xlim(0, sub[metric].max() * 1.15)

    for j in range(len(avail_metrics), len(axes_b)):
        axes_b[j].set_visible(False)

    plt.suptitle("Ranking por Métrica (verde = melhor)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_bars)
    bars_bytes = fig_to_bytes(fig_bars); plt.close(fig_bars)
    st.download_button("⬇️ Baixar ranking PNG", data=bars_bytes,
                       file_name="ranking_metricas.png", mime="image/png")
    st.divider()

    # ── Teoria das Métricas ─────────────────────────────────────────────────────
    st.markdown("### 📚 Teoria das Métricas de Avaliação")

    with st.expander("📖 Acurácia (Accuracy)"):
        st.markdown("""
**Fórmula:** `(TP + TN) / (TP + TN + FP + FN)`

Mede a proporção de previsões corretas sobre o total.

**⚠️ Problema:** Enganosa em datasets desbalanceados. Se 90% são da classe 0, um modelo
que prevê SEMPRE 0 teria 90% de acurácia — mas seria inútil na prática.

**Quando usar:** Apenas quando as classes são equilibradas (perto de 50/50).
        """)

    with st.expander("📖 Precisão — Precision"):
        st.markdown("""
**Fórmula:** `TP / (TP + FP)`

Dos que o modelo previu como **positivos**, qual fração realmente era positiva?

**Interpretação no negócio:** "Quantos dos clientes que eu liguei realmente converteram?"
- Alta Precisão = poucos falsos alarmes = eficiência operacional

**Quando priorizar:** quando o custo de **falso positivo é alto**
(ex: ligar para muitos clientes errados = custo operacional)
        """)

    with st.expander("📖 Recall (Sensibilidade)"):
        st.markdown("""
**Fórmula:** `TP / (TP + FN)`

Dos que **realmente eram positivos**, qual fração o modelo detectou?

**Interpretação no negócio:** "Quantos dos clientes que realmente converteriam eu identifiquei?"
- Alto Recall = poucos clientes perdidos = maximiza receita

**Quando priorizar:** quando o custo de **falso negativo é alto**
(ex: deixar de contatar clientes que converteriam = receita perdida)
        """)

    with st.expander("📖 F1-Score"):
        st.markdown("""
**Fórmula:** `2 × (Precisão × Recall) / (Precisão + Recall)`

**Média harmônica** entre Precisão e Recall. Penaliza fortemente quando um dos dois é muito baixo.

**Quando usar:** quando você precisa equilibrar Precisão e Recall e o dataset é desbalanceado.
F1 é mais informativo que acurácia em problemas binários desbalanceados.
        """)

    with st.expander("📖 AUC-ROC — Área sob a Curva ROC"):
        st.markdown("""
**Curva ROC:** plota TPR (Recall) vs. FPR para todos os thresholds de classificação.

**AUC-ROC:** área sob essa curva. Varia de 0.5 (aleatório) a 1.0 (perfeito).

**Interpretação:** probabilidade de que, dado um positivo e um negativo aleatórios, o modelo
atribua score maior ao positivo.

**⚠️ Limitação:** pode ser otimista em datasets muito desbalanceados — use AUC-PR como complemento.
        """)

    with st.expander("📖 AUC-PR — Área sob a Curva Precision-Recall ⭐ Principal para dados desbalanceados"):
        st.markdown("""
**Curva PR:** plota Precisão vs. Recall para todos os thresholds.

**AUC-PR:** área sob essa curva. **Mais informativa que AUC-ROC** quando as classes são desbalanceadas.

**Baseline:** um modelo sem informação teria AUC-PR ≈ prevalência da classe positiva.
Se a classe positiva é 11,3% dos dados, o baseline é ≈ 0,113.

**Por que é superior:** enquanto a curva ROC usa TN (que são abundantes em datasets
desbalanceados), a curva PR foca apenas nas classes positivas — onde o desafio real está.

> ⭐ **Recomendação:** use **AUC-PR como métrica principal** para datasets com desbalanceamento
> de classes ≥ 3:1.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    tabs = st.tabs([
        "📁 Upload & Dados",
        "📐 PCA & Clusters",
        "📊 Log. Reg.",
        "🔵 KNN",
        "⚡ SVM Linear",
        "🌀 SVM RBF",
        "🌳 Árvore",
        "🌲 Rand. Forest",
        "🧠 MLP",
        "🚀 AdaBoost",
        "🎲 Naive Bayes",
        "⚡ XGBoost",
        "📊 Comparação",
    ])

    with tabs[0]:  tab_upload()
    with tabs[1]:  tab_pca()
    with tabs[2]:  tab_logistic()
    with tabs[3]:  tab_knn()
    with tabs[4]:  tab_svm_linear()
    with tabs[5]:  tab_svm_rbf()
    with tabs[6]:  tab_decision_tree()
    with tabs[7]:  tab_random_forest()
    with tabs[8]:  tab_mlp()
    with tabs[9]:  tab_adaboost()
    with tabs[10]: tab_naive_bayes()
    with tabs[11]: tab_xgboost()
    with tabs[12]: tab_comparison()


if __name__ == "__main__":
    main()
