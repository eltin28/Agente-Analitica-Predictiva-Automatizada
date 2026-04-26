# pipeline/reporting.py

import io
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ─────────────────────────────────────────────
# PALETA Y ESTILOS
# ─────────────────────────────────────────────

BRAND_DARK   = colors.HexColor("#1B2A4A")   # azul marino
BRAND_MID    = colors.HexColor("#2E5FA3")   # azul medio
BRAND_LIGHT  = colors.HexColor("#EBF1FB")   # azul muy claro
ACCENT       = colors.HexColor("#E8622A")   # naranja acento
GRAY_LIGHT   = colors.HexColor("#F4F6F9")
GRAY_MID     = colors.HexColor("#C8CDD6")
TEXT_DARK    = colors.HexColor("#1A1A2E")
TEXT_MUTED   = colors.HexColor("#6B7280")

MPL_BLUE     = "#2E5FA3"
MPL_ORANGE   = "#E8622A"
MPL_LIGHT    = "#EBF1FB"
MPL_TEXT_DARK = "#1A1A2E"
MPL_GRAY_MID  = "#C8CDD6"


def _build_styles():
    base = getSampleStyleSheet()

    custom = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=26,
            textColor=colors.white,
            spaceAfter=4,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
        ),
        "ReportSubtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#C8D8F5"),
            spaceAfter=0,
            fontName="Helvetica",
            alignment=TA_LEFT,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontSize=13,
            textColor=BRAND_DARK,
            spaceBefore=18,
            spaceAfter=6,
            fontName="Helvetica-Bold",
            borderPad=0,
        ),
        "BodyText": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=9.5,
            textColor=TEXT_DARK,
            leading=15,
            spaceAfter=6,
            fontName="Helvetica",
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=8,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
            spaceAfter=10,
            fontName="Helvetica-Oblique",
        ),
        "MetricLabel": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontSize=8,
            textColor=TEXT_MUTED,
            fontName="Helvetica",
            alignment=TA_CENTER,
        ),
        "MetricValue": ParagraphStyle(
            "MetricValue",
            parent=base["Normal"],
            fontSize=18,
            textColor=BRAND_DARK,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
        "Footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
            fontName="Helvetica",
        ),
        "LimeCode": ParagraphStyle(
            "LimeCode",
            parent=base["Code"],
            fontSize=8.5,
            textColor=TEXT_DARK,
            backColor=GRAY_LIGHT,
            leading=14,
            leftIndent=10,
            fontName="Courier",
        ),
    }
    return base, custom


# ─────────────────────────────────────────────
# CABECERA CON FONDO DE COLOR
# ─────────────────────────────────────────────

def _header_block(styles, target, best_model, problem_type, elapsed):
    _, S = styles

    date_str = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    problem_label = "Clasificación" if problem_type == "classification" else "Regresión"

    # Tabla de cabecera: fondo azul marino, dos columnas
    meta_rows = [
        ["Target:", target],
        ["Modelo:", best_model],
        ["Tipo:", problem_label],
        ["Duración:", f"{elapsed}s"],
        ["Fecha:", date_str],
    ]

    meta_style = ParagraphStyle(
        "MetaKey", fontSize=8, textColor=colors.HexColor("#C8D8F5"),
        fontName="Helvetica-Bold"
    )
    meta_val_style = ParagraphStyle(
        "MetaVal", fontSize=8, textColor=colors.white, fontName="Helvetica"
    )

    meta_cells = [
        [Paragraph(k, meta_style), Paragraph(v, meta_val_style)]
        for k, v in meta_rows
    ]

    meta_table = Table(meta_cells, colWidths=[2.8*cm, 7*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    title_cell = [
        [Paragraph("Reporte Ejecutivo de Análisis de Datos", S["ReportTitle"])],
        [Paragraph("Generado automáticamente por el Agente de ML", S["ReportSubtitle"])],
        [Spacer(1, 10)],
        [meta_table],
    ]

    title_inner = Table(title_cell, colWidths=[12*cm])
    title_inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
    ]))

    # Barra de acento naranja a la derecha
    accent_bar = Table([[""]], colWidths=[0.5*cm], rowHeights=[5.5*cm])
    accent_bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
    ]))

    outer = Table([[title_inner, accent_bar]], colWidths=[14.5*cm, 0.5*cm])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
    ]))

    return [outer, Spacer(1, 16)]


# ─────────────────────────────────────────────
# TARJETAS DE MÉTRICAS PRINCIPALES
# ─────────────────────────────────────────────

def _metric_cards(styles, metrics_df, problem_type, best_model=None):
    _, S = styles

    if metrics_df.empty:
        return []

    # Buscar la fila del best_model por nombre (con o sin el prefijo ★)
    if best_model:
        mask = metrics_df["model"].str.contains(best_model, regex=False, na=False)
        row = metrics_df[mask].iloc[0] if mask.any() else metrics_df.iloc[0]
    else:
        row = metrics_df.iloc[0]

    if problem_type == "classification":
        cards = [
            ("Accuracy", f"{row.get('accuracy', 0):.1%}"),
            ("F1-Score", f"{row.get('f1', 0):.4f}"),
            ("Modelo", str(row.get("model", "—"))),
        ]
    else:
        cards = [
            ("R²", f"{row.get('r2', 0):.4f}"),
            ("RMSE", f"{row.get('rmse', 0):.4f}"),
            ("MAE", f"{row.get('mae', 0):.4f}"),
        ]

    cells = []
    for label, value in cards:
        cell = Table(
            [
                [Paragraph(value, S["MetricValue"])],
                [Paragraph(label, S["MetricLabel"])],
            ],
            colWidths=[4.5*cm],
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("ROUNDEDCORNERS", [4]),
            ("BOX", (0, 0), (-1, -1), 1, BRAND_MID),
        ]))
        cells.append(cell)

    row_table = Table([cells], colWidths=[4.5*cm, 4.5*cm, 4.5*cm],
                      hAlign="LEFT")
    row_table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    return [row_table, Spacer(1, 12)]


# ─────────────────────────────────────────────
# TABLA DE MÉTRICAS COMPLETA
# ─────────────────────────────────────────────

def _metrics_table(styles, metrics_df):
    _, S = styles

    if metrics_df.empty:
        return [Paragraph("No hay métricas disponibles.", S["BodyText"])]

    cols = list(metrics_df.columns)
    header = [Paragraph(f"<b>{c}</b>", S["BodyText"]) for c in cols]
    data = [header]

    for i, (_, row) in enumerate(metrics_df.iterrows()):
        data.append([
            Paragraph(
                str(round(row[c], 4)) if isinstance(row[c], float) else str(row[c]),
                S["BodyText"]
            )
            for c in cols
        ])

    col_w = 15 * cm / len(cols)
    table = Table(data, colWidths=[col_w] * len(cols), repeatRows=1)

    row_colors = []
    for i in range(1, len(data)):
        bg = GRAY_LIGHT if i % 2 == 0 else colors.white
        row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_MID),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, GRAY_MID),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY_LIGHT]),
    ] + row_colors))

    return [table, Spacer(1, 6)]


# ─────────────────────────────────────────────
# GRÁFICO DE BARRAS — COMPARACIÓN DE MODELOS
# ─────────────────────────────────────────────

def _models_chart(metrics_df, problem_type, best_model=None):
    """Genera imagen PNG en memoria con la comparación de modelos."""
    if metrics_df.empty or len(metrics_df) < 2:
        return None

    metric_col = "f1" if problem_type == "classification" else "r2"
    if metric_col not in metrics_df.columns:
        return None

    # Excluir fila "(final)" duplicada generada por run_analysis
    df = metrics_df[~metrics_df["model"].str.contains("final", na=False)]
    df = df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=True)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(df) * 0.55)))
    fig.patch.set_facecolor("white")

    # Colorear por nombre real del best_model, no por posición en ranking
    bar_colors = [
        MPL_ORANGE if (best_model and best_model in str(m)) else MPL_BLUE
        for m in df["model"]
    ]
    bars = ax.barh(df["model"], df[metric_col], color=bar_colors,
                   height=0.55, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df[metric_col]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8,
                color=MPL_TEXT_DARK, fontweight="bold")

    ax.set_xlabel("F1-Score (weighted)" if metric_col == "f1" else "R²",
                  fontsize=8.5, color="#6B7280")
    ax.set_xlim(0, df[metric_col].max() * 1.18)
    ax.set_facecolor("white")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5, colors=MPL_TEXT_DARK)
    ax.tick_params(axis="x", labelsize=7.5, colors="#6B7280")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color=MPL_GRAY_MID)

    best_patch = mpatches.Patch(color=MPL_ORANGE, label="Mejor modelo")
    rest_patch = mpatches.Patch(color=MPL_BLUE, label="Otros modelos")
    ax.legend(handles=[best_patch, rest_patch], fontsize=7.5,
              framealpha=0.5, loc="lower right")

    plt.tight_layout(pad=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# GRÁFICO SHAP
# ─────────────────────────────────────────────

def _shap_chart(shap_importance, top_n=12):
    """Gráfico horizontal de importancia SHAP (top N features)."""
    if not shap_importance:
        return None

    df = pd.DataFrame(shap_importance).head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(df) * 0.5)))
    fig.patch.set_facecolor("white")

    norm = plt.Normalize(df["importance"].min(), df["importance"].max())
    palette = plt.cm.Blues(norm(df["importance"].values))

    bars = ax.barh(df["feature"], df["importance"], color=palette,
                   height=0.55, edgecolor="white")

    for bar, val in zip(bars, df["importance"]):
        ax.text(bar.get_width() + df["importance"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7.5,
                color=MPL_TEXT_DARK)

    ax.set_xlabel("Importancia media |SHAP|", fontsize=8.5, color="#6B7280")
    ax.set_facecolor("white")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(pad=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# SECCIÓN LIME
# ─────────────────────────────────────────────

def _lime_section(styles, lime_text, probabilities):
    _, S = styles
    elements = []

    if not lime_text:
        return [Paragraph("LIME no disponible.", S["BodyText"])]

    elements.append(Paragraph(
        "La siguiente tabla muestra las features que más influyeron en la predicción "
        "de la instancia de ejemplo. Valores positivos aumentan la probabilidad de la "
        "clase positiva; negativos la reducen.",
        S["BodyText"]
    ))
    elements.append(Spacer(1, 6))

    # Parsear líneas LIME
    rows = []
    for line in lime_text.strip().split("\n"):
        if ":" in line:
            parts = line.rsplit(":", 1)
            feature = parts[0].strip()
            try:
                weight = float(parts[1].strip())
            except ValueError:
                weight = 0.0
            rows.append((feature, weight))

    if rows:
        header = [
            Paragraph("<b>Feature / Condición</b>", S["BodyText"]),
            Paragraph("<b>Peso</b>", S["BodyText"]),
            Paragraph("<b>Dirección</b>", S["BodyText"]),
        ]
        data = [header]
        for feat, w in rows:
            direction = Paragraph(
                "<font color='#2E7D32'>&#9650; Aumenta</font>" if w > 0
                else "<font color='#C62828'>&#9660; Reduce</font>",
                S["BodyText"]
            )
            data.append([
                Paragraph(feat, S["BodyText"]),
                Paragraph(f"{w:+.4f}", S["BodyText"]),
                direction,
            ])

        lime_table = Table(data, colWidths=[8.5*cm, 3*cm, 3.5*cm])
        lime_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_MID),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, GRAY_MID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY_LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(lime_table)
        elements.append(Spacer(1, 8))

    # Probabilidades
    if probabilities:
        elements.append(Paragraph(
            f"<b>Probabilidades predichas:</b> "
            f"Clase 0 = {probabilities[0]:.1%}  |  "
            f"Clase 1 = {probabilities[1]:.1%}",
            S["BodyText"]
        ))

    return elements


# ─────────────────────────────────────────────
# SECCIÓN PREPROCESAMIENTO
# ─────────────────────────────────────────────

def _preprocessing_section(styles, preprocessing):
    _, S = styles

    if not preprocessing:
        return []

    numeric  = preprocessing.get("numeric_features", [])
    nominal  = preprocessing.get("nominal_features", [])
    ordinal  = preprocessing.get("ordinal_features", [])
    dropped  = (
        preprocessing.get("dropped_high_missing", [])
        + preprocessing.get("dropped_high_cardinality", [])
        + preprocessing.get("dropped_id_like", [])
    )

    total_in  = preprocessing.get("total_input_features", "—")
    total_use = preprocessing.get("total_used_features", "—")

    summary = [
        ["Features entrada", str(total_in)],
        ["Features usadas", str(total_use)],
        ["Numéricas", str(len(numeric))],
        ["Nominales (OHE)", str(len(nominal))],
        ["Ordinales", str(len(ordinal))],
        ["Descartadas", str(len(dropped))],
    ]

    label_style = ParagraphStyle("pl", fontSize=8.5, fontName="Helvetica-Bold",
                                 textColor=BRAND_DARK)
    val_style   = ParagraphStyle("pv", fontSize=8.5, fontName="Helvetica",
                                 textColor=TEXT_DARK, alignment=TA_RIGHT)

    summary_data = [
        [Paragraph(k, label_style), Paragraph(v, val_style)]
        for k, v in summary
    ]
    summary_table = Table(summary_data, colWidths=[5*cm, 3*cm])
    summary_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BRAND_LIGHT, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.5, GRAY_MID),
    ]))

    def col_list(cols, color):
        if not cols:
            return Paragraph("<i>—</i>", S["BodyText"])
        items = ", ".join(f'<font color="{color}"><b>{c}</b></font>' for c in cols)
        return Paragraph(items, S["BodyText"])

    detail_data = [
        [Paragraph("<b>Grupo</b>", S["BodyText"]),
         Paragraph("<b>Columnas</b>", S["BodyText"])],
        [Paragraph("Numéricas", S["BodyText"]),
         col_list(numeric, "#1B5E20")],
        [Paragraph("Nominales", S["BodyText"]),
         col_list(nominal, "#1A237E")],
        [Paragraph("Ordinales", S["BodyText"]),
         col_list(ordinal, "#E65100")],
        [Paragraph("Descartadas", S["BodyText"]),
         col_list(dropped, "#B71C1C")],
    ]

    detail_table = Table(detail_data, colWidths=[3*cm, 12*cm])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_MID),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, GRAY_MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    return [summary_table, Spacer(1, 10), detail_table]


# ─────────────────────────────────────────────
# DIVISOR DE SECCIÓN
# ─────────────────────────────────────────────

def _section(title, styles, elements):
    _, S = styles
    return [
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.5, color=BRAND_MID, spaceAfter=4),
        Paragraph(title, S["SectionHeading"]),
        *elements,
    ]


# ─────────────────────────────────────────────
# PÁGINA FINAL — CANVAS (pie de página)
# ─────────────────────────────────────────────

def _footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(BRAND_MID)
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.5*cm, w - 2*cm, 1.5*cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawCentredString(
        w / 2, 1.1*cm,
        f"Agente de Análisis Automático  —  Página {doc.page}  —  "
        f"{datetime.datetime.now().strftime('%d/%m/%Y')}"
    )
    canvas.restoreState()


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def generate_pdf_report(
    output_path: str,
    target: str,
    best_model: str,
    metrics_df,
    lime_text,
    problem_type: str = "classification",
    shap_importance: list = None,
    preprocessing: dict = None,
    elapsed_seconds: float = None,
    probabilities: list = None,
):
    elapsed = f"{elapsed_seconds:.1f}" if elapsed_seconds else "—"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=1.5*cm,
        bottomMargin=2.2*cm,
    )

    styles = _build_styles()
    _, S = styles
    story = []

    # ── CABECERA ──────────────────────────────
    story += _header_block(styles, target, best_model, problem_type, elapsed)

    # ── TARJETAS DE MÉTRICAS ──────────────────
    story += _section("Resumen de Rendimiento", styles,
                      _metric_cards(styles, metrics_df, problem_type, best_model=best_model))

    # ── TABLA COMPLETA ────────────────────────
    story += _section("Comparación de Modelos — Test Set", styles,
                      _metrics_table(styles, metrics_df))

    # ── GRÁFICO DE MODELOS ────────────────────
    # Filtrar fila final para calcular altura real
    _df_chart = metrics_df[~metrics_df["model"].str.contains("final", na=False)]
    chart_buf = _models_chart(metrics_df, problem_type, best_model=best_model)
    if chart_buf:
        chart_h = max(3.5*cm, len(_df_chart) * 1.1*cm)
        img = Image(chart_buf, width=14*cm, height=chart_h)
        caption = Paragraph(
            "Figura 1. Comparación de modelos por F1-Score ponderado en test set. "
            "El modelo destacado en naranja es el seleccionado como final.",
            S["Caption"]
        )
        # KeepTogether evita que el gráfico se parta entre páginas
        chart_block = KeepTogether([img, Spacer(1, 4), caption])
        story += _section("Comparación Visual", styles, [chart_block])

    # ── SHAP ──────────────────────────────────
    shap_buf = _shap_chart(shap_importance) if shap_importance else None
    shap_elements = []

    if shap_buf:
        top_n = min(12, len(shap_importance))
        shap_img = Image(shap_buf, width=13*cm, height=max(3*cm, top_n*0.7*cm))
        shap_elements += [
            Paragraph(
                "SHAP (SHapley Additive exPlanations) mide la contribución promedio "
                "de cada feature al resultado del modelo a nivel global.",
                S["BodyText"]
            ),
            Spacer(1, 6),
            shap_img,
            Paragraph(
                f"Figura 2. Top {top_n} features por importancia SHAP media absoluta.",
                S["Caption"]
            ),
        ]
    else:
        shap_elements.append(Paragraph("SHAP no disponible.", S["BodyText"]))

    story += _section("Importancia Global de Features — SHAP", styles, shap_elements)

    # ── LIME ──────────────────────────────────
    story += _section("Explicación Local — LIME", styles,
                      _lime_section(styles, lime_text, probabilities))

    # ── PREPROCESAMIENTO ──────────────────────
    if preprocessing:
        story += _section("Preprocesamiento Aplicado", styles,
                          _preprocessing_section(styles, preprocessing))

    # ── CONCLUSIÓN ────────────────────────────
    problem_label = "clasificación" if problem_type == "classification" else "regresión"
    conclusion_text = (
        f"El agente evaluó todos los modelos disponibles y seleccionó <b>{best_model}</b> "
        f"como el mejor para el problema de <b>{problem_label}</b> sobre la variable "
        f"<b>{target}</b>, en base al equilibrio entre desempeño medio y estabilidad "
        f"entre folds (métrica robusta = media − desviación estándar). "
        f"Se recomienda validar el modelo con nuevos datos antes de desplegarlo en producción."
    )
    story += _section("Conclusiones", styles, [
        Paragraph(conclusion_text, S["BodyText"])
    ])

    # ── BUILD ─────────────────────────────────
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)