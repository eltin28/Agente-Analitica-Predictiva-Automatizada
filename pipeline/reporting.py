# pipeline/reporting.py

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ─────────────────────────────────────────────
# TEXTO
# ─────────────────────────────────────────────

def generate_executive_summary(target, best_model, problem_type):
    return f"""
    Este análisis tuvo como objetivo predecir la variable <b>{target}</b> utilizando técnicas de Machine Learning.
    Se evaluaron múltiples modelos y el mejor desempeño fue obtenido por el modelo <b>{best_model}</b>.

    Tipo de problema detectado: <b>{problem_type}</b>.
    """


def generate_conclusions(best_model, lime_text):
    return f"""
    El modelo seleccionado (<b>{best_model}</b>) presenta el mejor equilibrio entre desempeño y capacidad de generalización.

    <b>Interpretación del modelo:</b><br/>
    {lime_text.replace('\n', '<br/>')}

    Se recomienda utilizar este modelo como apoyo en la toma de decisiones.
    """


# ─────────────────────────────────────────────
# TABLA DE MÉTRICAS (PRO)
# ─────────────────────────────────────────────

def build_metrics_table(metrics_df):

    if metrics_df.empty:
        return None

    columns = list(metrics_df.columns)

    # Header
    data = [columns]

    # Rows
    for _, row in metrics_df.iterrows():
        data.append([
            str(row.get(col, ""))
            if not isinstance(row.get(col), float)
            else round(row.get(col), 4)
            for col in columns
        ])

    table = Table(data, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))

    return table


# ─────────────────────────────────────────────
# PDF PRINCIPAL
# ─────────────────────────────────────────────

def generate_pdf_report(
    output_path,
    target,
    best_model,
    metrics_df,
    lime_text,
    problem_type="classification",
):

    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()

    content = []

    # ── Título ─────────────────────────────
    content.append(Paragraph("Reporte Ejecutivo de Análisis de Datos", styles["Title"]))
    content.append(Spacer(1, 20))

    # ── Resumen ────────────────────────────
    summary = generate_executive_summary(target, best_model, problem_type)
    content.append(Paragraph("Resumen Ejecutivo", styles["Heading2"]))
    content.append(Paragraph(summary, styles["Normal"]))
    content.append(Spacer(1, 20))

    # ── Resultados ─────────────────────────
    content.append(Paragraph("Resultados de Modelos", styles["Heading2"]))

    table = build_metrics_table(metrics_df)
    if table:
        content.append(table)
    else:
        content.append(Paragraph("No hay métricas disponibles.", styles["Normal"]))

    content.append(Spacer(1, 20))

    # ── Conclusiones ───────────────────────
    conclusions = generate_conclusions(best_model, lime_text)
    content.append(Paragraph("Conclusiones", styles["Heading2"]))
    content.append(Paragraph(conclusions, styles["Normal"]))

    doc.build(content)