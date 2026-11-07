from __future__ import annotations
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def make_pdf(title: str, sections: dict[str, object]) -> bytes:
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    y = height - 50
    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, title)
    y -= 30
    pdf.setFont("Helvetica", 9)
    for heading, value in sections.items():
        if y < 60:
            pdf.showPage()
            y = height - 50
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y, str(heading))
        y -= 16
        pdf.setFont("Helvetica", 9)
        for line in str(value).splitlines() or [""]:
            pdf.drawString(50, y, line[:110])
            y -= 13
        y -= 8
    pdf.save()
    return output.getvalue()
