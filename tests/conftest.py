import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

PAGE1_LINES = [
    "Employee Benefits Overview.",
    "The company offers a tuition reimbursement program for eligible employees.",
    "Employees may receive up to 5,000 dollars per year for approved courses and certifications.",
    "Reimbursement requires a grade of B or better.",
    "Health insurance is provided through a national carrier.",
]

PAGE2_LINES = [
    "Paid Time Off Policy.",
    "Full time employees accrue 15 days of paid time off per year.",
    "Unused days roll over up to a maximum of 10 days.",
    "Requests must be submitted two weeks in advance.",
]


def _write_pdf(path, pages):
    c = canvas.Canvas(str(path), pagesize=LETTER)
    for lines in pages:
        y = 720
        for line in lines:
            c.drawString(72, y, line)
            y -= 18
        c.showPage()
    c.save()
    return str(path)


@pytest.fixture
def sample_pdf(tmp_path):
    return _write_pdf(tmp_path / "benefits.pdf", [PAGE1_LINES, PAGE2_LINES])


@pytest.fixture
def two_pdfs(tmp_path):
    a = _write_pdf(tmp_path / "benefits.pdf", [PAGE1_LINES, PAGE2_LINES])
    b = _write_pdf(
        tmp_path / "safety.pdf",
        [["Workplace Safety Manual.", "Hard hats are required on the factory floor at all times."]],
    )
    return a, b
