import io
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.domain import Attendance, Person, Camera
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

async def get_filtered_attendance(
    db: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    person_id: Optional[int] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
    camera_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetches attendance records joined with person and camera details based on filters."""
    stmt = (
        select(Attendance, Person, Camera)
        .join(Person, Attendance.person_id == Person.id)
        .outerjoin(Camera, Attendance.camera_id == Camera.id)
    )

    conditions = []
    if start_date:
        conditions.append(Attendance.attendance_date >= start_date)
    if end_date:
        conditions.append(Attendance.attendance_date <= end_date)
    if person_id:
        conditions.append(Attendance.person_id == person_id)
    if department:
        conditions.append(Person.department.ilike(f"%{department}%"))
    if status:
        conditions.append(Attendance.status == status.upper())
    if camera_id:
        conditions.append(Attendance.camera_id == camera_id)

    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.order_by(Attendance.attendance_date.desc(), Attendance.first_seen_time.desc())
    result = await db.execute(stmt)
    rows = result.all()

    records = []
    for att, p, c in rows:
        records.append({
            "id": att.id,
            "person_id": p.id,
            "unique_person_id": p.unique_person_id,
            "person_name": p.name,
            "roll_number": p.roll_number,
            "employee_number": p.employee_number,
            "department": p.department,
            "attendance_date": str(att.attendance_date),
            "first_seen_time": att.first_seen_time.strftime("%H:%M:%S") if att.first_seen_time else "",
            "status": att.status,
            "confidence": round(att.confidence, 4) if att.confidence else None,
            "camera_id": c.id if c else None,
            "camera_name": c.camera_name if c else "System/Cutoff",
            "snapshot_path": att.snapshot_path
        })

    return records

def export_attendance_csv(records: List[Dict[str, Any]]) -> str:
    """Exports attendance records to CSV string."""
    df = pd.DataFrame(records)
    if df.empty:
        return "ID,Unique Person ID,Name,Department,Date,First Seen,Status,Camera\n"
    
    cols = ["id", "unique_person_id", "person_name", "department", "attendance_date", "first_seen_time", "status", "camera_name"]
    available = [c for c in cols if c in df.columns]
    return df[available].to_csv(index=False)

def export_attendance_excel(records: List[Dict[str, Any]]) -> bytes:
    """Exports attendance records to Excel bytes."""
    df = pd.DataFrame(records)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if df.empty:
            pd.DataFrame(columns=["ID", "Name", "Department", "Date", "Status"]).to_excel(writer, index=False)
        else:
            df.to_excel(writer, index=False, sheet_name="Attendance_Report")
    return output.getvalue()

def export_attendance_pdf(records: List[Dict[str, Any]]) -> bytes:
    """Exports attendance records to PDF report bytes."""
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("<b>CCTV Automated Attendance Report</b>", styles['Heading1'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Table Header
    data = [["Date", "ID", "Name", "Dept", "Time", "Status"]]
    for r in records[:500]:  # Cap at 500 rows for PDF rendering efficiency
        data.append([
            r.get("attendance_date", ""),
            r.get("unique_person_id", ""),
            r.get("person_name", ""),
            r.get("department", "") or "-",
            r.get("first_seen_time", ""),
            r.get("status", "")
        ])

    table = Table(data, colWidths=[80, 80, 150, 100, 70, 70])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    
    story.append(table)
    doc.build(story)
    return output.getvalue()
