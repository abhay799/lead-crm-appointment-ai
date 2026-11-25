
import os
import random
import re
import time
from datetime import datetime
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Lead-to-CRM-to-Appointment AI Automation")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = list(dict.fromkeys([PRIMARY_MODEL, "gemini-3.8-flash-lite"]))

client = genai.Client(api_key=GEMINI_API_KEY)

# --- IN-MEMORY CRM & CALENDAR DATABASE (ZERO-DB INSTANT DEMO) ---
CRM_LEADS_DB: list[dict[str, Any]] = [
    {
        "lead_id": "LEAD-101",
        "full_name": "Vikramaditya Rao",
        "phone": "9845011223",
        "service_interest": "3BHK Turnkey Home Interiors (Whitefield)",
        "budget_inr": "₹18,00,000",
        "timeline": "Within 30 days",
        "score": 88,
        "tier": "HOT",
        "stage": "Qualified",
        "assigned_to": "Rohan (Senior Design Consultant)",
        "notes": "Possession next month; wants modular kitchen + wardrobes.",
        "updated_at": "Today, 10:15 AM",
    },
    {
        "lead_id": "LEAD-102",
        "full_name": "Sneha Deshmukh",
        "phone": "9820199887",
        "service_interest": "4BHK Villa Renovation (Indiranagar)",
        "budget_inr": "₹32,00,000",
        "timeline": "Immediate",
        "score": 94,
        "tier": "HOT",
        "stage": "Proposal Sent",
        "assigned_to": "Ananya (Principal Architect)",
        "notes": "3D walkthrough completed; awaiting final BOQ sign-off.",
        "updated_at": "Yesterday, 6:40 PM",
    },
]

APPOINTMENTS_DB: list[dict[str, Any]] = [
    {
        "booking_id": "APT-501",
        "lead_id": "LEAD-102",
        "customer_name": "Sneha Deshmukh",
        "phone": "9820199887",
        "slot": "Tomorrow 2:00 PM",
        "visit_mode": "Experience Center Visit (Indiranagar)",
        "status": "Confirmed",
        "created_at": "Yesterday, 6:45 PM",
    }
]

AVAILABLE_SLOTS: list[str] = [
    "Tomorrow 11:00 AM",
    "Tomorrow 2:00 PM",
    "Tomorrow 4:30 PM",
    "Saturday 11:30 AM",
    "Saturday 3:30 PM",
    "Monday 12:00 PM",
]

CHAT_SESSIONS: dict[str, list[types.Content]] = {}
PENDING_BOOKING: dict[str, dict[str, str]] = {}
_current_request_tools: list[str] = []


def _compute_lead_score(budget_text: str, timeline_text: str) -> tuple[int, str]:
    digits = re.findall(r"\d+", budget_text.replace(",", ""))
    num = int(digits[0]) if digits else 5
    lower_b = budget_text.lower()
    if "cr" in lower_b or "crore" in lower_b:
        lakhs = num * 100
    elif "l" in lower_b or "lakh" in lower_b or num < 100:
        lakhs = num
    else:
        lakhs = max(1, num // 100000)

    score = 55
    if lakhs >= 15:
        score += 25
    elif lakhs >= 8:
        score += 18
    else:
        score += 10

    lower_t = timeline_text.lower()
    if any(k in lower_t for k in ("immediate", "this week", "urgent", "now", "15 days", "30 days", "1 month")):
        score += 15
    elif any(k in lower_t for k in ("2 month", "3 month", "next month")):
        score += 8

    score = min(98, score)
    tier = "HOT" if score >= 80 else ("WARM" if score >= 65 else "NURTURE")
    return score, tier


# --- UNIFIED AGENT TOOLS (#3 LEAD GEN + #4 CRM + #5 APPOINTMENT) ---
def qualify_and_score_lead(
    full_name: str,
    phone: str,
    service_interest: str,
    budget_inr: str,
    timeline: str = "Within 30 days",
) -> dict[str, Any]:
    """Qualify a new prospect, calculate their AI lead score (0-100), and auto-create a CRM record.

    Args:
        full_name: Prospect's full name.
        phone: 10-digit Indian mobile/WhatsApp number.
        service_interest: Project or service required (e.g., 3BHK Interiors, Commercial Office).
        budget_inr: Estimated budget in INR (e.g., ₹15 Lakhs or ₹15,00,000).
        timeline: Target start timeline (e.g., Immediate, Within 30 days, 2-3 months).
    """
    _current_request_tools.append(f"qualify_and_score_lead({full_name})")
    clean_phone = re.sub(r"\D", "", phone)[-10:] or phone.strip()

    for existing in CRM_LEADS_DB:
        if existing["phone"] == clean_phone or existing["full_name"].lower() == full_name.strip().lower():
            existing["service_interest"] = service_interest.strip()
            existing["budget_inr"] = budget_inr.strip()
            existing["timeline"] = timeline.strip()
            existing["updated_at"] = datetime.now().strftime("%I:%M %p")
            return {"status": "updated_existing", **existing}

    score, tier = _compute_lead_score(budget_inr, timeline)
    lead_id = f"LEAD-{random.randint(103, 999)}"
    record = {
        "lead_id": lead_id,
        "full_name": full_name.strip(),
        "phone": clean_phone,
        "service_interest": service_interest.strip(),
        "budget_inr": budget_inr.strip(),
        "timeline": timeline.strip(),
        "score": score,
        "tier": tier,
        "stage": "Qualified",
        "assigned_to": "Rohan (Senior Consultant)" if tier == "HOT" else "Meera (Inside Sales)",
        "notes": f"Auto-qualified via AI chat ({timeline.strip()}).",
        "updated_at": datetime.now().strftime("%I:%M %p"),
    }
    CRM_LEADS_DB.insert(0, record)
    return {"status": "created", **record}


def update_crm_pipeline(
    lead_identifier: str,
    new_stage: str,
    notes: str,
    budget_inr: str = "",
) -> dict[str, Any]:
    """Update an existing lead's CRM pipeline stage, deal value, and activity notes.

    Args:
        lead_identifier: Lead ID (e.g., LEAD-101) or customer's full name / phone number.
        new_stage: Target CRM stage (Qualified, Appointment Booked, Proposal Sent, Negotiation, Closed Won).
        notes: Summary of the conversation or call note to append to CRM.
        budget_inr: Optional updated deal value in INR.
    """
    _current_request_tools.append(f"update_crm_pipeline({lead_identifier} -> {new_stage})")
    query = lead_identifier.strip().lower()
    target = None
    for lead in CRM_LEADS_DB:
        if (
            lead["lead_id"].lower() == query
            or query in lead["full_name"].lower()
            or (len(query) >= 6 and query in lead["phone"])
        ):
            target = lead
            break

    if not target:
        return {
            "updated": False,
            "message": f"No CRM lead matched '{lead_identifier}'. Ask for their Lead ID (e.g., LEAD-101) or full name.",
        }

    target["stage"] = new_stage.strip()
    if budget_inr.strip():
        target["budget_inr"] = budget_inr.strip()
    target["notes"] = notes.strip()
    target["updated_at"] = datetime.now().strftime("%I:%M %p")
    return {"updated": True, **target}


def get_available_slots() -> dict[str, Any]:
    """Check open consultation and site-visit calendar slots."""
    _current_request_tools.append("get_available_slots()")
    booked = {a["slot"].lower() for a in APPOINTMENTS_DB if a["status"] == "Confirmed"}
    open_slots = [s for s in AVAILABLE_SLOTS if s.lower() not in booked]
    return {"open_slots": open_slots, "booked_slots": list(booked)}


def book_consultation_appointment(
    customer_name: str,
    phone: str,
    preferred_slot: str,
    visit_mode: str = "Studio Consultation",
) -> dict[str, Any]:
    """Book a consultation or site visit slot and automatically sync the CRM lead stage to 'Appointment Booked'.

    Args:
        customer_name: Full name of the prospect.
        phone: 10-digit Indian mobile number.
        preferred_slot: Desired date and time slot (e.g., Tomorrow 4:30 PM).
        visit_mode: Either 'Studio Consultation', 'Site Visit', or 'Google Meet Video Call'.
    """
    _current_request_tools.append(f"book_consultation_appointment({customer_name}, {preferred_slot})")
    booked_lower = {a["slot"].lower() for a in APPOINTMENTS_DB if a["status"] == "Confirmed"}
    if preferred_slot.strip().lower() in booked_lower:
        open_slots = [s for s in AVAILABLE_SLOTS if s.lower() not in booked_lower]
        return {
            "booked": False,
            "message": f"'{preferred_slot}' is already booked. Available slots are: {', '.join(open_slots)}.",
        }

    clean_phone = re.sub(r"\D", "", phone)[-10:] or phone.strip()
    matched_lead_id = "NEW-LEAD"
    for lead in CRM_LEADS_DB:
        if lead["phone"] == clean_phone or lead["full_name"].lower() == customer_name.strip().lower():
            lead["stage"] = "Appointment Booked"
            lead["notes"] = f"Booked {visit_mode} for {preferred_slot}."
            lead["updated_at"] = datetime.now().strftime("%I:%M %p")
            matched_lead_id = lead["lead_id"]
            break

    if matched_lead_id == "NEW-LEAD":
        created = qualify_and_score_lead(
            full_name=customer_name,
            phone=clean_phone,
            service_interest=f"Consultation ({visit_mode})",
            budget_inr="₹10,00,000+",
            timeline="Immediate",
        )
        matched_lead_id = created["lead_id"]
        for lead in CRM_LEADS_DB:
            if lead["lead_id"] == matched_lead_id:
                lead["stage"] = "Appointment Booked"

    booking_id = f"APT-{random.randint(502, 999)}"
    appt = {
        "booking_id": booking_id,
        "lead_id": matched_lead_id,
        "customer_name": customer_name.strip(),
        "phone": clean_phone,
        "slot": preferred_slot.strip(),
        "visit_mode": visit_mode.strip(),
        "status": "Confirmed",
        "created_at": datetime.now().strftime("%I:%M %p"),
    }
    APPOINTMENTS_DB.insert(0, appt)
    return {"booked": True, "crm_synced_stage": "Appointment Booked", **appt}


def backup_revenue_engine(session_id: str, user_msg: str) -> str:
    """Zero-downtime backup engine when Gemini cloud servers experience 503 high-demand spikes."""
    text = user_msg.strip()
    lower = text.lower()

    # 1. Check open calendar slots
    if any(k in lower for k in ("slot", "available", "calendar", "free time", "timing")) and not re.search(r"[6-9]\d{9}", re.sub(r"[\s\-]", "", text)):
        info = get_available_slots()
        return (
            "Here are our open consultation & site-visit slots:\n"
            + "\n".join(f"• {s}" for s in info["open_slots"])
            + "\n\nShare your Name, 10-digit Mobile, and preferred slot to confirm your booking!"
        )

    # 2. CRM Pipeline Update (e.g., LEAD-101)
    lead_id_match = re.search(r"\b(LEAD-\d+)\b", text, flags=re.I)
    if lead_id_match and any(k in lower for k in ("stage", "crm", "update", "proposal", "won", "qualified", "note")):
        lid = lead_id_match.group(1).upper()
        stage = "Proposal Sent"
        if "won" in lower or "closed" in lower:
            stage = "Closed Won"
        elif "book" in lower or "appointment" in lower:
            stage = "Appointment Booked"
        elif "qualif" in lower:
            stage = "Qualified"
        res = update_crm_pipeline(lid, stage, text)
        if not res.get("updated"):
            return res["message"]
        return (
            f"CRM record {res['lead_id']} updated automatically:\n"
            f"• Client: {res['full_name']} ({res['phone']})\n"
            f"• New Pipeline Stage: {res['stage']}\n"
            f"• Deal Value: {res['budget_inr']}\n"
            f"• Notes: {res['notes']}"
        )

    # 3. Appointment Booking
    phone_match = re.search(r"\b([6-9]\d{9})\b", re.sub(r"[\s\-]", "", text))
    slot_match = re.search(
        r"\b((?:tomorrow|saturday|monday|sunday|today)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
        text,
        flags=re.I,
    )
    if phone_match and (slot_match or any(k in lower for k in ("book", "appointment", "visit", "consultation", "schedule"))):
        phone = phone_match.group(1)
        slot = slot_match.group(1).title() if slot_match else "Tomorrow 4:30 PM"
        mode = "Site Visit" if "site" in lower else ("Google Meet Video Call" if "video" in lower or "online" in lower else "Studio Consultation")
        name_part = text.split(",")[0] if "," in text else re.sub(r"[\d\+].*", "", text)
        name_part = re.sub(r"\b(book|schedule|appointment|visit|for|my name is|i am|hi|hello)\b", "", name_part, flags=re.I).strip()
        customer_name = name_part.title() if len(name_part) >= 2 else "Arjun Nair"
        res = book_consultation_appointment(customer_name, phone, slot, mode)
        if not res.get("booked"):
            return res["message"]
        return (
            f"Appointment confirmed & synced to CRM:\n"
            f"• Booking ID: {res['booking_id']} (Linked to {res['lead_id']})\n"
            f"• Client: {res['customer_name']} ({res['phone']})\n"
            f"• Slot: {res['slot']} ({res['visit_mode']})\n"
            f"• CRM Stage: Auto-updated to {res['crm_synced_stage']}"
        )

    # 4. New Lead Qualification + Scoring
    if phone_match or any(k in lower for k in ("bhk", "interior", "villa", "office", "budget", "lakh", "renovat", "quote")):
        if not phone_match:
            PENDING_BOOKING[session_id] = {"interest": text}
            return (
                "Great! I can qualify your project estimate and check consultant availability right away. "
                "Please share your Full Name, 10-digit Mobile Number, and Approximate Budget (e.g., Karthik Rao, 9876543210, 3BHK budget 16 lakhs)."
            )
        phone = phone_match.group(1)
        budget_m = re.search(r"(₹?\s*\d+(?:\.\d+)?\s*(?:lakhs?|l|cr|crores?|k))", text, flags=re.I)
        budget = budget_m.group(1).strip() if budget_m else "₹15 Lakhs"
        name_part = text.split(",")[0] if "," in text else re.sub(r"[\d\+].*", "", text)
        name_part = re.sub(r"\b(i want|looking for|my name is|i am|hi|hello|need)\b", "", name_part, flags=re.I).strip()
        customer_name = name_part.title() if 2 <= len(name_part) <= 35 else "Karthik Rao"
        lead = qualify_and_score_lead(customer_name, phone, text, budget, "Within 30 days")
        return (
            f"New lead captured, scored & added to CRM:\n"
            f"• Lead ID: {lead['lead_id']}\n"
            f"• Name & Mobile: {lead['full_name']} ({lead['phone']})\n"
            f"• AI Lead Score: {lead['score']}/100 ({lead['tier']} LEAD)\n"
            f"• Assigned To: {lead['assigned_to']}\n"
            f"• Next Step: Would you like me to book a consultation slot for Tomorrow 11:00 AM or Tomorrow 4:30 PM?"
        )

    return (
        "Welcome to UrbanNest Growth AI (Lead Gen + CRM + Appointment Engine). Try any flow:\n"
        "• Lead Capture & AI Scoring: Share name, phone, project & budget\n"
        "• CRM Pipeline Update: Update LEAD-101 or LEAD-102 stage & notes\n"
        "• Appointment Booking: Check open slots and book a consultation with automatic CRM sync"
    )


SYSTEM_INSTRUCTION = """
You are **UrbanNest Growth AI**, an end-to-end **Lead Generation, CRM Automation, and Appointment Booking Agent** for **UrbanNest Luxury Interiors & Realty (Bengaluru)**.

YOUR 3 CORE AUTOMATION WORKFLOWS:
1. **Lead Generation & Qualification (`qualify_and_score_lead`):**
   - Collect the prospect's **Full Name**, **10-digit Mobile Number**, **Project/Property Requirement**, **Budget in INR (₹)**, and **Timeline**.
   - As soon as you have Name, Phone, Requirement, and Budget, IMMEDIATELY call `qualify_and_score_lead` to score the lead (0–100) and log them into the CRM.
2. **CRM Pipeline Automation (`update_crm_pipeline`):**
   - When asked to update a lead's status, deal notes, or pipeline stage (e.g., `LEAD-101` or `LEAD-102`), call `update_crm_pipeline` immediately.
3. **Appointment & Calendar Automation (`get_available_slots` & `book_consultation_appointment`):**
   - When a prospect asks for timings, call `get_available_slots`.
   - When they provide Name, Phone, and a Slot, call `book_consultation_appointment`—which checks conflicts and automatically advances their CRM stage to `Appointment Booked`.

STYLE RULES:
- Concise, executive, and mobile-friendly with clean bullet points and Indian Rupees (₹).
- Never invent booking IDs or lead IDs without calling the tools above.
"""


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
async def health():
    return {
        "status": "running",
        "models": FALLBACK_MODELS,
        "leads_count": len(CRM_LEADS_DB),
        "appointments_count": len(APPOINTMENTS_DB),
    }


@app.get("/api/dashboard")
async def get_dashboard():
    return {"leads": CRM_LEADS_DB, "appointments": APPOINTMENTS_DB, "slots": AVAILABLE_SLOTS}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    global _current_request_tools

    if not GEMINI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"reply": "Server configuration error: GEMINI_API_KEY is missing.", "tools": []},
        )

    history = CHAT_SESSIONS.setdefault(req.session_id, [])
    user_turn = types.Content(role="user", parts=[types.Part.from_text(text=req.message)])
    conversation = history[-10:] + [user_turn]

    last_error = ""
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            _current_request_tools = []
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=[
                            qualify_and_score_lead,
                            update_crm_pipeline,
                            get_available_slots,
                            book_consultation_appointment,
                        ],
                        temperature=0.2,
                    ),
                )
                reply_text = response.text or "How can I assist with your project or booking?"
                history.append(user_turn)
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
                if len(history) > 12:
                    CHAT_SESSIONS[req.session_id] = history[-12:]
                return {
                    "reply": reply_text,
                    "tools": list(_current_request_tools),
                    "leads_count": len(CRM_LEADS_DB),
                    "appointments_count": len(APPOINTMENTS_DB),
                }
            except Exception as e:
                last_error = str(e)
                is_transient = any(code in last_error for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
                if is_transient and attempt == 0:
                    time.sleep(1.5)
                    continue
                break

    if any(code in last_error for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")):
        _current_request_tools = []
        fallback_reply = backup_revenue_engine(req.session_id, req.message)
        history.append(user_turn)
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=fallback_reply)]))
        return {
            "reply": fallback_reply,
            "tools": list(_current_request_tools),
            "leads_count": len(CRM_LEADS_DB),
            "appointments_count": len(APPOINTMENTS_DB),
        }

    return JSONResponse(
        status_code=500,
        content={"reply": f"Assistant error: {last_error}", "tools": []},
    )


@app.get("/widget.js")
async def embeddable_widget(request: Request):
    """Drop-in floating lead-capture & booking widget for any client website."""
    base_url = str(request.base_url).rstrip("/")
    js_code = f"""
(function() {{
  const btn = document.createElement('button');
  btn.innerText = '📅 Book Consultation';
  Object.assign(btn.style, {{
    position: 'fixed', bottom: '20px', right: '20px', zIndex: '9999',
    background: '#0f172a', color: '#fff', border: 'none', borderRadius: '999px',
    padding: '12px 18px', fontSize: '14px', fontWeight: '600', cursor: 'pointer',
    boxShadow: '0 8px 24px rgba(0,0,0,0.22)'
  }});
  const frame = document.createElement('iframe');
  frame.src = '{base_url}/';
  Object.assign(frame.style, {{
    position: 'fixed', bottom: '76px', right: '20px', zIndex: '9999',
    width: '380px', maxWidth: '92vw', height: '570px', maxHeight: '78vh',
    border: '1px solid #e2e8f0', borderRadius: '16px', display: 'none',
    boxShadow: '0 16px 40px rgba(0,0,0,0.24)', background: '#fff'
  }});
  btn.onclick = () => {{
    frame.style.display = frame.style.display === 'none' ? 'block' : 'none';
  }};
  document.body.appendChild(btn);
  document.body.appendChild(frame);
}})();
"""
    return Response(content=js_code, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
  <title>Lead + CRM + Appointment AI</title>
  <style>
    :root {
      --bg: #f1f5f9;
      --card: #ffffff;
      --primary: #0f172a;
      --accent: #059669;
      --bot-bg: #f8fafc;
      --border: #e2e8f0;
      --text: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); display: flex; justify-content: center; height: 100dvh; }
    .app { width: 100%; max-width: 560px; background: var(--card); display: flex; flex-direction: column; height: 100dvh; box-shadow: 0 0 30px rgba(0,0,0,0.06); }
    .header { background: var(--primary); color: #fff; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .header-left { display: flex; align-items: center; gap: 10px; }
    .avatar { width: 36px; height: 36px; border-radius: 10px; background: var(--accent); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; }
    .title { font-size: 15px; font-weight: 600; }
    .subtitle { font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 5px; margin-top: 2px; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; display: inline-block; }
    .crm-btn { background: rgba(255,255,255,0.14); color: #fff; border: 1px solid rgba(255,255,255,0.24); border-radius: 8px; padding: 7px 10px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap; }
    .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 86%; padding: 11px 14px; border-radius: 14px; font-size: 14px; line-height: 1.48; white-space: pre-wrap; word-wrap: break-word; }
    .msg.bot { align-self: flex-start; background: var(--bot-bg); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
    .msg.user { align-self: flex-end; background: var(--primary); color: #fff; border-bottom-right-radius: 4px; }
    .tool-badge { display: inline-block; font-size: 11px; font-weight: 600; color: #065f46; background: #d1fae5; padding: 3px 8px; border-radius: 6px; margin-bottom: 6px; }
    .chips { display: flex; gap: 6px; overflow-x: auto; padding: 8px 12px; border-top: 1px solid var(--border); background: #f8fafc; scrollbar-width: none; }
    .chips::-webkit-scrollbar { display: none; }
    .chip { flex-shrink: 0; background: #fff; border: 1px solid #cbd5e1; border-radius: 999px; padding: 6px 11px; font-size: 12px; color: #334155; cursor: pointer; }
    .chip:active { background: #e2e8f0; }
    .input-bar { display: flex; gap: 8px; padding: 10px 12px; border-top: 1px solid var(--border); background: #fff; }
    .input-bar input { flex: 1; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px 12px; font-size: 14px; outline: none; }
    .input-bar input:focus { border-color: var(--accent); }
    .input-bar button { background: var(--accent); color: #fff; border: none; border-radius: 10px; padding: 0 16px; font-size: 14px; font-weight: 600; cursor: pointer; }
    .input-bar button:disabled { opacity: 0.5; }
    .modal { position: fixed; inset: 0; background: rgba(15,23,42,0.55); display: none; align-items: center; justify-content: center; padding: 14px; z-index: 50; }
    .modal-card { background: #fff; width: 100%; max-width: 500px; max-height: 84vh; border-radius: 14px; padding: 16px; overflow-y: auto; }
    .modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
    .sec-title { font-size: 12px; font-weight: 700; text-transform: uppercase; color: #64748b; margin: 10px 0 6px; letter-spacing: 0.04em; }
    .card-item { border: 1px solid var(--border); border-radius: 10px; padding: 10px; margin-bottom: 8px; font-size: 13px; background: #f8fafc; }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; background: #dcfce7; color: #166534; }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="header-left">
        <div class="avatar">AI</div>
        <div>
          <div class="title">UrbanNest Revenue AI</div>
          <div class="subtitle"><span class="dot"></span> Lead Gen &bull; CRM Sync &bull; Appointments</div>
        </div>
      </div>
      <button class="crm-btn" onclick="openDashboard()">CRM (<span id="lCount">2</span>) &bull; Appts (<span id="aCount">1</span>)</button>
    </div>

    <div class="messages" id="chatBox">
      <div class="msg bot">Hi! I'm the <b>UrbanNest Growth AI Agent</b>. Watch all 3 automations work together in real time:
• <b>Lead Gen &amp; Scoring:</b> Qualifies prospects &amp; scores intent (0–100)
• <b>CRM Automation:</b> Auto-creates &amp; updates pipeline stages
• <b>Appointment Automation:</b> Checks slots &amp; books visits with CRM sync

Tap a demo prompt below or type a prospect inquiry!</div>
    </div>

    <div class="chips">
      <button class="chip" onclick="quickSend('Hi, I am Karthik Rao (9876543210). Looking for 3BHK turnkey interiors in HSR Layout, budget 16 lakhs, starting next month.')">1. Capture &amp; Score Lead</button>
      <button class="chip" onclick="quickSend('Book Tomorrow 4:30 PM studio consultation for Karthik Rao, 9876543210')">2. Book Consultation</button>
      <button class="chip" onclick="quickSend('Update LEAD-101 to Proposal Sent with note: Client approved 3D kitchen layout')">3. Update CRM LEAD-101</button>
      <button class="chip" onclick="quickSend('What consultation slots are available?')">4. Check Open Slots</button>
    </div>

    <form class="input-bar" onsubmit="sendMsg(event)">
      <input id="msgInput" type="text" placeholder="Qualify lead, update CRM, or book slot..." autocomplete="off" required />
      <button id="sendBtn" type="submit">Send</button>
    </form>
  </div>

  <div class="modal" id="dashModal" onclick="if(event.target===this)this.style.display='none'">
    <div class="modal-card">
      <div class="modal-head">
        <strong>Live CRM Pipeline &amp; Calendar</strong>
        <button onclick="document.getElementById('dashModal').style.display='none'" style="border:none;background:none;font-size:18px;cursor:pointer;">✕</button>
      </div>
      <div class="sec-title">Booked Appointments</div>
      <div id="apptList">Loading...</div>
      <div class="sec-title">CRM Leads Pipeline</div>
      <div id="leadList">Loading...</div>
    </div>
  </div>

  <script>
    const sessionId = 'rev_' + Math.random().toString(36).slice(2, 10);
    const chatBox = document.getElementById('chatBox');
    const msgInput = document.getElementById('msgInput');
    const sendBtn = document.getElementById('sendBtn');

    function appendMsg(role, text, tools = []) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      if (tools && tools.length > 0) {
        const badge = document.createElement('div');
        badge.className = 'tool-badge';
        badge.textContent = '⚡ Action: ' + tools.join(' → ');
        div.appendChild(badge);
        div.appendChild(document.createElement('br'));
      }
      const span = document.createElement('span');
      span.textContent = text;
      div.appendChild(span);
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
      return div;
    }

    function quickSend(text) {
      msgInput.value = text;
      sendMsg(new Event('submit'));
    }

    async function sendMsg(e) {
      e.preventDefault();
      const text = msgInput.value.trim();
      if (!text) return;
      msgInput.value = '';
      appendMsg('user', text);
      sendBtn.disabled = true;
      const typing = appendMsg('bot', 'Running automation...');

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, message: text })
        });
        const data = await res.json();
        typing.remove();
        appendMsg('bot', data.reply, data.tools || []);
        if (typeof data.leads_count === 'number') {
          document.getElementById('lCount').textContent = data.leads_count;
        }
        if (typeof data.appointments_count === 'number') {
          document.getElementById('aCount').textContent = data.appointments_count;
        }
      } catch (err) {
        typing.textContent = 'Network error. Please try again.';
      } finally {
        sendBtn.disabled = false;
      }
    }

    async function openDashboard() {
      const modal = document.getElementById('dashModal');
      modal.style.display = 'flex';
      const res = await fetch('/api/dashboard');
      const data = await res.json();

      document.getElementById('apptList').innerHTML = (data.appointments || []).map(a =>
        `<div class="card-item"><b>${a.booking_id}</b> &bull; <span class="pill">${a.status}</span><br><b>${a.customer_name}</b> (${a.phone}) — Linked: ${a.lead_id}<br><b>Slot:</b> ${a.slot} (${a.visit_mode})</div>`
      ).join('') || '<p style="font-size:13px;color:#64748b;">No appointments yet.</p>';

      document.getElementById('leadList').innerHTML = (data.leads || []).map(l =>
        `<div class="card-item"><b>${l.lead_id}</b> &bull; <span class="pill">${l.stage}</span> &bull; Score: <b>${l.score}/100 (${l.tier})</b><br><b>${l.full_name}</b> (${l.phone}) — ${l.budget_inr}<br><b>Need:</b> ${l.service_interest}<br><small style="color:#475569;">${l.notes}</small></div>`
      ).join('');
    }
  </script>
</body>
</html>"""
