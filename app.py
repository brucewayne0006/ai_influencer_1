from fastapi import FastAPI, Request
import os, httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query

load_dotenv()
app = FastAPI()

AIRTABLE_BASE  = os.getenv("AIRTABLE_BASE")         # e.g., appXXXX
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE", "Jobs")
AIRTABLE_PAT   = os.getenv("AIRTABLE_PAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AIRTABLE_API = f"https://api.airtable.com/v0/{AIRTABLE_BASE}/{AIRTABLE_TABLE}"
AT_HEADERS   = {"Authorization": f"Bearer {AIRTABLE_PAT}", "Content-Type": "application/json"}

def _at_record_url(record_id: str) -> str:
    return f"{AIRTABLE_API}/{record_id}"

async def airtable_update(record_id: str, fields: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        await c.patch(_at_record_url(record_id), json={"fields": fields}, headers=AT_HEADERS)

@app.post("/generate")
async def generate(req: Request):
    """
    Expected JSON body from Make.com:
      { "recordId": "...", "prompt": "...", "image_count": 2 }
    """
    body = await req.json()
    record_id  = body.get("recordId")
    prompt     = (body.get("prompt") or "").strip()
    image_cnt  = int(body.get("image_count") or 1)
    image_cnt  = max(1, min(image_cnt, 4))  # OpenAI cap in this example

    if not (record_id and prompt):
        return {"ok": False, "error": "recordId and prompt required"}

    # mark generating
    await airtable_update(record_id, {"Status": "generating"})

    try:
        # 1) call OpenAI Images
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "n": image_cnt,
                    "size": "1024x1024"
                },
            )
        r.raise_for_status()
        data = r.json()
        urls = [d["url"] for d in data.get("data", []) if "url" in d]

        if not urls:
            await airtable_update(record_id, {"Status": "error", "Log": "No image URLs returned"})
            return {"ok": False, "error": "No image URLs returned"}

        # 2) write back to Airtable
        await airtable_update(
            record_id,
            {
                "OutputImages": [{"url": u} for u in urls],
                "Status": "done",
                "Generate": False
            },
        )
        return {"ok": True, "urls": urls}

    except httpx.HTTPStatusError as e:
        msg = f"OpenAI {e.response.status_code}: {e.response.text}"
        await airtable_update(record_id, {"Status": "error", "Log": msg})
        return {"ok": False, "error": msg}
    except Exception as e:
        await airtable_update(record_id, {"Status": "error", "Log": str(e)})
        return {"ok": False, "error": str(e)}

@app.get("/")
def health():
    return {"ok": True, "airtable": bool(AIRTABLE_PAT), "openai": bool(OPENAI_API_KEY)}

@app.get("/generate")
async def generate_get(
    recordId: str = Query(None),
    prompt: str = Query(None),
    image_count: int = Query(1)
):
    # forward to the POST handler shape
    class FakeReq:
        async def json(self):
            return {"recordId": recordId, "prompt": prompt, "image_count": image_count}
    return await generate(FakeReq())