from fastapi import FastAPI, Request
import httpx, os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

AIRTABLE_BASE   = os.getenv("AIRTABLE_BASE")
AIRTABLE_TABLE  = os.getenv("AIRTABLE_TABLE", "Jobs")
AIRTABLE_PAT    = os.getenv("AIRTABLE_PAT")
REPLICATE_TOKEN = os.getenv("REPLICATE_TOKEN")
PUBLIC_DOMAIN   = os.getenv("PUBLIC_DOMAIN")


# two versions: text-only FLUX and image-guided InstantID-for-FLUX
VER_FLUX        = os.getenv("REPLICATE_VERSION_FLUX")
VER_INSTANTID   = os.getenv("REPLICATE_VERSION_INSTANTID")  # optional until you add it
def _mask(s): 
    return (s[:6] + "…" + s[-4:]) if s else "MISSING"

print("ENV CHECK:",
      "AIRTABLE_BASE:", os.getenv("AIRTABLE_BASE"),
      "| VER_FLUX:", os.getenv("REPLICATE_VERSION_FLUX"),
      "| PUBLIC_DOMAIN:", os.getenv("PUBLIC_DOMAIN"),
      "| REP TOKEN:", _mask(os.getenv("REPLICATE_TOKEN")))

AIRTABLE_API = "https://api.airtable.com/v0"

async def airtable_get_record(record_id: str):
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{AIRTABLE_API}/{AIRTABLE_BASE}/{AIRTABLE_TABLE}/{record_id}",
            headers={"Authorization": f"Bearer {AIRTABLE_PAT}"}
        )
        r.raise_for_status()
        return r.json().get("fields", {})

async def airtable_update(record_id: str, fields: dict):
    async with httpx.AsyncClient() as c:
        await c.patch(
            f"{AIRTABLE_API}/{AIRTABLE_BASE}/{AIRTABLE_TABLE}/{record_id}",
            json={"fields": fields},
            headers={"Authorization": f"Bearer {AIRTABLE_PAT}"}
        )

async def replicate_create(payload: dict):
    async with httpx.AsyncClient(
        headers={"Authorization": f"Token {REPLICATE_TOKEN}"}, timeout=60
    ) as c:
        who = await c.get("https://api.replicate.com/v1/account")
        print("WHOAMI:", who.status_code, who.text)  # must be 200
        r = await c.post("https://api.replicate.com/v1/predictions", json=payload)
        print("REPLICATE RESP:", r.status_code, r.text)  # see exact error
        return r

@app.get("/generate")
async def generate(rid: str):
    fields = await airtable_get_record(rid)

    prompt    = fields.get("Prompt", "")
    images    = fields.get("Reference Image") or []
    ref_img   = (images[0].get("url") if images else None)
    model_sel = (fields.get("Model") or "flux").strip().lower()  # "flux" or "instantid"
    count     = int(fields.get("Image Count", 1))
    strength  = float(fields.get("Guidance Strength", 0.75))
    seed      = fields.get("Seed")

    # Choose model automatically:
    # - If a Reference Image exists and you selected "instantid", use InstantID.
    # - Else default to FLUX text-to-image.
    use_instantid = (model_sel == "instantid") and ref_img and VER_INSTANTID

    if use_instantid:
        # inside the "else:" FLUX block
        payload = {
            **({"version": VER_FLUX} if VER_FLUX else {"model": "black-forest-labs/flux-1.1-pro"}),
            "input": {
                "prompt": prompt,
                # optional:
                # "aspect_ratio": "1:1",
                # "guidance": 3.5,
                # "seed": seed,
            },
            "webhook": f"{PUBLIC_DOMAIN}/done",
            "webhook_events_filter": ["completed", "failed"],
            "metadata": {"airtable_record_id": rid, "model": "flux"}
        }

    else:
        # FLUX 1.1 Pro (text-to-image)
        payload = {
            "version": VER_FLUX,
            "input": {
                "prompt": prompt,
                # common FLUX inputs you can add:
                # "width": 1024, "height": 1024,
                # "num_outputs": count,
                # "steps": 20, "guidance": 3.5, "seed": seed,
                # Some FLUX deployments call it "go_fast" or "safety_tolerance"
            },
            "webhook": f"{PUBLIC_DOMAIN}/done",
            "webhook_events_filter": ["completed", "failed"],
            "metadata": {"airtable_record_id": rid, "model": "flux"}
        }

    r = await replicate_create(payload)
    if r.status_code >= 400:
        err = f"Replicate {r.status_code}: {r.text}"
        await airtable_update(rid, {"Status": "failed", "Error": f"Replicate {r.status_code}: {r.text}"})
        return {"ok": False, "error":err}

    await airtable_update(rid, {"Status": "processing"})
    return {"ok": True}


@app.post("/done")
async def done(request: Request):
    body = await request.json()
    rid = (body.get("metadata") or {}).get("airtable_record_id")
    status = body.get("status")
    outputs = body.get("output") or []

    if not rid:
        return {"ok": False, "error": "no record id in webhook"}

    if status == "failed":
        await airtable_update(rid, {"Status": "failed", "Error": body.get("error")})
        return {"ok": True}

    await airtable_update(rid, {
        "Status": "completed",
        "Outputs": [{"url": u} for u in outputs]
    })
    return {"ok": True}

@app.get("/debug_env")
def debug_env():
    return {
        "AIRTABLE_BASE": bool(AIRTABLE_BASE),
        "AIRTABLE_TABLE": AIRTABLE_TABLE,
        "REPLICATE_VERSION_FLUX": bool(os.getenv("REPLICATE_VERSION_FLUX")),
        "PUBLIC_DOMAIN": PUBLIC_DOMAIN,
        "REPLICATE_TOKEN_loaded": bool(REPLICATE_TOKEN)
    }
