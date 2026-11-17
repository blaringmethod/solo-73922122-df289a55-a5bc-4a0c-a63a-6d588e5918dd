"""Self-contained OAI-PMH demo source.

Run with::

    uvicorn demo_oai.main:app --host 0.0.0.0 --port 8001

Endpoints:

* ``GET /oai`` – OAI-PMH endpoint (Identify, ListRecords, ListMetadataFormats);
* ``GET /`` – control panel (fault injection, add/correct/withdraw records);
* ``GET /healthz`` – readiness probe.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .repository import DemoRepo, apply_slow_delay

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="OAI-PMH Demo Repository", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DB_PATH = os.environ.get(
    "DEMO_OAI_DB", str(Path(os.environ.get("DATA_DIR", "./data")) / "demo_oai.sqlite3")
)
PAGE_SIZE = int(os.environ.get("DEMO_PAGE_SIZE", "3"))

repo = DemoRepo(DB_PATH, page_size=PAGE_SIZE)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/oai")
async def oai_endpoint(request: Request) -> Response:
    raw = request.url.query
    # The demo endpoint follows OAI-PMH's parameter rule: ``&`` separates
    # arguments.  Semicolons are accepted as well for convenience.
    raw = raw.replace(";", "&")
    pairs = parse_qsl(raw, keep_blank_values=True)
    params: dict[str, str] = {}
    for key, value in pairs:
        # OAI-PMH arguments must not repeat.
        if key in params:
            xml = repo._error("badArgument", f"参数 {key} 重复")
            return Response(content=xml, media_type="text/xml; charset=utf-8")
        params[key] = value
    verb = params.get("verb", "")
    if not verb:
        xml = repo._error("badVerb", "缺少 verb 参数")
        return Response(content=xml, media_type="text/xml; charset=utf-8")

    await apply_slow_delay(repo)
    xml = repo.dispatch(verb, params)
    return Response(content=xml, media_type="text/xml; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
async def panel(request: Request, message: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "panel.html",
        {
            "state": repo.state(),
            "records": repo.all_records(),
            "message": message,
        },
    )


@app.post("/demo/settings")
async def settings(
    page_size: int = Form(...),
    slow_seconds: float = Form(...),
    token_failures: int = Form(...),
    token_ttl_seconds: int = Form(...),
) -> RedirectResponse:
    repo.set_meta("page_size", str(max(1, min(page_size, 50))))
    repo.set_meta("slow_seconds", str(max(0.0, min(slow_seconds, 60.0))))
    repo.set_meta("token_failures", str(max(0, min(token_failures, 20))))
    repo.set_meta("token_ttl_seconds", str(max(0, min(token_ttl_seconds, 86400))))
    return _back("设置已保存")


@app.post("/demo/add")
async def add(
    title: str = Form(...),
    creator: str = Form(""),
    identifier: str = Form(""),
) -> RedirectResponse:
    if not title.strip():
        return _back("标题不能为空")
    repo.add_record(
        title=title.strip(),
        creator=creator.strip() or None,
        identifier=identifier.strip() or None,
    )
    return _back("已新增记录（源端 datestamp 为当前秒）")


@app.post("/demo/correct")
async def correct(identifier: str = Form(...), title: str = Form(...)) -> RedirectResponse:
    try:
        repo.touch_record(identifier.strip(), title=title.strip() or None)
    except KeyError:
        return _back(f"记录 {identifier} 不存在")
    return _back("已更正记录（datestamp 更新为当前秒）")


@app.post("/demo/withdraw")
async def withdraw(identifier: str = Form(...)) -> RedirectResponse:
    try:
        repo.withdraw(identifier.strip())
    except KeyError:
        return _back(f"记录 {identifier} 不存在")
    return _back("已撤回（header status=deleted）")


@app.post("/demo/restore")
async def restore(identifier: str = Form(...)) -> RedirectResponse:
    try:
        repo.restore(identifier.strip())
    except KeyError:
        return _back(f"记录 {identifier} 不存在")
    return _back("已恢复记录")


@app.post("/demo/withdraw-all")
async def withdraw_all() -> RedirectResponse:
    count = repo.withdraw_all()
    return _back(f"已撤回 {count} 条记录")


@app.post("/demo/pagination-mutation")
async def pagination_mutation(
    mode: str = Form("auto"),
    identifier: str = Form(""),
    new_datestamp: str = Form(""),
    late_identifier: str = Form(""),
    late_title: str = Form("翻页期间补入的迟到记录"),
    late_datestamp: str = Form(""),
    correct_title: str = Form(""),
    repeat: int = Form(1),
) -> RedirectResponse:
    """Arm one "members/content changed between two pages" event.

    On the next resumptionToken request the source changes the window (keeping
    the record count constant) and only then validates the now-stale token.

    * ``auto`` - the source picks a current window member, corrects its
      datestamp to just outside the harvest window and inserts one late record
      just inside it;
    * ``explicit`` - the identifiers/datestamps below are used verbatim.

    ``repeat`` > 1 re-arms the same event for subsequent retries, so a round
    that keeps retrying eventually exhausts ``HARVEST_MAX_ATTEMPTS``.
    """
    repeat = max(1, min(repeat, 20))
    if mode == "explicit":
        if not (identifier.strip() and late_identifier.strip()):
            return _back("显式模式：更正标识与迟到标识都不能为空")
        try:
            from .repository import _identified

            _identified(new_datestamp, "new_datestamp")
            _identified(late_datestamp, "late_datestamp")
        except ValueError as exc:
            return _back(f"时间戳不合法：{exc}")
        spec = {
            "correct": [identifier.strip(), new_datestamp.strip()],
            "correct_title": correct_title.strip() or None,
            "add": [
                late_identifier.strip(),
                late_title.strip() or "翻页期间补入的迟到记录",
                late_datestamp.strip(),
            ],
            "repeat": repeat,
        }
    else:
        spec = {
            "auto": True,
            "late_title": late_title.strip() or "翻页期间补入的迟到记录",
            "correct_title": correct_title.strip() or None,
            "repeat": repeat,
        }
    repo.arm_pagination_mutation(spec)
    suffix = f"，并连续安排 {repeat} 次" if repeat > 1 else ""
    return _back(f"已安排：下一次翻页时先更正移出并补入记录，再作废旧令牌{suffix}")


@app.get("/demo/state.json")
async def state_json() -> JSONResponse:
    return JSONResponse(
        {
            "state": repo.state(),
            "records": repo.all_records(),
        }
    )


def _back(message: str) -> RedirectResponse:
    from urllib.parse import quote

    return RedirectResponse(url=f"/?message={quote(message)}", status_code=303)
