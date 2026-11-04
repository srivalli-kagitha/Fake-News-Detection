"""Browser system tests + report screenshots (Chapter 7 and 8). Needs the app running on :5173.
Usage: python run_ui_tests.py <image folder> <screenshot folder>"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

IMG, SS = Path(sys.argv[1]), Path(sys.argv[2]); SS.mkdir(exist_ok=True)
APP = "http://localhost:5173"
OUT = Path(__file__).parent / "results"
S = {
 "enFake": "BREAKING: Scientists confirm that 5G mobile towers are spreading the coronavirus across India. The government has secretly ordered all towers to be shut down by Monday. Share this before it gets deleted!",
 "te": "తెలంగాణ రాష్ట్రం 2 జూన్ 2014న ఏర్పడింది. హైదరాబాద్ తెలంగాణ రాజధాని. 5జీ టవర్ల వల్ల కరోనా వైరస్ వ్యాపిస్తుందని సోషల్ మీడియాలో ప్రచారం జరుగుతోంది.",
 "article": "The new 2000 rupee notes contain a GPS nano chip that lets the government track them, officials confirmed on Monday. India's Chandrayaan-3 lander touched down near the Moon's south pole on 23 August 2023. UNESCO has declared Jana Gana Mana the best national anthem in the world. Forward this message to everyone you know!",
}
rows = []
def rec(cid, scen, expected, fn):
    t = time.time()
    try: actual, ok = fn()
    except Exception as e: actual, ok = f"error: {str(e).splitlines()[0][:120]}", False
    rows.append(dict(id=cid, scenario=scen, expected=expected, actual=actual, status="Pass" if ok else "Fail", seconds=round(time.time()-t,1)))
    print(cid, "PASS" if ok else "FAIL", actual, flush=True)

LONG = 20 * 60 * 1000
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5, accept_downloads=True)
    pg = ctx.new_page()

    pg.goto(APP); pg.wait_for_selector("text=Is this news real?"); pg.wait_for_timeout(1500)
    pg.screenshot(path=SS / "ss_home.png")

    def st01():
        pg.goto(APP); pg.fill("textarea", S["article"]); pg.get_by_role("button", name="Check this text").click()
        pg.wait_for_url("**/text", timeout=10000)
        t0 = time.time(); pg.wait_for_selector(".verdict-strip", timeout=LONG); t1 = time.time() - t0
        rec.t_verdict = t1
        return f"opened /text, verdict shown after {t1:.0f} s", "/text" in pg.url
    rec("ST01", "Start page sends text to the text check", "Text page opens and the check starts", st01)

    def st02():
        pending = pg.locator("text=Highlighting the words behind the verdict").count() > 0
        t0 = time.time(); pg.wait_for_selector(".mark", timeout=LONG)
        pg.wait_for_selector("text=Highlighting the words behind the verdict", state="detached", timeout=LONG)
        t2 = time.time() - t0
        n = pg.locator(".mark").count()
        return f"verdict on screen while highlights pending: {pending}; {n} words marked {t2:.0f} s later", pending and n > 0
    rec("ST02", "Verdict shown before highlights", "Verdict first, highlights added later with a progress note", st02)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=SS / "ss_text_report.png", full_page=True)

    def st03():
        li = pg.locator("ol li").first; li.scroll_into_view_if_needed(); li.hover(); pg.wait_for_timeout(600)
        n = pg.locator(".claim-active").count()
        return f"{n} words of the claim lit up in the article", n > 0
    rec("ST03", "Hovering a claim highlights it in the article", "Matching sentence lit up", st03)
    pg.locator("section[aria-label=Report]").screenshot(path=SS / "ss_text_claims_hover.png")

    def st04():
        pg.goto(APP + "/image"); pg.wait_for_timeout(800); pg.goto(APP + "/text"); pg.wait_for_timeout(1000)
        ok = pg.locator(".verdict-strip").count() > 0 and pg.locator("textarea").input_value().startswith("The new 2000")
        return f"report and input still present after leaving the page: {ok}", ok
    rec("ST04", "Results kept when moving between pages", "Report still shown on return", st04)

    def st05():
        pg.get_by_role("button", name="Dark").click(); pg.reload(); pg.wait_for_timeout(1200)
        th = pg.evaluate("document.documentElement.getAttribute('data-theme')")
        return f"data-theme after reload = {th}", th == "dark"
    rec("ST05", "Dark theme is remembered", "Dark theme kept after reload", st05)
    pg.wait_for_selector(".verdict-strip", timeout=10000); pg.wait_for_timeout(1500)
    pg.screenshot(path=SS / "ss_text_dark.png")
    pg.get_by_role("button", name="Light").click(); pg.wait_for_timeout(500)

    def st06():
        pg.emulate_media(media="print"); vis = pg.locator("aside").is_visible(); pg.emulate_media(media="screen")
        return f"navigation rail visible in print layout: {vis}", not vis
    rec("ST06", "Save as PDF uses a print layout", "Navigation hidden when printing", st06)

    def st07():
        pg.goto(APP + "/text"); pg.fill("textarea", S["te"]); pg.get_by_role("button", name="Check this text").click()
        pg.wait_for_selector(".verdict-strip", timeout=LONG); pg.wait_for_selector(".mark", timeout=LONG)
        pg.wait_for_selector("text=Highlighting the words behind the verdict", state="detached", timeout=LONG); pg.wait_for_timeout(2500)
        lang = pg.locator("text=Telugu").count()
        pg.screenshot(path=SS / "ss_text_telugu.png", full_page=True)
        return f"Telugu report shown with {pg.locator('.mark').count()} marked words", lang > 0
    rec("ST07", "Telugu article checked end to end", "Telugu detected; verdict, highlights and claims shown", st07)

    def st08():
        pg.goto(APP + "/image"); pg.locator("input[type=file]").set_input_files(str(IMG / "fake_1.jpg"))
        pg.get_by_role("button", name="Check this image").click()
        pg.wait_for_selector("input[aria-label='Compare image and heatmap']", timeout=LONG); pg.wait_for_timeout(1500)
        sl = pg.locator("input[aria-label='Compare image and heatmap']"); sl.fill("35"); pg.wait_for_timeout(300)
        clip = pg.locator(".compare img.overlay").get_attribute("style")
        pg.screenshot(path=SS / "ss_image_report.png", full_page=True)
        return f"slider moved, heatmap style: {clip}", "35%" in (clip or "")
    rec("ST08", "Image report with heatmap slider", "Slider wipes between image and heatmap", st08)

    def st09():
        pg.goto(APP); pg.locator("input[type=file]").set_input_files(str(IMG / "real_2.jpg"))
        pg.wait_for_url("**/image", timeout=10000); pg.wait_for_selector(".verdict-strip", timeout=LONG); pg.wait_for_timeout(1200)
        pg.screenshot(path=SS / "ss_image_real.png", full_page=True)
        return "image page opened and checked the dropped image", "/image" in pg.url
    rec("ST09", "Start page sends an image to the image check", "Image page opens and checks the image", st09)

    def st10():
        pg.goto(APP + "/claims"); pg.fill("textarea", S["article"]); pg.get_by_role("button", name="Verify claims").last.click()
        pg.wait_for_selector("ol li", timeout=LONG); pg.wait_for_timeout(1500)
        pg.screenshot(path=SS / "ss_claims.png", full_page=True)
        return f"{pg.locator('ol li').count()} claims listed", pg.locator("ol li").count() >= 3
    rec("ST10", "Verify claims page", "Each claim listed with status and evidence", st10)

    def st11():
        csv = SS.parent / "_batch_sample.csv"
        csv.write_text("text\n" + "\n".join('"' + s.replace('"', '""') + '"' for s in S.values()), encoding="utf-8")
        pg.goto(APP + "/batch"); pg.locator("input[type=file]").first.set_input_files(str(csv))
        pg.locator("button.btn").first.click()
        pg.wait_for_selector("text=Download results (CSV)", timeout=LONG); pg.wait_for_timeout(1000)
        pg.screenshot(path=SS / "ss_batch.png", full_page=True)
        with pg.expect_download() as d: pg.get_by_text("Download results (CSV)").click()
        f = d.value.path(); lines = Path(f).read_text(encoding="utf-8").strip().splitlines()
        csv.unlink()
        return f"downloaded {d.value.suggested_filename} with {len(lines)-1} rows", len(lines) - 1 == 3
    rec("ST11", "Batch CSV check and download", "Result CSV with one row per input", st11)

    def st12():
        pg.goto(APP + "/history"); pg.wait_for_timeout(1500)
        pg.screenshot(path=SS / "ss_history.png", full_page=True)
        pg.get_by_role("button", name="Images").click(); pg.wait_for_timeout(800)
        body = pg.locator("tbody").inner_text() if pg.locator("tbody").count() else ""
        n = pg.locator("tbody tr").count()
        return f"{n} rows shown after filtering by Images", n > 0
    rec("ST12", "History filter", "Only image checks shown", st12)

    def st13():
        m = b.new_context(viewport={"width": 375, "height": 812}, device_scale_factor=2); mp = m.new_page()
        mp.goto(APP); mp.wait_for_timeout(1500); w = mp.evaluate("document.documentElement.scrollWidth")
        mp.screenshot(path=SS / "ss_mobile.png"); m.close()
        return f"page width {w} px at a 375 px screen", w <= 375
    rec("ST13", "Phone layout", "No horizontal scrolling at 375 px", st13)

    def st14():
        pg.goto(APP + "/text"); pg.keyboard.press("Tab"); seen = set()
        for _ in range(25):
            pg.keyboard.press("Tab"); seen.add(pg.evaluate("document.activeElement.tagName + ':' + (document.activeElement.innerText||document.activeElement.getAttribute('aria-label')||'').slice(0,20)"))
        ok = any("TEXTAREA" in s for s in seen) and any("Check this text" in s for s in seen)
        return f"{len(seen)} controls reached with Tab, including the text box and Check button: {ok}", ok
    rec("ST14", "Keyboard-only use", "Text box and Check button reachable with Tab", st14)

    pg.goto("http://127.0.0.1:8000/docs"); pg.wait_for_timeout(3000); pg.screenshot(path=SS / "ss_api_docs.png")
    b.close()

json.dump(rows, open(OUT / "system_tests.json", "w"), ensure_ascii=False, indent=1)
print(sum(r["status"] == "Pass" for r in rows), "/", len(rows))
