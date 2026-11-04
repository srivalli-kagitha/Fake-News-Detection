"""Re-run ST03, ST04, ST14 with corrected steps (in-app navigation instead of page reload)."""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
SS = Path(sys.argv[1]); APP = "http://localhost:5173"; OUT = Path(__file__).parent / "results"
ART = "The new 2000 rupee notes contain a GPS nano chip that lets the government track them, officials confirmed on Monday. India's Chandrayaan-3 lander touched down near the Moon's south pole on 23 August 2023. UNESCO has declared Jana Gana Mana the best national anthem in the world. Forward this message to everyone you know!"
rows = json.load(open(OUT / "system_tests.json"))
def put(cid, actual, ok):
    for r in rows:
        if r["id"] == cid: r["actual"], r["status"] = actual, "Pass" if ok else "Fail"
    print(cid, "PASS" if ok else "FAIL", actual, flush=True)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5).new_page()
    pg.goto(APP + "/text"); pg.fill("textarea", ART); pg.get_by_role("button", name="Check this text").click()
    pg.wait_for_selector(".verdict-strip", timeout=1200000); pg.wait_for_selector(".mark", timeout=1200000)
    pg.wait_for_selector("text=Highlighting the words behind the verdict", state="detached", timeout=1200000); pg.wait_for_timeout(2000)
    # ST03: hover the second claim in the evidence list
    li = pg.locator("ol.divide-y > li").nth(1); li.scroll_into_view_if_needed(); li.hover(); pg.wait_for_timeout(700)
    n = pg.locator(".claim-active").count()
    put("ST03", f"hovering claim 2 lit up {n} words of that sentence in the article", n > 0)
    pg.locator(".proof").scroll_into_view_if_needed(); pg.wait_for_timeout(300)
    pg.locator("section[aria-label=Report]").screenshot(path=SS / "ss_text_claims_hover.png")
    # ST04: leave through the rail and come back through the rail
    pg.locator("aside").get_by_role("link", name="Check an image").click(); pg.wait_for_url("**/image")
    pg.locator("aside").get_by_role("link", name="Check news text").click(); pg.wait_for_url("**/text"); pg.wait_for_timeout(1000)
    ok = pg.locator(".verdict-strip").count() > 0 and pg.locator("textarea").input_value().startswith("The new 2000")
    put("ST04", f"after moving to the image page and back with the rail, report and input still shown: {ok}", ok)
    # ST14: keyboard only, starting from the top of the page
    pg.evaluate("document.activeElement && document.activeElement.blur()"); seen = []
    for _ in range(40):
        pg.keyboard.press("Tab")
        seen.append(pg.evaluate("(e => e.tagName + ':' + (e.innerText || e.getAttribute('aria-label') || '').trim().slice(0, 24))(document.activeElement)"))
    ok = any(s.startswith("TEXTAREA") for s in seen) and any("Check this text" in s for s in seen)
    put("ST14", f"{len(set(seen))} controls reached with Tab, including the text box and the Check button: {ok}", ok)
    b.close()
json.dump(rows, open(OUT / "system_tests.json", "w"), ensure_ascii=False, indent=1)
print(sum(r["status"] == "Pass" for r in rows), "/", len(rows))
