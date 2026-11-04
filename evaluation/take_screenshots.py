"""Screenshots for the report (Chapter 8). Uses a tall window instead of full-page capture."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
IMG, SS = Path(sys.argv[1]), Path(sys.argv[2]); APP = "http://localhost:5173"; LONG = 1200000
FAKE = "BREAKING: Scientists confirm that 5G mobile towers are spreading the coronavirus across India. The government has secretly ordered all towers to be shut down by Monday. Share this before it gets deleted!"
MIXED = "The new 2000 rupee notes contain a GPS nano chip that lets the government track them, officials confirmed on Monday. India's Chandrayaan-3 lander touched down near the Moon's south pole on 23 August 2023. UNESCO has declared Jana Gana Mana the best national anthem in the world. Forward this message to everyone you know!"
TE = "తెలంగాణ రాష్ట్రం 2 జూన్ 2014న ఏర్పడింది. హైదరాబాద్ తెలంగాణ రాజధాని. 5జీ టవర్ల వల్ల కరోనా వైరస్ వ్యాపిస్తుందని సోషల్ మీడియాలో ప్రచారం జరుగుతోంది."

def tall(pg, name, max_h=3200):
    h = min(pg.evaluate("document.documentElement.scrollHeight"), max_h)
    pg.set_viewport_size({"width": 1440, "height": h}); pg.wait_for_timeout(1800)
    pg.screenshot(path=SS / name); pg.set_viewport_size({"width": 1440, "height": 900}); pg.wait_for_timeout(300)
    print("saved", name, h, flush=True)

def nav(pg, label, url):
    pg.locator("aside").get_by_role("link", name=label).click(); pg.wait_for_url(url); pg.wait_for_timeout(1200)

def text_check(pg, text):
    pg.fill("textarea", text); pg.get_by_role("button", name="Check this text").click()
    pg.wait_for_selector(".verdict-strip", timeout=LONG); pg.wait_for_selector(".mark", timeout=LONG)
    pg.wait_for_selector("text=Highlighting the words behind the verdict", state="detached", timeout=LONG); pg.wait_for_timeout(2500)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5).new_page()
    pg.goto(APP); pg.wait_for_timeout(2000)
    nav(pg, "Check news text", "**/text"); text_check(pg, FAKE); tall(pg, "ss_text_report.png")
    pg.locator("aside").get_by_role("button", name="Dark").click(); pg.wait_for_timeout(1200)
    pg.screenshot(path=SS / "ss_text_dark.png"); print("saved dark", flush=True)
    pg.locator("aside").get_by_role("button", name="Light").click(); pg.wait_for_timeout(800)
    pg.get_by_role("button", name="Start a new check").first.click(); pg.wait_for_timeout(500)
    text_check(pg, MIXED); tall(pg, "ss_text_mixed.png")
    pg.get_by_role("button", name="Start a new check").first.click(); pg.wait_for_timeout(500)
    text_check(pg, TE); tall(pg, "ss_text_telugu.png")
    nav(pg, "Check an image", "**/image")
    pg.locator("input[type=file]").set_input_files(str(IMG / "fake_1.jpg")); pg.get_by_role("button", name="Check this image").click()
    pg.wait_for_selector("input[aria-label='Compare image and heatmap']", timeout=LONG); pg.wait_for_timeout(1500)
    pg.locator("input[aria-label='Compare image and heatmap']").fill("50"); tall(pg, "ss_image_report.png")
    pg.locator("input[type=file]").set_input_files(str(IMG / "real_2.jpg")); pg.get_by_role("button", name="Check this image").click()
    pg.wait_for_timeout(1500); pg.wait_for_selector("input[aria-label='Compare image and heatmap']", timeout=LONG); pg.wait_for_timeout(1500)
    tall(pg, "ss_image_real.png")
    nav(pg, "Verify claims", "**/claims"); pg.fill("textarea", MIXED); pg.get_by_role("button", name="Verify claims").last.click()
    pg.wait_for_selector("ol.divide-y > li", timeout=LONG); pg.wait_for_timeout(1500); tall(pg, "ss_claims.png")
    nav(pg, "Check many at once", "**/batch")
    csv = SS.parent / "_news_sample.csv"; csv.write_text("text\n" + "\n".join('"' + s.replace('"', '""') + '"' for s in (FAKE, MIXED, TE)), encoding="utf-8")
    pg.locator("input[type=file]").first.set_input_files(str(csv)); pg.locator("button.btn").first.click()
    pg.wait_for_selector("text=Download results (CSV)", timeout=LONG); pg.wait_for_timeout(1000); tall(pg, "ss_batch.png"); csv.unlink()
    nav(pg, "History", "**/history"); pg.wait_for_timeout(800); pg.screenshot(path=SS / "ss_history.png"); print("saved history", flush=True)
    b.close()
