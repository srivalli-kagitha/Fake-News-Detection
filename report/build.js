// Builds the project report (.docx) from chapters/*.md.
// Usage: node build.js <output.docx>
//
// Chapter markup (one block per paragraph, blank line between blocks):
//   # Title                 chapter (new page, "CHAPTER N")
//   ## Title / ### Title    numbered sections N.M / N.M.K
//   - item / 1. item        bullet / numbered list (consecutive lines)
//   ![Caption](path){w=15}  figure, width in cm, auto "Figure N.k"
//   TABLE: Caption          followed by | a | b | rows; first row is the header
//   ```                     code block (monospace)
//   EQ: text                centred equation with (N.k)
//   **bold**, *italic*, `code` inline
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, NumberFormat, Packer,
  PageBreak, PageNumber, Paragraph, ShadingType, Table, TableCell, TableOfContents, TableRow, TabStopType,
  TextRun, VerticalAlign, WidthType, TableLayoutType,
} = require("docx");

const FONT = "Times New Roman";
const MONO = "Consolas";
const PAGE_W = 11906; // A4 in DXA
const MARGIN = { top: 1440, bottom: 1440, left: 2160, right: 1440 }; // 1.5" binding edge
const TEXT_W = PAGE_W - MARGIN.left - MARGIN.right; // 8306 DXA
const front = require("./front.json");

// ------------------------------------------------------------------ inline markup
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: 21, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const body = (text, opts = {}) =>
  new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 360 }, ...opts });

const centered = (text, opts = {}) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120, line: 300 }, children: runs(text, opts.run || {}), ...(opts.para || {}) });

function imageSize(file) {
  const buf = fs.readFileSync(file);
  if (buf.toString("ascii", 1, 4) === "PNG") return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), type: "png" };
  // JPEG: scan for SOF marker
  let i = 2;
  while (i < buf.length) {
    const marker = buf.readUInt16BE(i);
    const len = buf.readUInt16BE(i + 2);
    if (marker >= 0xffc0 && marker <= 0xffc3) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7), type: "jpg" };
    i += 2 + len;
  }
  throw new Error("unknown image size: " + file);
}

// ------------------------------------------------------------------ chapter parser
const figures = [];
const tables = [];

function parseChapter(src, chapterNo, title) {
  const out = [
    new Paragraph({ children: [new PageBreak()] }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 },
      children: [new TextRun({ text: `CHAPTER ${chapterNo}`, bold: true, size: 28, font: FONT })],
    }),
    new Paragraph({
      heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `${chapterNo}. ${title.toUpperCase()}` })],
    }),
  ];
  let sec = 0, sub = 0, fig = 0, tab = 0, eq = 0;
  const lines = src.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();
    if (!t) { i++; continue; }
    if (t.startsWith("## ")) {
      sec++; sub = 0;
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(`${chapterNo}.${sec} ${t.slice(3)}`)] }));
      i++; continue;
    }
    if (t.startsWith("### ")) {
      sub++;
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(`${chapterNo}.${sec}.${sub} ${t.slice(4)}`)] }));
      i++; continue;
    }
    if (t.startsWith("```")) {
      const code = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
      i++;
      code.forEach((c, k) => out.push(new Paragraph({
        spacing: { after: 0, line: 240, before: k === 0 ? 80 : 0 },
        shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" },
        indent: { left: 200, right: 200 },
        children: [new TextRun({ text: c.replace(/\t/g, "    ") || " ", font: MONO, size: 16 })],
      })));
      out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
      continue;
    }
    const figm = t.match(/^!\[(.+)\]\((.+?)\)(\{w=([\d.]+)\})?$/);
    if (figm) {
      fig++;
      const file = path.resolve(__dirname, figm[2]);
      const label = `Figure ${chapterNo}.${fig}`;
      figures.push(label);
      if (!fs.existsSync(file)) {
        out.push(centered(`[Missing image: ${figm[2]}]`, { run: { color: "C00000" } }));
      } else {
        const { w, h, type } = imageSize(file);
        const widthCm = Math.min(parseFloat(figm[4] || "14"), 14.6);
        const px = (widthCm / 2.54) * 96;
        out.push(new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
          children: [new ImageRun({ type, data: fs.readFileSync(file), transformation: { width: px, height: px * (h / w) },
            altText: { title: label, description: figm[1], name: label } })],
        }));
      }
      out.push(new Paragraph({ style: "FigureCaption", children: runs(`${label}: ${figm[1]}`) }));
      i++; continue;
    }
    if (t.startsWith("TABLE: ")) {
      tab++;
      const label = `Table ${chapterNo}.${tab}`;
      tables.push(label);
      out.push(new Paragraph({ style: "TableCaption", keepNext: true, children: runs(`${label}: ${t.slice(7)}`) }));
      i++;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      out.push(makeTable(rows));
      out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
      continue;
    }
    if (t.startsWith("EQ: ")) {
      eq++;
      out.push(new Paragraph({
        tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
        spacing: { before: 120, after: 160 },
        children: [new TextRun({ text: "\t" }), ...runs(t.slice(4), { italics: true }), new TextRun({ text: `\t(${chapterNo}.${eq})` })],
      }));
      i++; continue;
    }
    if (/^- /.test(t)) {
      while (i < lines.length && /^- /.test(lines[i].trim())) {
        out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
          spacing: { after: 80, line: 340 }, children: runs(lines[i].trim().slice(2)) }));
        i++;
      }
      out.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
      continue;
    }
    if (/^\d+\. /.test(t)) {
      const ref = `num-${chapterNo}-${i}`;
      numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] });
      while (i < lines.length && /^\d+\. /.test(lines[i].trim())) {
        out.push(new Paragraph({ numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED,
          spacing: { after: 80, line: 340 }, children: runs(lines[i].trim().replace(/^\d+\. /, "")) }));
        i++;
      }
      out.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
      continue;
    }
    // paragraph: join until blank line
    const para = [t];
    i++;
    while (i < lines.length && lines[i].trim() && !/^(#|- |\d+\. |!\[|TABLE: |```|EQ: |\|)/.test(lines[i].trim())) para.push(lines[i++].trim());
    out.push(body(para.join(" ")));
  }
  return out;
}

function makeTable(rows) {
  const n = rows[0].length;
  // width per column from text length, bounded
  const plain = (t) => (t || "").replace(/\*\*|`/g, "");
  const charW = n >= 6 ? 95 : 110; // DXA per character at the table font size
  // minimum: the longest unbreakable word (bold header counts a bit wider), plus cell margins
  const mins = Array.from({ length: n }, (_, c) => Math.min(
    Math.max(...rows.map((r, ri) => Math.max(...plain(r[c]).split(/\s+/).map((w) => w.length)) * (ri === 0 ? 1.12 : 1))) * charW + 240,
    TEXT_W * 0.3));
  const lens = Array.from({ length: n }, (_, c) => Math.max(...rows.map((r) => plain(r[c]).length), 3));
  const weights = lens.map((l) => Math.min(Math.max(l, 6), 60));
  let widths = mins.slice();
  let spare = TEXT_W - widths.reduce((a, b) => a + b, 0);
  if (spare < 0) widths = mins.map((m) => Math.floor((m / mins.reduce((a, b) => a + b, 0)) * TEXT_W));
  else {
    // give spare width to columns whose content is longer than their minimum, by weight
    const need = weights.map((w, c) => Math.max(w * charW - mins[c], 0));
    const tn = need.reduce((a, b) => a + b, 0) || 1;
    widths = widths.map((w, c) => w + Math.floor((need[c] / tn) * spare));
  }
  widths = widths.map(Math.floor);
  widths[n - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0,
      cantSplit: true,
      children: Array.from({ length: n }, (_, c) => new TableCell({
        width: { size: widths[c], type: WidthType.DXA },
        borders,
        verticalAlign: VerticalAlign.CENTER,
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "E7EAF0", color: "auto" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ spacing: { after: 0, line: 260 }, children: runs(r[c] || "", { size: n >= 6 ? 18 : 21, bold: ri === 0 || undefined }) })],
      })),
    })),
  });
}

// ------------------------------------------------------------------ front matter
const numberingConfigs = [{
  reference: "bullets",
  levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
}];

function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }
function frontHeading(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { before: 240, after: 360 },
    children: [new TextRun(text)] });
}
const gap = (pt) => new Paragraph({ spacing: { after: pt * 20 }, children: [] });

function titlePage() {
  const f = front;
  const big = (t, size, bold = true, color) => centered(t, { run: { size, bold, color } });
  const team = f.students.map((s) => centered(`${s.name} (${s.roll})`, { run: { size: 24, bold: true } }));
  return [
    gap(10),
    big(f.title, 34, true, "1F2937"),
    gap(14),
    big("A Project Report", 24, false),
    big("submitted in partial fulfilment of the requirements for the award of the degree of", 22, false),
    gap(6),
    big("BACHELOR OF TECHNOLOGY", 26),
    big("in", 22, false),
    big("INFORMATION TECHNOLOGY", 26),
    gap(10),
    big("Submitted by", 22, false),
    big(`Batch ${f.batch}`, 22, false),
    ...team,
    gap(10),
    big("Under the guidance of", 22, false),
    big(f.guide.name, 24),
    big(f.guide.designation, 22, false),
    gap(16),
    big("DEPARTMENT OF INFORMATION TECHNOLOGY", 24),
    big(f.college.toUpperCase(), 24),
    big(f.collegeNote, 20, false),
    big(f.address, 20, false),
    big(f.year, 22),
  ];
}

function certificate() {
  const f = front;
  return [
    pageBreak(),
    centered("DEPARTMENT OF INFORMATION TECHNOLOGY", { run: { bold: true, size: 26 } }),
    centered(f.college.toUpperCase(), { run: { bold: true, size: 24 } }),
    gap(18),
    frontHeading("CERTIFICATE"),
    body(`This is to certify that the project report entitled **"${f.title}"** is a bonafide record of work carried out by ${f.students.map((s) => `**${s.name} (${s.roll})**`).join(", ")} in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in Information Technology from ${f.college}, during the academic year ${f.year}.`),
    gap(70),
    new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }],
      children: [new TextRun({ text: "Project Guide", bold: true }), new TextRun({ text: "\tHead of the Department", bold: true })],
    }),
    new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }],
      children: [new TextRun(f.guide.name), new TextRun(`\t${f.hod.name}`)],
    }),
    new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }],
      children: [new TextRun(f.guide.designation), new TextRun(`\t${f.hod.designation}`)],
    }),
    gap(70),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "External Examiner", bold: true })] }),
  ];
}

function declaration() {
  const f = front;
  return [
    pageBreak(),
    frontHeading("DECLARATION"),
    body(`We hereby declare that the project report entitled **"${f.title}"** submitted to the Department of Information Technology, ${f.college}, is a record of original work done by us under the guidance of ${f.guide.name}, ${f.guide.designation}. The results in this report have not been submitted to any other university or institute for the award of any degree or diploma.`),
    gap(50),
    ...f.students.map((s) => new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 200 },
      children: [new TextRun({ text: `${s.name} (${s.roll})`, bold: true })] })),
  ];
}

function proseSection(title, file) {
  const src = fs.readFileSync(path.join(__dirname, file), "utf8").trim();
  return [pageBreak(), frontHeading(title), ...src.split(/\n\s*\n/).map((p) => body(p.replace(/\n/g, " ")))];
}

function listPage(title, style) {
  return [
    pageBreak(),
    frontHeading(title),
    new TableOfContents(title, { hyperlink: true, stylesWithLevels: [{ styleName: style, level: 1 }] }),
  ];
}

function abbreviations() {
  const rows = [["Abbreviation", "Full form"], ...front.abbreviations];
  return [pageBreak(), frontHeading("LIST OF ABBREVIATIONS"), makeTable(rows)];
}

// ------------------------------------------------------------------ assemble
const order = JSON.parse(fs.readFileSync(path.join(__dirname, "chapters/order.json"), "utf8"));
const chapterParas = [];
order.forEach((c, k) => {
  const src = fs.readFileSync(path.join(__dirname, "chapters", c.file), "utf8");
  chapterParas.push(...parseChapter(src, k + 1, c.title));
});

function backMatter(title, file) {
  const src = fs.readFileSync(path.join(__dirname, "chapters", file), "utf8");
  const out = [pageBreak(), frontHeading(title)];
  src.split("\n").filter((l) => l.trim()).forEach((l, k) => out.push(new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 }, indent: { left: 567, hanging: 567 },
    children: [new TextRun(`[${k + 1}]\t`), ...runs(l.trim())],
  })));
  return out;
}

const footer = (fmt) => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 22 })] })] });

const pageProps = (start, fmt) => ({ page: { size: { width: PAGE_W, height: 16838 }, margin: MARGIN, pageNumbers: { start, formatType: fmt } } });

const doc = new Document({
  creator: "Batch B03",
  title: front.title,
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 120, after: 360 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 280, after: 140 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
      { id: "FigureCaption", name: "Figure Caption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 22, bold: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 } } },
      { id: "TableCaption", name: "Table Caption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 22, bold: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 120, after: 100 }, keepNext: true } },
    ],
  },
  numbering: { config: numberingConfigs },
  features: { updateFields: true },
  sections: [
    { properties: pageProps(1, NumberFormat.LOWER_ROMAN), children: titlePage() },
    {
      properties: pageProps(2, NumberFormat.LOWER_ROMAN),
      footers: { default: footer() },
      children: [
        ...certificate(),
        ...declaration(),
        ...proseSection("ACKNOWLEDGEMENT", "chapters/acknowledgement.md"),
        ...proseSection("ABSTRACT", "chapters/abstract.md"),
        pageBreak(),
        frontHeading("TABLE OF CONTENTS"),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
        ...listPage("LIST OF FIGURES", "Figure Caption"),
        ...listPage("LIST OF TABLES", "Table Caption"),
        ...abbreviations(),
      ],
    },
    {
      properties: pageProps(1, NumberFormat.DECIMAL),
      footers: { default: footer() },
      children: [...chapterParas, ...backMatter("REFERENCES", "references.md")],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2] || path.join(__dirname, "TruthLens_Project_Report.docx");
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out}: ${order.length} chapters, ${figures.length} figures, ${tables.length} tables`);
});
