param([string]$In, [string]$Out, [string]$Pdf, [string]$Authors)
$word = New-Object -ComObject Word.Application; $word.Visible=$false; $word.DisplayAlerts=0
function SetStyle($doc, $name, $size, $bold, $italic, $align, $before, $after, $indent) {
  try { $st = $doc.Styles.Item($name) } catch { return }
  $st.Font.Name = "Times New Roman"; $st.Font.Size = $size; $st.Font.Color = 0
  if ($bold -ne $null) { $st.Font.Bold = $bold }; if ($italic -ne $null) { $st.Font.Italic = $italic }
  $pf = $st.ParagraphFormat; $pf.Alignment = $align; $pf.SpaceBefore = $before; $pf.SpaceAfter = $after
  $pf.LineSpacingRule = 0; $pf.FirstLineIndent = $indent
}
try {
  $doc = $word.Documents.Open($In)
  $ps = $doc.Sections(1).PageSetup
  $ps.PageWidth = 612; $ps.PageHeight = 792; $ps.TopMargin = 54; $ps.BottomMargin = 72; $ps.LeftMargin = 45; $ps.RightMargin = 45
  SetStyle $doc "Normal" 10 $null $null 3 0 0 0
  SetStyle $doc "Body Text" 10 0 0 3 0 0 10
  SetStyle $doc "First Paragraph" 10 0 0 3 0 0 10
  SetStyle $doc "Compact" 8 $null $null 0 0 0 0
  SetStyle $doc "Title" 22 0 0 1 0 8 0
  SetStyle $doc "Heading 1" 10 0 0 1 8 4 0
  SetStyle $doc "Heading 2" 10 0 1 0 6 3 0
  SetStyle $doc "Abstract Title" 9 1 1 0 0 0 0
  SetStyle $doc "Abstract" 9 1 0 3 0 4 10
  SetStyle $doc "Table Caption" 8 0 0 1 6 3 0
  SetStyle $doc "Image Caption" 8 0 0 3 3 6 0
  SetStyle $doc "Captioned Figure" 10 0 0 1 4 0 0
  SetStyle $doc "Bibliography" 8 0 0 3 0 1 0
  try { $doc.Styles.Item("Heading 1").Font.SmallCaps = $true } catch {}
  foreach ($n in @("Heading 1","Heading 2","Title")) { try { $doc.Styles.Item($n).ParagraphFormat.KeepWithNext = $true } catch {} }
  # merge "Abstract" heading into the abstract paragraph, IEEE style
  # authors table after the title
  $title = $doc.Paragraphs(1).Range
  $title.InsertParagraphAfter()
  $p2 = $doc.Paragraphs(2).Range
  $people = Get-Content -Encoding UTF8 $Authors | ForEach-Object { ,($_ -split '\|') }
  $tbl = $doc.Tables.Add($p2, 2, 4)
  $tbl.Borders.Enable = $false
  $tbl.Range.Font.Name = "Times New Roman"; $tbl.Range.Font.Size = 9
  for ($i=0; $i -lt $people.Count; $i++) {
    $r = [math]::Floor($i/4)+1; $c = ($i % 4)+1
    $cell = $tbl.Cell($r,$c).Range
    $cell.Text = ($people[$i] -join [char]11)
    $cell.ParagraphFormat.Alignment = 1; $cell.ParagraphFormat.FirstLineIndent = 0; $cell.ParagraphFormat.SpaceAfter = 6
    $cell.Font.Size = 9
    $line2 = $cell.Duplicate; $line2.Find.Execute($people[$i][1]) | Out-Null; $line2.Font.Italic = $true
    $nm = $cell.Duplicate; $nm.Find.Execute($people[$i][0]) | Out-Null; $nm.Font.Size = 10
  }
  # continuous section break after the authors table -> two columns for the body
  $after = $tbl.Range; $after.Collapse(0)
  $after.InsertBreak(3)
  $doc.Sections(2).PageSetup.TextColumns.SetCount(2)
  $doc.Sections(2).PageSetup.TextColumns.Spacing = 18
  # empty paragraph left between title and table
  # images to column width
  $colW = (612 - 45 - 45 - 18) / 2
  foreach ($s in $doc.InlineShapes) { if ($s.Width -gt $colW) { $ratio = $colW / $s.Width; $s.Width = $colW; $s.Height = $s.Height * $ratio } }
  # result tables: fit to column, 8 pt
  for ($t = 2; $t -le $doc.Tables.Count; $t++) {
    $tb = $doc.Tables($t); $tb.Range.Font.Size = 8; $tb.Range.Font.Name = "Times New Roman"
    $tb.Range.ParagraphFormat.FirstLineIndent = 0; $tb.Range.ParagraphFormat.SpaceAfter = 0
    $tb.AllowAutoFit = $true; $tb.AutoFitBehavior(1)
    $tb.AutoFitBehavior(0)
    $tot = 0; for ($c=1; $c -le $tb.Columns.Count; $c++) { $tot += $tb.Columns($c).Width }
    $scale = ($colW - 4) / $tot
    for ($c=1; $c -le $tb.Columns.Count; $c++) { $tb.Columns($c).Width = $tb.Columns($c).Width * $scale }
    $tb.Rows.LeftIndent = 0
    $tb.Rows.Alignment = 1
  }
  foreach ($par in $doc.Paragraphs) { $tx = $par.Range.Text; if ($tx -like "aEffective batch*" -or $tx -like "aFull pipeline*") { $par.Range.Font.Size = 7; $par.FirstLineIndent = 0; $par.SpaceAfter = 6 } }
  $doc.SaveAs2($Out, 16)
  $doc.ExportAsFixedFormat($Pdf, 17)
  Write-Output ("pages=" + $doc.ComputeStatistics(2) + " tables=" + $doc.Tables.Count + " shapes=" + $doc.InlineShapes.Count)
  $doc.Close($false)
} finally { $word.Quit() }
