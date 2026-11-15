param([string]$In, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($In, $false, $false)
  foreach ($t in $doc.TablesOfContents) { $t.Update() }
  $doc.Fields.Update() | Out-Null
  foreach ($t in $doc.TablesOfContents) { $t.Update() }
  $doc.Save()
  $doc.ExportAsFixedFormat($Pdf, 17)
  Write-Output ("pages=" + $doc.ComputeStatistics(2))
  $doc.Close($false)
} finally { $word.Quit() }
