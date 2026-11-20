# Opens Final_Report.docx in the installed word processor (Word or WPS, through the Word.Application COM interface),
# refreshes the table of contents and the lists of figures and tables, saves the DOCX and exports the PDF.
param(
    [string]$Docx = (Join-Path $PSScriptRoot "Final_Report.docx"),
    [string]$Pdf = (Join-Path $PSScriptRoot "Final_Report.pdf")
)
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::GetFullPath($Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)
    $doc.Fields.Update() | Out-Null
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) { $doc.TablesOfContents.Item($i).Update() }
    $doc.Fields.Update() | Out-Null   # page numbers settle after the lists grow
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) { $doc.TablesOfContents.Item($i).UpdatePageNumbers() }
    "TOCs: " + $doc.TablesOfContents.Count + "; pages: " + $doc.ComputeStatistics(2)
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)
    "pdf: $Pdf"
    $doc.Close(0)
}
finally {
    try { $word.Quit() } catch { }
}
