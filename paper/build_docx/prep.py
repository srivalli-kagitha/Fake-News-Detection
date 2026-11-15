import re
s=open("truthlens_ieee.tex").read()
ROM=["I","II","III","IV","V","VI","VII","VIII","IX","X"]
# --- authors -> borderless table rows (rebuilt later in Word)
a0=s.index("\\author{"); a1=s.index("\\maketitle")
auth=s[a0:a1]
blocks=re.findall(r"\\IEEEauthorblockN\{(.*?)\}\s*\\IEEEauthorblockA\{(.*?)\}\s*(?=\\and|\}\s*$)",auth,re.S)
people=[]
for n,a in blocks:
    a=a.replace("\\textit{","").replace("}","").replace("\\phantom{email@srgec.ac.in","")
    lines=[x.strip() for x in a.split("\\\\") if x.strip()]
    people.append([n]+lines)
s=s[:a0]+s[a1:]
open("authors.txt","w").write("\n".join("|".join(p) for p in people))
# --- keywords
s=re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", lambda m:"\\noindent\\textbf{\\textit{Index Terms}---"+m.group(1).strip()+"}", s, flags=re.S)
# --- Telugu font macro
s=re.sub(r"\{\\telugufont\s+([^}]*)\}", r"\1", s)
# --- tikz figure -> image
t0=s.index("\\begin{tikzpicture}"); t1=s.index("\\end{tikzpicture}")+len("\\end{tikzpicture}")
s=s[:t0]+"\\includegraphics[width=\\columnwidth]{figures/architecture_fig1.png}"+s[t1:]
# --- numbering of sections, tables, figures, equations
labels={}; sec=0; sub=0; out=[]
tab=0; fig=0
for line in s.split("\n"):
    m=re.match(r"\\section\{(.*)\}",line)
    if m:
        sec+=1; sub=0; cur=ROM[sec-1]; line="\\section*{"+ROM[sec-1]+". "+m.group(1)+"}"
    m=re.match(r"\\subsection\{(.*)\}",line)
    if m:
        sub+=1; cur=chr(64+sub); line="\\subsection*{"+chr(64+sub)+". "+m.group(1)+"}"
    if "\\begin{table}" in line: tab+=1; cur=ROM[tab-1]
    if "\\begin{figure}" in line: fig+=1; cur=str(fig)
    m=re.search(r"\\label\{([^}]*)\}",line)
    if m:
        k=m.group(1)
        labels[k]= (ROM[sec-1]+("-"+chr(64+sub) if sub and k.startswith("sec:") and not line.strip().startswith("\\section") and sub else "")) if k.startswith("sec:") else cur
    out.append(line)
s="\n".join(out)
# section labels sit right after their heading; recompute simply
labels["sec:data"]="III"; labels["sec:verify"]="IV-D"; labels["sec:claimres"]="VI-B"
s=re.sub(r"\\label\{[^}]*\}","",s)
s=re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], s)
# captions with numbers
tabn=[0]; fign=[0]
def table_env(m):
    tabn[0]+=1; body=m.group(0)
    body=re.sub(r"\\caption\{(.*?)\}\n", lambda c:"\\caption{TABLE "+ROM[tabn[0]-1]+". "+c.group(1)+"}\n", body, count=1, flags=re.S)
    note=re.search(r"\\multicolumn\{\d+\}\{@\{\}l\}\{\\scriptsize\s*(.*?)\}\s*\n\\end\{tabular\}",body,re.S)
    if note:
        body=body.replace(note.group(0),"\\end{tabular}")+"\n\n"+note.group(1).replace("$^{a}$","\\textsuperscript{a}")+"\n"
    return body
s=re.sub(r"\\begin\{table\}.*?\\end\{table\}", table_env, s, flags=re.S)
def fig_env(m):
    fign[0]+=1
    return m.group(0).replace("\\caption{", "\\caption{Fig. "+str(fign[0])+". ", 1)
s=re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", fig_env, s, flags=re.S)
eq=[0]
def eqn(m):
    eq[0]+=1; body=m.group(1).replace("\\begin{split}","\\begin{aligned}").replace("\\end{split}","\\end{aligned}")
    return "\\begin{equation*}"+body+"\\qquad\\qquad ("+str(eq[0])+")\\end{equation*}"
s=re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eqn, s, flags=re.S)
s=s.replace("\\usepackage{fontspec}\n","").replace("\\newfontfamily\\telugufont{NotoSansTelugu-Regular.ttf}[Path=./fonts/, Script=Telugu]\n","")
s=re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", s)
s=s.replace("Climate-FEVER & Similarity","Climate-F. & Similarity")
s=s.replace("Curated$^{a$} &","Curated$^{a}$ &")
open("paper_pandoc.tex","w").write(s)
print(labels, tabn, fign, eq, people, sep="\n")
