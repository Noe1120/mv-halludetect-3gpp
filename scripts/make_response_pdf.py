"""把对技术检查的回复写成 PDF（投稿系统的回复上传框只接受 PDF）：submission_AnnTelecom/Response_to_technical_check.pdf。

用法：
  python scripts/make_response_pdf.py

用 sn-jnl 同样的字体（Computer Modern）排版成一页信件，编译后检查页数。
"""
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from _common import ROOT

OUT = ROOT / "submission_AnnTelecom" / "Response_to_technical_check.pdf"
TITLE = ("Perturbing the Evidence, Not the Sampler: Hallucination Detection for Retrieval-Augmented "
         "Question Answering over 3GPP Specifications")

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\pagestyle{empty}
\begin{document}

\noindent Shengming Ni\\
School of Computer Science, School of Software and School of Cyberspace Security\\
Nanjing University of Posts and Telecommunications, Nanjing 210023, China\\
E-mail: b25041225@njupt.edu.cn

\bigskip
\noindent %(date)s

\bigskip
\noindent Editorial Office\\
\emph{Annals of Telecommunications}

\bigskip
\noindent \textbf{Re: Response to the technical check}\\
\noindent Manuscript: ``%(title)s''

\bigskip
Dear Editorial Office,

Thank you for the technical check of my submission. I have addressed both points as follows.

\begin{enumerate}
\item \textbf{Table 4 was not cited in the text.} Table 4 is now cited at the start of Sect.~5.4 (Ablations), immediately
before the table: ``Table 4 removes one component of MVLV at a time.''
\item \textbf{Fig.~2 was not cited in the text.} Fig.~2 is now cited in Sect.~4.3 (Multi-View Answering), where the
training-free score MV-vote is defined and before the figure appears: ``Fig.~2 shows an example in which two of the four
views disagree with the host answer.''
\end{enumerate}

I have also checked all other figures and tables. Figs.~1--5 and Tables~1--4 are now cited in the text in numerical
order, and each is placed after its first citation. The supplementary tables (Tables~S1 and S2) are likewise cited in
Online Resource~1. No other content of the manuscript has been changed.

Thank you for your help.

\bigskip
Sincerely,\\
Shengming Ni

\end{document}
"""


def main():
    tex = TEX % {"date": date.today().strftime("%d %B %Y").lstrip("0"), "title": TITLE}
    with tempfile.TemporaryDirectory(dir="D:/claude-home/cache") as tmp:
        (Path(tmp) / "response.tex").write_text(tex, encoding="utf-8")
        cmd = ["pdflatex", "--disable-installer", "-interaction=nonstopmode", "-halt-on-error", "response.tex"]
        subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=300)
        log = (Path(tmp) / "response.log").read_text(encoding="utf-8", errors="replace")
        pages = re.search(r"Output written on response\.pdf \((\d+) page", log)
        errs = re.findall(r"^! .*$", log, re.M)
        if pages:
            shutil.copyfile(Path(tmp) / "response.pdf", OUT)
    print("回复 PDF:", OUT if pages else "未生成", "| 页数:", pages.group(1) if pages else "-", "| 错误:", errs[:3] or "无")


if __name__ == "__main__":
    main()
