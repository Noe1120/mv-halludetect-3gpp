"""整理 IJMLC 投稿文件包：submission_IJMLC/。

用法（先运行 build_paper.py 和 build_latex.py）：
  python scripts/make_submission.py

生成：
  submission_IJMLC/latex/           LaTeX 源文件（main.tex、.bib、.bbl、模板、5 个图），可直接上传
  submission_IJMLC/latex_source.zip 同上的压缩包（投稿系统支持整包上传时用）
  submission_IJMLC/main.pdf         编译好的 PDF，用来和投稿系统生成的 PDF 对照
  submission_IJMLC/manuscript.docx  Word 版（备用）
  submission_IJMLC/cover_letter.docx / .txt
  submission_IJMLC/投稿系统填写内容.txt  标题、摘要、关键词、声明等，填表时直接复制
最后在一个临时目录里重新编译 latex/，确认文件包是完整的。
"""
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from _common import ROOT

LATEX = ROOT / "paper" / "latex"
SEC = ROOT / "paper" / "sections"
OUT = ROOT / "submission_IJMLC"
SOURCES = ["main.tex", "references.bib", "main.bbl", "sn-jnl.cls", "sn-basic.bst",
           "fig_framework.pdf", "fig_mv.pdf", "fig_contrast.pdf", "fig_riskcov.pdf", "fig_cost.pdf"]

COVER = """Shengming Ni
School of Computer Science, School of Software and School of Cyberspace Security
Nanjing University of Posts and Telecommunications, Nanjing 210023, China
E-mail: b25041225@njupt.edu.cn | ORCID: 0009-0007-7156-8276

{date}

To the Editor-in-Chief
International Journal of Machine Learning and Cybernetics

Dear Editor,

I am submitting the manuscript "{title}" for consideration as an original research article in the International Journal of Machine Learning and Cybernetics.

Context. Engineers increasingly use retrieval-augmented large language models (LLMs) to look up procedures and parameters in 3GPP specifications, the standards behind mobile networks. Answers in this domain must be exact, yet in our experiments a retrieval-augmented LLM still answers about one standards question in five incorrectly, and a wrong answer reads as fluently as a right one. A practical assistant therefore needs a per-answer risk score so that it can withhold the answers it is likely to get wrong. The two common black-box options, asking an LLM to judge the answer and checking agreement across repeated samples, work poorly here: the model is often confidently wrong, and 63% of its errors on our validation questions recur identically in five samples.

Contribution. The manuscript proposes to perturb the evidence instead of the sampler. The same model answers each question closed-book and with lexical, dense and hybrid retrieval, and disagreement among these answers is fused, by logistic regression, with option-level evidence checks from two small models that run on a laptop GPU. On 2,500 held-out TeleQnA questions:
- the detector reaches an AUROC of 0.790, compared with 0.699 for an LLM judge and 0.700 for five-sample self-consistency, and both paired bootstrap intervals exclude zero;
- agreement across evidence views alone, without any training, reaches 0.737, a direct comparison of evidence perturbation with sampling;
- a lighter variant needs one short extra API call, costs about one sixth of the judge and reaches 0.762, and abstaining on the riskiest 20% of questions raises accuracy from 80.8% to 89.2%.
All weights and thresholds were fit on a separate validation split, the detector transfers without retraining to a larger model of the same family, and the manuscript also reports the experiments that did not help, with an analysis of their causes.

Fit with the journal. The journal's scope includes foundation models, trustworthy AI and data-driven intelligent systems in applied domains. This work addresses trustworthiness of foundation models in a concrete engineering application. Its methods, uncertainty estimation from model disagreement, learned fusion of heterogeneous signals and selective prediction, are general machine learning techniques, while the evaluation reflects a real deployment setting with API-only models, a fixed cost budget and an exact-answer domain. I therefore believe the manuscript will interest the journal's readers working on reliable machine learning systems as well as those applying LLMs in engineering.

This manuscript is original, has not been published previously and is not under consideration for publication elsewhere. The author has no competing interests to declare, and no funding was received for this work. The use of an AI assistant for writing the experiment code and drafting the English text is declared in Sect. 4.9 of the manuscript. The code, per-question outputs and cached model responses will be made publicly available upon acceptance.

Thank you for considering this submission.

Sincerely,
Shengming Ni
"""


def para(md, label):
    m = re.search(r"^\*\*" + re.escape(label) + r"\*\*\.? (.+)$", md, re.M)
    return m.group(1).strip() if m else ""


def plain(s):
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", s)
    return s.replace("\\*", "*")


def main():
    from datetime import date

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "latex").mkdir(parents=True)
    for f in SOURCES:
        shutil.copyfile(LATEX / f, OUT / "latex" / f)
    with zipfile.ZipFile(OUT / "latex_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in SOURCES:
            z.write(OUT / "latex" / f, f)
    shutil.copyfile(LATEX / "main.pdf", OUT / "main.pdf")
    shutil.copyfile(ROOT / "paper" / "manuscript_draft.docx", OUT / "manuscript.docx")

    front = (SEC / "01_front_intro_related.md").read_text(encoding="utf-8")
    decl = (SEC / "07_discussion.md").read_text(encoding="utf-8")
    method = (SEC / "04_method.md").read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", front, re.M).group(1)
    abstract = re.search(r"^\*\*Abstract\*\* (.+)$", front, re.M).group(1)
    keywords = [k.strip() for k in re.search(r"^\*\*Keywords\*\* (.+)$", front, re.M).group(1).split("·")]
    ai = re.search(r"^\*\*Use of AI tools\.\*\* (.+)$", method, re.M).group(1)

    letter = COVER.format(date=date.today().strftime("%d %B %Y").lstrip("0"), title=title)
    (OUT / "cover_letter.txt").write_text(letter, encoding="utf-8")
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)
    for block in letter.split("\n\n"):
        text = []
        for ln in block.split("\n"):
            if ln.startswith("- "):
                if text:
                    doc.add_paragraph("\n".join(text))
                    text = []
                doc.add_paragraph(ln[2:], style="List Bullet")
            else:
                text.append(ln)
        if text:
            # 标题页的地址行保留换行；正文段落中项目符号之后的那句话单独成段
            doc.add_paragraph("\n".join(text))
    doc.save(OUT / "cover_letter.docx")

    fields = [
        ("文章类型 Article type", "Original Article / Research Article（按系统里的选项选）"),
        ("标题 Title", title),
        ("短标题 Running title", "Evidence Perturbation for Hallucination Detection in 3GPP RAG"),
        ("摘要 Abstract（" + str(len(abstract.split())) + " 词）", abstract),
        ("关键词 Keywords（6 个，逐个填）", "\n".join(keywords)),
        ("作者 Author", "Shengming Ni（Given name: Shengming；Family name: Ni）；通讯作者"),
        ("单位 Affiliation", "School of Computer Science, School of Software and School of Cyberspace Security, "
                            "Nanjing University of Posts and Telecommunications, Nanjing 210023, China"),
        ("邮箱 E-mail", "b25041225@njupt.edu.cn"),
        ("ORCID", "0009-0007-7156-8276"),
        ("资助 Funding", plain(para(decl, "Funding"))),
        ("利益冲突 Competing interests", plain(para(decl, "Competing interests"))),
        ("作者贡献 Author contributions", plain(para(decl, "Author contributions"))),
        ("数据可用性 Data availability", plain(para(decl, "Data availability"))),
        ("代码可用性 Code availability", plain(para(decl, "Code availability"))),
        ("伦理 Ethics approval", plain(para(decl, "Ethics approval"))),
        ("知情同意 Consent to participate", plain(para(decl, "Consent to participate"))),
        ("发表同意 Consent for publication", plain(para(decl, "Consent for publication"))),
        ("AI 使用说明 Use of AI tools（如系统单独询问）", plain(ai)),
        ("预印本 Preprint / In Review", "不公开（录用前不公开稿件全文）"),
        ("推荐审稿人 Suggested reviewers", "可不填"),
        ("出版模式 Publishing model", "Subscription（订阅模式，不收费）；不要选 Open Access（约 £2690）"),
    ]
    (OUT / "投稿系统填写内容.txt").write_text(
        "\n\n".join(f"【{k}】\n{v}" for k, v in fields) + "\n", encoding="utf-8")

    # 在临时目录里只用文件包里的文件重新编译，确认包是完整的
    with tempfile.TemporaryDirectory(dir="D:/claude-home/cache") as tmp:
        for f in SOURCES:
            shutil.copyfile(OUT / "latex" / f, Path(tmp) / f)
        cmd = ["pdflatex", "--disable-installer", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        rc = [subprocess.run(c, cwd=tmp, capture_output=True, timeout=300).returncode
              for c in (cmd, ["bibtex", "main"], cmd, cmd)]
        log = (Path(tmp) / "main.log").read_text(encoding="utf-8", errors="replace")
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
        undef = re.findall(r"(?:Citation|Reference) `[^']+' .*undefined", log)
    print("文件包:", OUT)
    print("独立编译返回码:", rc, "| 页数:", pages.group(1) if pages else "失败", "| 未定义引用:", len(undef))


if __name__ == "__main__":
    main()
