"""整理 Annals of Telecommunications 投稿文件包：submission_AnnTelecom/。

用法（先运行 build_paper.py、build_latex.py 和 build_supplement.py）：
  python scripts/make_submission.py

生成：
  submission_AnnTelecom/latex/           LaTeX 源文件（main.tex、.bib、.bbl、模板、5 个图）。期刊只接受 LaTeX 投稿
  submission_AnnTelecom/latex_source.zip 同上的压缩包（投稿系统支持整包上传时用）
  submission_AnnTelecom/main.pdf         编译好的 PDF，用来和投稿系统生成的 PDF 对照
  submission_AnnTelecom/ESM_1.pdf        Online Resource 1（补充材料：提示词、实现细节、实验计划表、次要结果）
  submission_AnnTelecom/cover_letter.docx / .txt
  submission_AnnTelecom/投稿系统填写内容.txt  标题、摘要、关键词、声明等，填表时直接复制
最后在一个临时目录里重新编译 latex/，确认文件包是完整的；并检查期刊的 55,000 字符（含空格）上限。
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
OUT = ROOT / "submission_AnnTelecom"
SOURCES = ["main.tex", "references.bib", "main.bbl", "sn-jnl.cls", "sn-mathphys-num.bst",
           "fig_framework.pdf", "fig_mv.pdf", "fig_contrast.pdf", "fig_riskcov.pdf", "fig_cost.pdf"]

COVER = """Shengming Ni
School of Computer Science, School of Software and School of Cyberspace Security
Nanjing University of Posts and Telecommunications, Nanjing 210023, China
E-mail: b25041225@njupt.edu.cn | ORCID: 0009-0007-7156-8276

{date}

To the Editor-in-Chief
Annals of Telecommunications

Dear Editor,

I am submitting the manuscript "{title}" for consideration as an original research article in Annals of Telecommunications.

Context. Network engineers increasingly use large language models (LLMs) with retrieval to look up procedures, parameters and message names in 3GPP specifications, in standards assistants, network-operations copilots and test tooling. Answers in this domain must be exact, yet in our experiments a retrieval-augmented LLM still answers about one standards question in five incorrectly, and a wrong answer reads as fluently as a right one. A deployed assistant therefore needs a per-answer risk score, so that it can withhold or escalate the answers it is likely to get wrong. Telecom-specific work on LLMs and 3GPP documents, such as TeleQnA, TSpec-LLM and Telco-RAG, has focused on answer accuracy; which individual answers can be trusted has not been addressed.

Contribution. The manuscript proposes to perturb the evidence instead of resampling the model. The same LLM answers each question closed-book and with lexical, dense and hybrid retrieval over the 3GPP Release 18 corpus, and disagreement among these answers is fused with option-level evidence checks from two small models that run on a laptop GPU. On 2,500 held-out TeleQnA standards questions:
- the detector reaches an AUROC of 0.790, compared with 0.699 for an LLM judge and 0.700 for five-sample self-consistency, and both paired bootstrap intervals exclude zero;
- abstaining on the riskiest 20% of questions raises the accuracy of the answers given from 80.8% to 89.2%;
- a lighter variant needs one short extra API call, costs about one sixth of the judge and reaches 0.762.
The manuscript also explains why the method works: resampled answers repeat the model's errors almost exactly, whereas answers under changed evidence fail differently. It also reports practical findings for 3GPP retrieval, for example that BM25 matches a hybrid retriever and that routing to specification series fails without a trained router.

Fit with the journal. Annals of Telecommunications covers software, protocols and services in telecommunications. This work targets a concrete telecom use case, reliable LLM assistance for engineers who work with 3GPP standards, and is evaluated on a telecom benchmark and the 3GPP corpus under realistic constraints: API-only models, a cost budget and answers that must be exact. I therefore believe it will interest the journal's readers working on AI for network operations and standardization.

This manuscript is original, has not been published previously and is not under consideration for publication elsewhere. An earlier version was submitted to the International Journal of Machine Learning and Cybernetics and was declined without review as outside that journal's scope; the present version has been revised substantially, including a new analysis of why evidence perturbation works. The author has no competing interests to declare, and no funding was received for this work. The use of an AI assistant for writing the experiment code and drafting the English text is declared in Sect. 4.8 of the manuscript. The code, per-question outputs and cached model responses will be made publicly available upon acceptance.

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
    shutil.copyfile(ROOT / "paper" / "supplementary" / "ESM_1.pdf", OUT / "ESM_1.pdf")

    front = (SEC / "01_front_intro_related.md").read_text(encoding="utf-8")
    decl = (SEC / "07_discussion.md").read_text(encoding="utf-8")
    method = (SEC / "04_method.md").read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", front, re.M).group(1)
    abstract = re.search(r"^\*\*Abstract\*\* (.+)$", front, re.M).group(1)
    keywords = [k.strip() for k in re.search(r"^\*\*Keywords\*\* (.+)$", front, re.M).group(1).split("·")]
    ai = re.search(r"^(An AI assistant \(Claude Code.+)$", method, re.M).group(1)

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
        ("补充材料 Supplementary material", "ESM_1.pdf，文件类型选 Electronic Supplementary Material / Supplementary Material，"
                                         "正文里称为 Online Resource 1"),
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
        ("出版模式 Publishing model", "录用后在 MyPublication 表单里选订阅模式（不收费）；不要选 Open Access"),
        ("是否投过别的期刊 Previous submission", "如系统询问：曾投 International Journal of Machine Learning and Cybernetics，"
                                             "未送审即以不在范围内为由拒稿（desk reject）；本稿已大幅修改"),
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
    chars = None
    try:
        txt = subprocess.run(["pdftotext", "-enc", "UTF-8", str(OUT / "main.pdf"), "-"], capture_output=True,
                             timeout=120).stdout.decode("utf-8", errors="replace")
        chars = len(re.sub(r"\s+", " ", txt))
    except (OSError, subprocess.SubprocessError):
        pass
    print("文件包:", OUT)
    print("正文字符数（含空格，PDF 抽取）:", chars, "| 上限 55,000:", "通过" if chars and chars <= 55000 else "请检查")
    print("独立编译返回码:", rc, "| 页数:", pages.group(1) if pages else "失败", "| 未定义引用:", len(undef))


if __name__ == "__main__":
    main()
