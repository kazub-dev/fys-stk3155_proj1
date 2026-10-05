"""Render report.md to report.pdf using only matplotlib (mathtext for LaTeX).

LLM-assisted: GitHub Copilot GPT-6 Astra (2026-10-05), Level 4 substantial generation.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report.md"
OUTPUT = ROOT / "report.pdf"

PAGE_W, PAGE_H = 8.27, 11.69  # A4, inches
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 1.0, 0.9, 0.95
TEXT_W = PAGE_W - 2 * MARGIN_X
BODY_SIZE = 10.5
TABLE_SIZE = 9.5
CAPTION_SIZE = 9.5
LEADING = 1.38
PARAGRAPH_GAP = 0.09
HEADINGS = {1: (20, 0.10, 0.18), 2: (15, 0.26, 0.10), 3: (12.5, 0.18, 0.07),
            4: (11, 0.14, 0.05)}  # size, space before, space after

matplotlib.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "text.hinting": "no_hinting",
    "text.hinting_factor": 1,
})
CODE_FONT = "DejaVu Sans Mono"


@dataclass
class Token:
    text: str
    kind: str = "text"  # text, math, code, br
    bold: bool = False
    italic: bool = False
    space_before: bool = True


def normalize_math(expression):
    """Adapt LaTeX to matplotlib mathtext."""
    expression = " ".join(expression.split())
    expression = expression.replace(r"\operatorname", r"\mathrm")
    return re.sub(r"\\(mathbf|mathbb|mathcal|mathrm)\s*(\\[A-Za-z]+|[A-Za-z0-9])",
                  r"\\\1{\2}", expression)


INLINE = re.compile(r"\\\((.+?)\\\)|`([^`]+)`|(\*\*\*|\*\*|\*)", re.S)


def parse_inline(source):
    """Split Markdown inline text into styled tokens."""
    tokens, bold, italic, pending_space = [], False, False, False

    def add_text(text):
        nonlocal pending_space
        for piece in re.split(r"(\s+|\x00)", text.replace("--", "\u2013")):
            if piece == "\x00":
                tokens.append(Token("", "br"))
                pending_space = False
            elif not piece:
                continue
            elif piece.isspace():
                pending_space = True
            else:
                tokens.append(Token(piece, "text", bold, italic, pending_space))
                pending_space = False

    position = 0
    for match in INLINE.finditer(source):
        add_text(source[position:match.start()])
        position = match.end()
        if match.group(1) is not None:
            tokens.append(Token("$" + normalize_math(match.group(1)) + "$", "math",
                                bold, italic, pending_space))
            pending_space = False
        elif match.group(2) is not None:
            for index, word in enumerate(match.group(2).split()):
                tokens.append(Token(word, "code", False, False,
                                    pending_space if index == 0 else True))
                pending_space = False
        else:
            marker = match.group(3)
            if marker in ("**", "***"):
                bold = not bold
            if marker in ("*", "***"):
                italic = not italic
    add_text(source[position:])
    return tokens


class Measurer:
    # Measure at high resolution: low-dpi raster hinting distorts glyph widths.
    DPI = 720

    def __init__(self):
        self.figure = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=self.DPI)
        self.renderer = self.figure.canvas.get_renderer()
        self.cache = {}

    def text_kwargs(self, token, size):
        if token.kind == "code":
            return {"fontsize": size * 0.88, "family": CODE_FONT, "parse_math": False}
        return {"fontsize": size, "fontweight": "bold" if token.bold else "normal",
                "fontstyle": "italic" if token.italic else "normal",
                "parse_math": token.kind == "math"}

    def measure(self, token, size):
        """Return width, ascent and descent in inches."""
        key = (token.text, token.kind, token.bold, token.italic, size)
        if key not in self.cache:
            artist = self.figure.text(0, 0, token.text, va="baseline",
                                      **self.text_kwargs(token, size))
            box = artist.get_window_extent(self.renderer)
            artist.remove()
            self.cache[key] = (box.width / self.DPI, max(box.y1, 0) / self.DPI,
                               max(-box.y0, 0) / self.DPI)
        return self.cache[key]

    def space(self, size):
        double = self.measure(Token("n n"), size)[0]
        return double - self.measure(Token("nn"), size)[0]


MEASURE = Measurer()


def words_of(tokens):
    """Group tokens that are not separated by whitespace (break only between words)."""
    words = []
    for token in tokens:
        if token.kind == "br":
            words.append("br")
        elif words and words[-1] != "br" and not token.space_before:
            words[-1].append(token)
        else:
            words.append([token])
    return words


def layout(tokens, width, size):
    """Greedy line breaking; returns lines of (words, natural width, ascent, descent, hard_break)."""
    space = MEASURE.space(size)
    lines, current, current_width = [], [], 0.0

    def finish(hard):
        nonlocal current, current_width
        ascent = max((MEASURE.measure(t, size)[1] for word in current for t in word), default=0)
        descent = max((MEASURE.measure(t, size)[2] for word in current for t in word), default=0)
        lines.append((current, current_width, ascent, descent, hard))
        current, current_width = [], 0.0

    for word in words_of(tokens):
        if word == "br":
            finish(True)
            continue
        word_width = sum(MEASURE.measure(t, size)[0] for t in word)
        extra = word_width + (space if current else 0)
        if current and current_width + extra > width:
            finish(False)
            extra = word_width
        current.append(word)
        current_width += extra
    if current:
        finish(True)
    return lines


def line_metrics(line, size):
    _, _, ascent, descent, _ = line
    above = max(ascent, 0.78 * size * LEADING / 72)
    below = max(descent, 0.22 * size * LEADING / 72)
    return above, below


class Document:
    def __init__(self, pdf):
        self.pdf = pdf
        self.figure = None
        self.page = 0
        self.y = 0.0
        self.new_page()

    def new_page(self):
        if self.figure is not None:
            self.pdf.savefig(self.figure, dpi=200)
            plt.close(self.figure)
        self.page += 1
        self.figure = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=72)
        self.figure.text(0.5, 0.45 / PAGE_H, str(self.page), ha="center",
                         va="baseline", fontsize=9)
        self.y = MARGIN_TOP

    def close(self):
        self.pdf.savefig(self.figure, dpi=200)
        plt.close(self.figure)

    def remaining(self):
        return PAGE_H - MARGIN_BOTTOM - self.y

    def ensure(self, height):
        if height > self.remaining() and self.y > MARGIN_TOP + 1e-6:
            self.new_page()

    def to_figure(self, x, y):
        return x / PAGE_W, 1 - y / PAGE_H

    def draw_token(self, token, x, baseline, size):
        fx, fy = self.to_figure(x, baseline)
        self.figure.text(fx, fy, token.text, va="baseline",
                         **MEASURE.text_kwargs(token, size))

    def draw_line(self, line, x, baseline, width, size, align="left", justify=False):
        words, natural, _, _, hard = line
        space = MEASURE.space(size)
        gaps = len(words) - 1
        extra_gap = 0.0
        if justify and not hard and gaps > 0:
            extra_gap = (width - natural) / gaps
            if extra_gap > 3 * space:
                extra_gap = 0.0
        elif align == "center":
            x += (width - natural) / 2
        elif align == "right":
            x += width - natural
        for word in words:
            for token in word:
                self.draw_token(token, x, baseline, size)
                x += MEASURE.measure(token, size)[0]
            x += space + extra_gap

    def paragraph(self, tokens, size=BODY_SIZE, indent=0.0, align="left", justify=True,
                  gap_after=PARAGRAPH_GAP, marker=None, marker_indent=0.0):
        width = TEXT_W - indent
        for index, line in enumerate(layout(tokens, width, size)):
            above, below = line_metrics(line, size)
            self.ensure(above + below)
            baseline = self.y + above
            if index == 0 and marker is not None:
                self.draw_token(marker, MARGIN_X + marker_indent, baseline, size)
            self.draw_line(line, MARGIN_X + indent, baseline, width, size, align, justify)
            self.y = baseline + below
        self.y += gap_after

    def paragraph_height(self, tokens, size, indent=0.0):
        return sum(sum(line_metrics(line, size))
                   for line in layout(tokens, TEXT_W - indent, size))

    def heading(self, level, text):
        size, before, after = HEADINGS.get(level, HEADINGS[4])
        tokens = parse_inline(text)
        for token in tokens:
            token.bold = True
        if self.y > MARGIN_TOP + 1e-6:
            self.y += before
        self.ensure(self.paragraph_height(tokens, size) + 0.7)  # keep with next block
        self.paragraph(tokens, size, align="center" if level == 1 else "left",
                       justify=False, gap_after=after)

    def display_math(self, expression):
        expression = normalize_math(expression).replace(r"\frac", r"\dfrac")
        token = Token("$" + expression + "$", "math")
        size = BODY_SIZE * 1.12
        width, ascent, descent = MEASURE.measure(token, size)
        self.ensure(ascent + descent + 0.2)
        baseline = self.y + 0.08 + ascent
        self.draw_token(token, MARGIN_X + (TEXT_W - width) / 2, baseline, size)
        self.y = baseline + descent + 0.14

    def image(self, path, caption_tokens=None):
        picture = mpimg.imread(ROOT / path)
        aspect = picture.shape[0] / picture.shape[1]
        width = TEXT_W * 0.88
        height = width * aspect
        if height > 4.3:
            height, width = 4.3, 4.3 / aspect
        caption_height = (self.paragraph_height(caption_tokens, CAPTION_SIZE, 0.3)
                          if caption_tokens else 0.0)
        self.ensure(height + caption_height + 0.15)
        x = MARGIN_X + (TEXT_W - width) / 2
        fx, fy = self.to_figure(x, self.y + 0.05 + height)
        axis = self.figure.add_axes([fx, fy, width / PAGE_W, height / PAGE_H])
        axis.imshow(picture, interpolation="none")
        axis.axis("off")
        self.y += height + 0.12
        if caption_tokens:
            self.paragraph(caption_tokens, CAPTION_SIZE, indent=0.3, gap_after=0.15)
            self.y -= 0.0

    def rule(self, y, x0, x1, width):
        x_start, y_figure = self.to_figure(x0, y)
        x_end, _ = self.to_figure(x1, y)
        self.figure.add_artist(Line2D([x_start, x_end], [y_figure, y_figure],
                                      transform=self.figure.transFigure, lw=width,
                                      color="black"))

    def table(self, rows, alignments):
        size, pad_x, pad_y = TABLE_SIZE, 0.07, 0.04
        cells = [[parse_inline(cell) for cell in row] for row in rows]
        for token in (t for cell in cells[0] for t in cell):
            token.bold = True
        columns = len(alignments)
        space = MEASURE.space(size)

        def natural(cell):
            words = [w for w in words_of(cell) if w != "br"]
            widths = [sum(MEASURE.measure(t, size)[0] for t in word) for word in words]
            return sum(widths) + space * max(len(widths) - 1, 0), max(widths, default=0)

        stats = [[natural(cell) for cell in row] for row in cells]
        natural_widths = [max(row[c][0] for row in stats) for c in range(columns)]
        minimum_widths = [max(row[c][1] for row in stats) for c in range(columns)]
        available = TEXT_W - 2 * pad_x * columns
        widths = natural_widths[:]
        if sum(widths) > available:
            fixed = set()
            while True:
                flexible = [c for c in range(columns) if c not in fixed]
                budget = available - sum(widths[c] for c in fixed)
                share = budget / len(flexible)
                newly_fixed = {c for c in flexible if natural_widths[c] <= share}
                if not newly_fixed:
                    total = sum(natural_widths[c] for c in flexible)
                    for c in flexible:
                        widths[c] = max(minimum_widths[c], budget * natural_widths[c] / total)
                    break
                fixed |= newly_fixed
        table_width = sum(widths) + 2 * pad_x * columns
        left = MARGIN_X + (TEXT_W - table_width) / 2
        right = left + table_width

        laid_out = [[layout(cell, widths[c], size) for c, cell in enumerate(row)]
                    for row in cells]

        def row_height(row_lines):
            return max(sum(sum(line_metrics(line, size)) for line in lines)
                       for lines in row_lines) + 2 * pad_y

        def draw_row(row_lines):
            x = left
            for column, lines in enumerate(row_lines):
                y = self.y + pad_y
                for line in lines:
                    above, below = line_metrics(line, size)
                    self.draw_line(line, x + pad_x, y + above, widths[column], size,
                                   align=alignments[column])
                    y += above + below
                x += widths[column] + 2 * pad_x
            self.y += row_height(row_lines)

        header_height = row_height(laid_out[0])
        self.y += 0.06
        self.ensure(header_height + row_height(laid_out[1]) + 0.05)
        self.rule(self.y, left, right, 0.9)
        draw_row(laid_out[0])
        self.rule(self.y, left, right, 0.5)
        for row_lines in laid_out[1:]:
            if row_height(row_lines) > self.remaining():
                self.rule(self.y, left, right, 0.9)
                self.new_page()
                self.rule(self.y, left, right, 0.9)
                draw_row(laid_out[0])
                self.rule(self.y, left, right, 0.5)
            draw_row(row_lines)
        self.rule(self.y, left, right, 0.9)
        self.y += 0.16


def split_table_row(line):
    """Split a Markdown table row on pipes outside inline math."""
    line = line.strip().strip("|")
    cells, current, depth, index = [], "", 0, 0
    while index < len(line):
        if line.startswith(r"\(", index):
            depth += 1
        elif line.startswith(r"\)", index):
            depth -= 1
        if line[index] == "|" and depth == 0:
            cells.append(current.strip())
            current = ""
        else:
            current += line[index]
        index += 1
    cells.append(current.strip())
    return cells


BLOCK_START = re.compile(r"^(#{1,6}\s|\$\$|\||!\[|\s*([-*]|\d+\.)\s+)")
LIST_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)")


def parse_blocks(markdown):
    lines = markdown.splitlines()
    blocks, index = [], 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
        elif heading := re.match(r"^(#{1,6})\s+(.*)", line):
            blocks.append(("heading", len(heading.group(1)), heading.group(2).strip()))
            index += 1
        elif stripped.startswith("$$"):
            content = stripped[2:]
            if content.endswith("$$"):
                blocks.append(("math", content[:-2]))
                index += 1
                continue
            index += 1
            while index < len(lines) and not lines[index].strip().endswith("$$"):
                content += " " + lines[index]
                index += 1
            content += " " + lines[index].strip()[:-2]
            blocks.append(("math", content))
            index += 1
        elif image := re.match(r"^!\[(.*?)\]\((.*?)\)", stripped):
            blocks.append(("image", image.group(2)))
            index += 1
        elif stripped.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_table_row(lines[index]))
                index += 1
            alignments = []
            for cell in rows[1]:
                cell = cell.strip()
                if cell.startswith(":") and cell.endswith(":"):
                    alignments.append("center")
                elif cell.endswith(":"):
                    alignments.append("right")
                else:
                    alignments.append("left")
            blocks.append(("table", [rows[0]] + rows[2:], alignments))
        elif re.match(r"^(-{3,}|\*{3,})$", stripped):
            blocks.append(("rule",))
            index += 1
        elif item := LIST_ITEM.match(line):
            marker, text = item.group(2), [item.group(3)]
            index += 1
            while (index < len(lines) and lines[index].strip()
                   and not BLOCK_START.match(lines[index])):
                text.append(lines[index].strip())
                index += 1
            blocks.append(("item", marker, " ".join(text)))
        else:
            text = []
            while (index < len(lines) and lines[index].strip()
                   and not (text and BLOCK_START.match(lines[index]))):
                current = lines[index]
                text.append(current.rstrip() + ("\x00" if current.endswith("  ") else ""))
                index += 1
            joined = " ".join(text).replace("\x00 ", "\x00").rstrip("\x00")
            blocks.append(("paragraph", joined))
    return blocks


def render(blocks, pdf):
    document = Document(pdf)
    index, after_title = 0, False
    while index < len(blocks):
        block = blocks[index]
        kind = block[0]
        if kind == "heading":
            document.heading(block[1], block[2])
            after_title = block[1] == 1
        elif kind == "paragraph":
            tokens = parse_inline(block[1])
            if after_title:
                document.paragraph(tokens, align="center", justify=False, gap_after=0.2)
            else:
                document.paragraph(tokens)
            after_title = False
        elif kind == "math":
            document.display_math(block[1])
        elif kind == "image":
            caption = None
            if (index + 1 < len(blocks) and blocks[index + 1][0] == "paragraph"
                    and blocks[index + 1][1].startswith("*Figure")):
                caption = parse_inline(blocks[index + 1][1])
                index += 1
            document.image(block[1], caption)
        elif kind == "table":
            document.table(block[1], block[2])
        elif kind == "rule":
            document.y += 0.08
            document.rule(document.y, MARGIN_X, MARGIN_X + TEXT_W, 0.5)
            document.y += 0.12
        elif kind == "item":
            marker = "\u2022" if block[1] in "-*" else block[1]
            next_is_item = index + 1 < len(blocks) and blocks[index + 1][0] == "item"
            document.paragraph(parse_inline(block[2]), indent=0.35,
                               marker=Token(marker), marker_indent=0.12,
                               gap_after=0.03 if next_is_item else PARAGRAPH_GAP)
        index += 1
    document.close()


def main():
    blocks = parse_blocks(SOURCE.read_text(encoding="utf-8"))
    with PdfPages(OUTPUT) as pdf:
        info = pdf.infodict()
        info["Title"] = "Polynomial Regression for the Runge Function"
        info["Author"] = "Katarzyna Anna Zubowicz"
        render(blocks, pdf)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
