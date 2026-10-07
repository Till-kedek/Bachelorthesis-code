"""Turn Marker output into inspectable blocks without pretending to repair formulas."""

import re
from bs4 import BeautifulSoup

SCHEMA_VERSION = 2
EXCLUDED_TYPES = {"PageHeader", "PageFooter", "Reference"}
REFERENCE_HEADING = re.compile(r"^(?:\d+[.\s]*)?(?:references|bibliography|literature cited)$", re.I)


def html_text(html, *, omit_math=False):
    soup = BeautifulSoup(html or "", "html.parser")
    for math in soup.find_all("math"):
        if omit_math:
            math.replace_with(" [FORMULA] ")
            continue
        delimiter = "$$" if math.get("display") == "block" else "$"
        math.replace_with(f" {delimiter}{math.get_text(' ', strip=True)}{delimiter} ")
    for row in soup.find_all("tr"):
        row.replace_with("\n" + " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])) + "\n")
    for tag in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "br"]):
        tag.insert_before("\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in soup.get_text().splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_document(raw, *, pdf_sha256, source_filename, metadata=None):
    leaves = []

    def visit(node, page=None):
        match = re.search(r"/page/(\d+)", node.get("id", ""))
        if match:
            page = int(match.group(1)) + 1
        if node.get("children"):
            for child in node["children"]:
                visit(child, page)
        else:
            leaves.append((node, page))

    for page in raw.get("children", []):
        visit(page)
    headings = {node["id"]: html_text(node.get("html")) for node, _ in leaves
                if node.get("block_type", "").split(".")[-1] == "SectionHeader"}
    blocks = []
    for node, page in leaves:
        kind = node.get("block_type", "Unknown").split(".")[-1]
        hierarchy = node.get("section_hierarchy") or {}
        section = [headings.get(value, value) for _, value in sorted(hierarchy.items(), key=lambda x: int(x[0]))]
        content = html_text(node.get("html"))
        reference = any(REFERENCE_HEADING.fullmatch(h.strip()) for h in section)
        reference = reference or (kind == "SectionHeader" and bool(REFERENCE_HEADING.fullmatch(content)))
        reason = "reference_section" if reference else (kind if kind in EXCLUDED_TYPES else None)
        blocks.append({"id": node["id"], "type": kind, "page": page,
                       "bbox": node.get("bbox"), "section": section, "text": content,
                       "html": node.get("html", ""), "include": bool(content) and reason is None,
                       "exclusion_reason": reason})
    metadata = metadata or {}
    flags = []
    if not any(b["include"] for b in blocks):
        flags.append("no_usable_text")
    if any("�" in b["text"] for b in blocks):
        flags.append("replacement_characters")
    if any(b["type"] == "Equation" or "<math" in b["html"] for b in blocks):
        flags.append("contains_detected_formulas")
    title = metadata.get("title") or next((b["text"] for b in blocks if b["type"] == "Title"), None)
    title_source = "metadata_or_title_block"
    if not title:
        title = next((html_text(b["html"], omit_math=True).rstrip("$ ") for b in blocks
                      if b["page"] == 1 and b["type"] == "SectionHeader" and "<h1>" in b["html"]), source_filename)
        title_source = "first_page_heading_or_filename (inspect)"
    return {"schema_version": SCHEMA_VERSION, "pdf_sha256": pdf_sha256,
            "source_filename": source_filename, "Key": metadata.get("Key"),
            "title": title, "title_source": title_source,
            "metadata_source": metadata.get("metadata_source"),
            "quality_flags": flags, "blocks": blocks}


def embedding_blocks(document, math_policy="preserve"):
    """Filter marked formulas without deleting currency, percentages or unmarked prose."""
    if math_policy not in {"preserve", "omit"}:
        raise ValueError("math_policy must be preserve or omit")
    result = []
    for block in document["blocks"]:
        if not block["include"] or not block["text"].strip():
            continue
        if math_policy == "omit" and block["type"] == "Equation":
            continue
        if math_policy == "omit" and re.search(r"<math\b", block.get("html", ""), re.I):
            block = {**block, "text": html_text(block["html"], omit_math=True)}
        result.append(block)
    return result


def preview(document):
    return "\n\n".join(f"[page {b['page']} | {b['type']} | {b['id']}]\n{b['text']}"
                       for b in embedding_blocks(document))
