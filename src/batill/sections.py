"""Conservative section grouping in extracted reading order, without rewriting text."""

import re


def heading_number(text):
    match = re.match(r'^(?:section\s+)?(\d+(?:\.\d+)*|[IVX]+)(?:[.)]?\s+)(?=\S)', text, re.I)
    return match.group(1) if match else None


def heading_name(text):
    return re.sub(r'^(?:section\s+)?(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+', '', text, flags=re.I).strip().lower()


def heading_level(block):
    match = re.search(r'<h([1-6])\b', block.get('html', ''), re.I)
    return int(match.group(1)) if match else max(1, len(block.get('section', [])))


def group_sections(blocks, title):
    """Return contiguous block groups and review flags.

    Front matter, abstract and introduction share an opening group. Numbered main
    headings override unreliable extracted HTML levels; decimal subsections stay
    with their parent. Unnumbered papers use the introduction's heading level.
    No prose is discarded. Missing/uncertain structure is explicitly flagged.
    """
    headings = [b for b in blocks if b['type'] == 'SectionHeader'
                and b['text'].strip().casefold() != title.strip().casefold()]
    intros = [b for b in headings if re.match(r'^introduction\b', heading_name(b['text']))]
    numbered = any(heading_number(b['text']) is not None for b in headings)
    main_level = heading_level(intros[0]) if intros else min(
        (heading_level(b) for b in headings if heading_name(b['text']) != 'abstract'), default=1)
    flags = []
    if not intros:
        flags.append('introduction_not_detected')
    if not numbered:
        flags.append('unnumbered_heading_levels_need_review')
    groups = [{'name': 'Opening: title, abstract and introduction', 'blocks': []}]
    seen_intro = False
    for block in blocks:
        text = block['text'].strip()
        if block['type'] == 'SectionHeader' and text.casefold() != title.strip().casefold():
            name = heading_name(text)
            number = heading_number(text)
            opening = name == 'abstract' or bool(re.match(r'^introduction\b', name))
            appendix = bool(re.match(r'^(?:appendix|appendices|online appendix|supplement)\b', name))
            if numbered:
                boundary = (number is not None and '.' not in number) or appendix
            else:
                boundary = heading_level(block) <= main_level or appendix
            # Before a detected introduction, all front matter belongs to the opening.
            before_intro = bool(intros) and not seen_intro
            if intros and block['id'] == intros[0]['id']:
                seen_intro = True
            if boundary and not opening and not before_intro:
                groups.append({'name': text, 'blocks': []})
        groups[-1]['blocks'].append(block)
    groups = [g for g in groups if g['blocks']]
    if len(groups) < 2:
        flags.append('no_section_boundaries_detected')
    return groups, flags
