"""Fetch the four approved full-abstract supplements; never run embedding inference.

Three publisher abstracts were reviewed against their source records. P263
uses the existing Scopus export, matched by title, authors, year and journal
because its Scopus DOI is blank. Short publisher descriptions are excluded.
Run from the repository environment; later preparations read only cached files.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from batill.storage import file_hash, write_json, read_json

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/selected_abstracts/supplements'
IDS = ['P047', 'P229', 'P263', 'P284']


def download(url, destination):
    if not destination.exists():
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        destination.write_bytes(response.content)
    return destination.read_bytes()


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    from batill.selected_abstracts import load_selected_documents
    papers, _, _, _, _ = load_selected_documents(ROOT)
    audit = papers.set_index('paper_id')
    config_path = ROOT / 'configs/selected_abstracts_supplements.json'
    config = {}
    for pid in IDS:
        paper = audit.loc[pid]
        record_path = FOLDER / f'{pid}.json'
        if record_path.exists():
            record = read_json(record_path)
        else:
            record = {'paper_id': pid, 'pdf_sha256': paper.pdf_sha256,
                      'title': paper.title, 'doi': paper.doi,
                      'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                      'source_kind': 'external_full_abstract'}
            if pid in {'P047', 'P229', 'P284'}:
                url = 'https://www.aeaweb.org/articles?id=' + paper.doi
                origin = FOLDER / f'{pid}_aea.html'
                soup = BeautifulSoup(download(url, origin), 'html.parser')
                section = soup.select_one('section.article-information.abstract')
                if section is None:
                    raise ValueError(f'No publisher abstract for {pid}')
                for h in section.find_all(['h3', 'h4']):
                    h.decompose()
                record.update(abstract=section.get_text(' ', strip=True), source_url=url,
                              source_record_id=paper.doi, retrieval_url=url)
            elif pid == 'P263':
                origin = ROOT / 'data/bibliography/All_private_equity_bib.csv'
                raw = pd.read_csv(origin, dtype=str, keep_default_na=False)
                matches = raw.loc[raw.EID.eq('2-s2.0-66249098556')]
                assert len(matches) == 1
                row = matches.iloc[0]
                assert row.Year == '2009' and 'Masulis' in row.Authors and 'Thomas' in row.Authors
                assert row.Title.casefold() == paper.title.casefold()
                record.update(abstract=row.Abstract, doi='', source_record_id=row.EID,
                              source_url='https://chicagounbound.uchicago.edu/uclrev/vol76/iss1/8/',
                              retrieval_url='local: data/bibliography/All_private_equity_bib.csv',
                              identity_note='Scopus DOI blank; 2009 Chicago Law Review article matched by title, both authors, journal, volume 76, issue 1, pages 219-259. Existing selected-work DOI refers to a working-paper version; no DOI match is claimed.')
            assert record['abstract'].strip() and '[No abstract available]' not in record['abstract']
            record.update(origin_path=str(origin.relative_to(ROOT)), origin_sha256=file_hash(origin))
            write_json(record_path, record)
        assert record['source_kind'] == 'external_full_abstract'
        assert record['pdf_sha256'] == paper.pdf_sha256 and record['paper_id'] == pid
        config[paper.pdf_sha256] = {
            'paper_id': pid, 'source_kind': record['source_kind'],
            'source_path': str(record_path.relative_to(ROOT)), 'source_sha256': file_hash(record_path),
            'identity_review': record.get('identity_note', 'Exact DOI and publication identity reviewed against the publisher/PubMed record.') + ' User approved inclusion of this full abstract; short publisher descriptions are excluded.'}
        print(pid, record['source_kind'], len(record['abstract'].split()), 'words')
    assert len(config) == 4
    write_json(config_path, config)


if __name__ == '__main__':
    main()
