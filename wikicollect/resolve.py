"""Resolve stage: English titles -> Wikidata QIDs -> titles in every language.

Only items with an article in all languages become triples."""
from wikicollect.api import WikiAPI, chunks, query_pages


def resolve(api: WikiAPI, seeds: list[dict], langs: list[str]) -> list[dict]:
    title_to_qid: dict[str, str] = {}
    titles = [s["title_en"] for s in seeds]
    for i, batch in enumerate(chunks(titles, 50)):
        for t, page in query_pages(api, "en", batch, prop="pageprops",
                                   ppprop="wikibase_item", redirects=1).items():
            qid = (page or {}).get("pageprops", {}).get("wikibase_item")
            if qid:
                title_to_qid[t] = qid
        print(f"  en titles -> QIDs: {min((i + 1) * 50, len(titles))}/{len(titles)}", end="\r")
    print()

    sitelinks: dict[str, dict[str, str]] = {}
    qids = sorted(set(title_to_qid.values()))
    for i, batch in enumerate(chunks(qids, 50)):
        data = api.get("www.wikidata.org", action="wbgetentities", ids="|".join(batch),
                       props="sitelinks", sitefilter="|".join(f"{l}wiki" for l in langs))
        for qid, ent in data.get("entities", {}).items():
            links = ent.get("sitelinks", {})
            sitelinks[qid] = {l: links[f"{l}wiki"]["title"] for l in langs if f"{l}wiki" in links}
        print(f"  QIDs -> sitelinks: {min((i + 1) * 50, len(qids))}/{len(qids)}", end="\r")
    print()

    triples, seen = [], set()
    for s in seeds:
        qid = title_to_qid.get(s["title_en"])
        if not qid or qid in seen or len(sitelinks.get(qid, {})) != len(langs):
            continue
        seen.add(qid)
        triples.append({"qid": qid, "titles": sitelinks[qid], "topic": s["topic"],
                        "subtopic": s["subtopic"], "seed_title": s["title_en"]})
    return triples
