"""Inspect external supervision without training or changing the IMCS pipeline."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[2] / "outputs/58_external_evidence_data"
SOURCES = {
    "arts": {
        "repo": "zhijing-jin/ARTS_TestSet",
        "commit": "bb725433ec49a89cfed221e8b0b72a6a1053282a",
        "license": "No repository license detected; redistribution not cleared.",
        "files": ["README.md", "data/arts_testset/README.md"]
        + ["data/arts_testset/" + name + "_test_enriched.json" for name in ("laptop", "rest")]
        + ["data/src_data/" + name + "/" + split + ".json"
           for name in ("laptop", "rest") for split in ("train", "dev", "test")],
    },
    "condaqa": {
        "repo": "AbhilashaRavichander/CondaQA",
        "commit": "bd4857f1f2819937f1505925ad8c809544e8e5ef",
        "license": "Repository Apache-2.0; preserve upstream notices and source provenance.",
        "files": ["README.md", "LICENSE", "data/condaqa_train.json", "data/condaqa_dev.json"],
    },
    "negcomment": {
        "repo": "Easonsi/SSENE",
        "commit": "2e70b2beb43b00a66232da2a2de3223b3679b457",
        "license": "No repository license detected; redistribution not cleared.",
        "files": ["README.md", "data/NegComment/train.json", "data/NegComment/dev.json"],
    },
}


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch():
    session = requests.Session()
    session.headers["User-Agent"] = "IMCS21-external-supervision-check"
    manifest = {}
    for name, source in SOURCES.items():
        api = "https://api.github.com/repos/" + source["repo"]
        response = session.get(api + "/git/trees/" + source["commit"], params={"recursive": 1}, timeout=60)
        response.raise_for_status()
        tree = {item["path"]: item for item in response.json()["tree"]}
        records = []
        for relative in source["files"]:
            path = OUT / "raw" / name / relative
            if not path.exists():
                response = session.get(api + "/git/blobs/" + tree[relative]["sha"], timeout=120)
                response.raise_for_status()
                blob = response.json()
                assert blob["encoding"] == "base64", relative
                content = base64.b64decode(blob["content"])
                assert len(content) == tree[relative]["size"], relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            content = path.read_bytes()
            git_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            assert git_blob == tree[relative]["sha"], relative
            records.append({"path": relative, "bytes": len(content), "git_blob": git_blob})
            print(name, relative, len(content), flush=True)
        manifest[name] = dict(source, downloaded=records)
        dump("sources.json", manifest)


def read_json(name, relative):
    return json.loads((OUT / "raw" / name / relative).read_text(encoding="utf-8"))


def read_lines(name, relative):
    # CONDAQA and NegComment use JSON Lines despite their .json extension.
    return [json.loads(line) for line in (OUT / "raw" / name / relative)
            .read_text(encoding="utf-8").splitlines() if line.strip()]


def check_arts(issues):
    result, pairs, examples = {}, [], []
    for domain in ("laptop", "rest"):
        source = {split: read_json("arts", "data/src_data/" + domain + "/" + split + ".json")
                  for split in ("train", "dev", "test")}
        enriched = read_json("arts", "data/arts_testset/" + domain + "_test_enriched.json")
        bad_spans = []
        for key, row in enriched.items():
            if row["sentence"][row["from"]:row["to"]] != row["term"]:
                bad_spans.append(key)
                issues.append({"dataset": "arts", "domain": domain, "id": key, "type": "span_mismatch"})
        families = defaultdict(list)
        types = Counter()
        expectation_differences, target_surface_changes = [], []
        difference_types = Counter()
        for key, row in enriched.items():
            base = row["id"]
            if key == base:
                continue
            assert key.startswith(base + "_adv"), key
            edit = key[len(base) + 1:]
            assert edit in ("adv1", "adv2", "adv3"), edit
            types[edit] += 1
            if base not in enriched:
                issues.append({"dataset": "arts", "id": key, "type": "missing_original"})
                continue
            original = enriched[base]
            if row["term"] != original["term"]:
                target_surface_changes.append(key)
                issues.append({"dataset": "arts", "domain": domain, "id": key,
                               "type": "target_surface_changed", "original": original["term"],
                               "edited": row["term"]})
            changed = row["polarity"] != original["polarity"]
            if changed != (edit == "adv1"):
                expectation_differences.append(key)
                difference_types[(edit, original["polarity"], row["polarity"])] += 1
                issues.append({"dataset": "arts", "domain": domain, "id": key,
                               "type": "edit_strategy_not_label_flip", "original": original["polarity"],
                               "edited": row["polarity"]})
            pair = {"domain": domain, "family": base, "edited_id": key, "edit": edit,
                    "original_target": original["term"], "edited_target": row["term"],
                    "same_target_surface": original["term"] == row["term"],
                    "original_label": original["polarity"],
                    "edited_label": row["polarity"], "changed": changed,
                    "valid_marker_offsets": base not in bad_spans and key not in bad_spans}
            pairs.append(pair)
            families[base].append(pair)
        originals = {key: row for key, row in enriched.items() if key == row["id"]}
        source_mismatches = [key for key, row in originals.items() if key not in source["test"]
                             or any(row[f] != source["test"][key][f] for f in ("sentence", "term", "polarity"))]
        sentences = {split: {r["sentence"] for r in rows.values()} for split, rows in source.items()}
        same_text = defaultdict(set)
        for row in enriched.values():
            same_text[row["sentence"]].add((row["term"], row["from"], row["to"]))
        result[domain] = {
            "source_rows": {split: len(rows) for split, rows in source.items()},
            "enriched_rows": len(enriched), "originals": len(originals), "variants": dict(types),
            "families_with_both_changed_and_unchanged_target": sum(
                len({p["changed"] for p in group}) == 2 for group in families.values()),
            "span_mismatches": len(bad_spans),
            "edit_strategy_label_differences": len(expectation_differences),
            "strategy_difference_types": {" / ".join(k): v for k, v in difference_types.items()},
            "target_surface_changes": len(target_surface_changes),
            "original_source_mismatches": len(source_mismatches),
            "exact_sentence_overlap": {a + "_" + b: len(sentences[a] & sentences[b])
                                       for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))},
            "distinct_enriched_texts": len(same_text),
            "texts_with_multiple_annotated_targets": sum(len(v) > 1 for v in same_text.values()),
        }
        key = sorted(families)[0]
        examples.append({"domain": domain, "original": originals[key],
                         "variants": [enriched[p["edited_id"]] for p in families[key]]})
    dump("arts_pairs.json", pairs)
    dump("arts_examples.json", examples)
    return result


def check_condaqa(issues):
    result, pairs, edit_groups, examples = {}, [], [], []
    passage_sets, text_sets = {}, {}
    bounded = {"YES", "NO", "DON'T KNOW"}
    for split in ("train", "dev"):
        rows = read_lines("condaqa", "data/condaqa_" + split + ".json")
        assert len({r["SampleID"] for r in rows}) == len(rows), split
        question_groups = defaultdict(dict)
        passage_versions = defaultdict(set)
        for row in rows:
            passage_versions[(row["PassageID"], row["PassageEditID"])].add(row["sentence1"])
            group = question_groups[(row["PassageID"], row["QuestionID"])]
            assert row["PassageEditID"] not in group, (split, row["SampleID"])
            group[row["PassageEditID"]] = row
        assert all(len(texts) == 1 for texts in passage_versions.values()), split
        split_pairs, mismatched_questions, unpaired = [], 0, 0
        by_edit = defaultdict(list)
        for (passage, question), group in question_groups.items():
            if 0 not in group:
                unpaired += len(group)
                continue
            original = group[0]
            for edit, row in group.items():
                if edit == 0:
                    continue
                if original["sentence2"] != row["sentence2"]:
                    mismatched_questions += 1
                    issues.append({"dataset": "condaqa", "split": split, "id": row["SampleID"],
                                   "type": "question_changed_under_same_id"})
                    continue
                pair = {"split": split, "passage": passage, "question_id": question, "edit": edit,
                        "original_id": original["SampleID"], "edited_id": row["SampleID"],
                        "original_label": original["label"], "edited_label": row["label"],
                        "changed": original["label"] != row["label"],
                        "bounded_labels": original["label"] in bounded and row["label"] in bounded}
                split_pairs.append(pair)
                if pair["bounded_labels"]:
                    by_edit[(passage, edit)].append(pair)
        mixed = []
        for (passage, edit), group in by_edit.items():
            entry = {"split": split, "passage": passage, "edit": edit, "queries": group,
                     "has_changed_and_unchanged_queries": len({p["changed"] for p in group}) == 2}
            edit_groups.append(entry)
            if entry["has_changed_and_unchanged_queries"]:
                mixed.append(entry)
        by_id = {r["SampleID"]: r for r in rows}
        for group in mixed[:2]:
            example = dict(group)
            example["original_passage"] = by_id[group["queries"][0]["original_id"]]["sentence1"]
            example["edited_passage"] = by_id[group["queries"][0]["edited_id"]]["sentence1"]
            example["question_texts"] = [by_id[p["original_id"]]["sentence2"] for p in group["queries"]]
            examples.append(example)
        pairs.extend(split_pairs)
        passage_sets[split] = {r["PassageID"] for r in rows}
        text_sets[split] = {r["original passage"] for r in rows}
        result[split] = {
            "rows": len(rows), "passages": len(passage_sets[split]),
            "labels": dict(Counter(r["label"] for r in rows)), "question_groups": len(question_groups),
            "four_version_question_groups": sum(set(g) == {0, 1, 2, 3} for g in question_groups.values()),
            "paired_comparisons": len(split_pairs),
            "bounded_label_pairs": sum(p["bounded_labels"] for p in split_pairs),
            "unpaired_edited_rows_without_original": unpaired,
            "question_text_mismatches": mismatched_questions,
            "mixed_response_edit_groups": len(mixed),
            "distinct_passages_with_mixed_response": len({g["passage"] for g in mixed}),
            "mixed_groups_by_edit": dict(Counter(g["edit"] for g in mixed)),
            "mixed_groups_by_query_count": dict(Counter(len(g["queries"]) for g in mixed)),
            "mixed_groups_with_known_preserved_answer": sum(any(
                not p["changed"] and p["original_label"] in {"YES", "NO"}
                for p in g["queries"]) for g in mixed),
            "same_edit_passage_text_verified": True,
            "per_edit_bounded_changes": {str(edit): dict(Counter(
                "changed" if p["changed"] else "unchanged" for p in split_pairs
                if p["edit"] == edit and p["bounded_labels"])) for edit in (1, 2, 3)},
        }
    result["train_dev_passage_id_overlap"] = len(passage_sets["train"] & passage_sets["dev"])
    result["train_dev_original_text_overlap"] = len(text_sets["train"] & text_sets["dev"])
    dump("condaqa_pairs.json", pairs)
    dump("condaqa_edit_groups.json", edit_groups)
    dump("condaqa_examples.json", examples)
    return result


def check_negcomment(issues):
    result, texts, examples = {}, {}, []
    for split in ("train", "dev"):
        rows = read_lines("negcomment", "data/NegComment/" + split + ".json")
        groups = defaultdict(set)
        missing, ambiguous, triples = 0, 0, 0
        empty_subject, nonliteral_nonempty = 0, 0
        for index, row in enumerate(rows):
            for triple in row["label"]:
                assert len(triple) == 3 and all(isinstance(v, str) for v in triple), (split, index)
                triples += 1
                groups[row["text"]].add(tuple(triple))
                if any(not part or part not in row["text"] for part in triple):
                    missing += 1
                    empty_subject += not triple[0]
                    nonliteral_nonempty += all(triple) and any(part not in row["text"] for part in triple)
                    issues.append({"dataset": "negcomment", "split": split, "row": index,
                                   "type": "triple_part_not_literal", "triple": triple})
                elif any(row["text"].count(part) > 1 for part in triple):
                    ambiguous += 1
        texts[split] = set(groups)
        multi = [(text, sorted(group)) for text, group in groups.items() if len(group) > 1]
        examples.extend({"split": split, "text": text, "triples": group} for text, group in multi[:3])
        result[split] = {
            "rows": len(rows), "domains": dict(Counter(r["type"] for r in rows)),
            "triples": triples, "row_triple_counts": dict(Counter(len(r["label"]) for r in rows)),
            "distinct_texts": len(groups), "duplicate_text_rows": len(rows) - len(groups),
            "texts_with_multiple_triples": len(multi),
            "triples_with_nonliteral_or_empty_part": missing,
            "triples_with_empty_subject": empty_subject,
            "nonempty_triples_with_nonliteral_part": nonliteral_nonempty,
            "triples_with_ambiguous_string_positions": ambiguous,
            "token_reconstruction_mismatches": sum("".join(r["token"]) != r["text"] for r in rows),
        }
    result["train_dev_exact_text_overlap"] = len(texts["train"] & texts["dev"])
    dump("negcomment_examples.json", examples)
    return result


def response_counts(before_gold, after_gold, before_pred, after_pred):
    assert len(before_gold) == len(after_gold) == len(before_pred) == len(after_pred)
    changed, preserved, correct_changed, correct_preserved = 0, 0, 0, 0
    for y0, y1, p0, p1 in zip(before_gold, after_gold, before_pred, after_pred):
        if y0 != y1:
            changed += 1
            correct_changed += p0 == y0 and p1 == y1
        else:
            preserved += 1
            correct_preserved += p0 == y0 and p1 == y1
    return {"changed": changed, "preserved": preserved, "correct_changed": correct_changed,
            "correct_preserved": correct_preserved,
            "all_correct": correct_changed + correct_preserved == changed + preserved}


def check():
    assert not any("test" in p.name.lower() for name in ("condaqa", "negcomment")
                   for p in (OUT / "raw" / name).rglob("*.json"))
    issues = []
    result = {"arts": check_arts(issues), "condaqa": check_condaqa(issues),
              "negcomment": check_negcomment(issues)}
    good = response_counts(["YES", "NO"], ["NO", "NO"], ["YES", "NO"], ["NO", "NO"])
    static = response_counts(["YES", "NO"], ["NO", "NO"], ["YES", "NO"], ["YES", "NO"])
    spill = response_counts(["YES", "NO"], ["NO", "NO"], ["YES", "NO"], ["NO", "YES"])
    assert good["all_correct"] and not static["all_correct"] and not spill["all_correct"]
    assert static["correct_changed"] == 0 and spill["correct_preserved"] == 0
    result["verification"] = {"response_metric_fixtures_pass": True, "model_inference": False,
                              "model_training": False, "external_uploads": False,
                              "condaqa_and_negcomment_test_downloaded": False}
    dump("issues.json", issues)
    dump("result.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch", "check"])
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch()
    else:
        check()
