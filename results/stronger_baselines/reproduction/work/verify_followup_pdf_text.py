"""Check actual PDF text against the completed report's bound numeric payload.

This supplements the builder's source gates and separate page-by-page visual QA.
It does not replace either of them or assess scientific inference from typography.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pypdf import PdfReader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def ordered_row_present(text, label, fields):
    """Keep a label associated with its own displayed values and column order."""
    return normalized(" ".join([label, *fields])) in normalized(text)


def report_sections(text, studies):
    """Bound row associations by the builder's unique, ordered section titles.

    Global sensitivity repeats the study contrast labels and some numeric fields.
    Its rows cannot substitute for an omitted row in a primary study table.
    """
    names = ("equalization", "vision", "text", "tabular")
    if set(studies) != set(names):
        raise ValueError("Exactly the four declared study sections are required")
    headers = []
    for name in names:
        for role in ("methodology", "confirmation results", "paired comparisons"):
            headers.append(normalized(studies[name]["title"] + " | " + role))
    headers += ["Global multiplicity sensitivity | 1 of 2",
                "Global multiplicity sensitivity | 2 of 2",
                "Practical interpretation and limitations"]
    text = normalized(text)
    positions = []
    for header in headers:
        if text.count(header) != 1:
            raise ValueError("Missing or duplicate report section header: " + header)
        positions.append(text.index(header))
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("Report section headers are out of the builder's order")

    def body(index):
        return text[positions[index] + len(headers[index]):positions[index + 1]].strip()

    return {
        "confirmation": {name: body(3 * i + 1) for i, name in enumerate(names)},
        "primary": {name: body(3 * i + 2) for i, name in enumerate(names)},
        "global": [body(12), body(13)],
        "headers": headers,
    }


def verify(evidence, pdf):
    evidence, pdf = Path(evidence), Path(pdf)
    data = json.loads(evidence.read_text())
    assert data["status"] == "all_studies_complete_independently_verified"
    assert len(data["studies"]) == 4
    assert len(data["global_analysis"]["comparisons"]) == 22
    reader = PdfReader(pdf, strict=True)
    assert not reader.is_encrypted
    pages = [page.extract_text() or "" for page in reader.pages]
    assert pages and all(normalized(text) for text in pages)
    text = normalized("\n".join(pages))
    sections = report_sections(text, data["studies"])
    compact = re.sub(r"\s+", "", text)
    missing = []

    def require(value, role):
        if normalized(value) not in text:
            missing.append(role)

    def metric(value):
        return f"{100 * value['mean']:.2f} +/- {100 * value['sd']:.2f}"

    model_rows = 0
    for name, study in data["studies"].items():
        require(study["title"], name + " study title")
        for count in ("source_rows", "fit_rows", "confirmation_rows"):
            require(f"{study[count]:,}", name + " " + count)
        for row in study["summary"]:
            label = row["label"]
            require(label, name + " model label")
            fields = []
            for key in ("macro_f1", "tail_f1", "accuracy"):
                fields.append(metric(row["metrics"][key]))
                require(fields[-1], name + " " + label + " " + key)
            if not ordered_row_present(sections["confirmation"][name], label, fields):
                missing.append(name + " " + label + " ordered metric table row")
            model_rows += 1

    for index, row in enumerate(data["global_analysis"]["comparisons"]):
        require(row["label"], "primary contrast label")
        precision = 4 if row["study"] == "equalization" and row["name"] in (
            "fixed_natural_EQ_effect", "fixed_balanced_EQ_effect",
            "sampling_effect_EQoff", "fixed_factorial_interaction",
        ) else 2
        effect = (f"{100 * row['mean']:+.{precision}f} "
                  f"[{100 * row['ci95'][0]:+.{precision}f}, "
                  f"{100 * row['ci95'][1]:+.{precision}f}]")
        require(effect, row["study"] + " primary effect and interval")
        for key in ("signflip_p", "signflip_p_study_holm", "signflip_p_global_holm"):
            require(f"{row[key]:.6f}", row["study"] + " " + key)
        if not ordered_row_present(sections["primary"][row["study"]], row["label"], [effect,
                f"{row['signflip_p']:.6f}", f"{row['signflip_p_study_holm']:.6f}"]):
            missing.append(row["study"] + " ordered primary contrast table row")
        if not ordered_row_present(sections["global"][index // 11], row["study"] + " / " + row["label"],
                [effect, f"{row['signflip_p_study_holm']:.6f}",
                 f"{row['signflip_p_global_holm']:.6f}"]):
            missing.append(row["study"] + " ordered global sensitivity table row")

    for row in data["identity_table"]:
        if row["sha256"] not in compact:
            missing.append("artifact identity " + row["role"])
    for ref in data["references"]:
        if re.sub(r"\s+", "", ref["url"]) not in compact:
            missing.append("reference URL " + ref["title"])
    assert not missing, "Rendered numeric/text fields missing: " + "; ".join(missing)
    assert all(abs(float(page.mediabox.width) - 612) < .01 and
               abs(float(page.mediabox.height) - 792) < .01 for page in reader.pages)
    return dict(status="completed_pdf_text_and_numeric_fields_verified",
                verified_utc=datetime.now(timezone.utc).isoformat(),
                pdf_sha256=sha(pdf), evidence_sha256=sha(evidence),
                verifier_sha256=sha(__file__), pages=len(pages),
                model_rows_verified=model_rows, primary_contrasts_verified=22,
                displayed_metrics_per_model=3,
                ordered_model_rows_verified=model_rows,
                ordered_primary_rows_verified=22,
                ordered_global_sensitivity_rows_verified=22,
                unique_ordered_section_headers_verified=len(sections["headers"]),
                rows_bound_to_declared_report_sections=True,
                artifact_identities_verified=len(data["identity_table"]),
                reference_urls_verified=len(data["references"]),
                visual_inspection_required_separately=True,
                scope="Text extraction contains every source-bound displayed metric, paired effect/interval, primary p-value, identity and reference. Model, primary and global rows occur in their own uniquely bounded report sections. This does not prove spatial layout or independently validate training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.evidence, args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
