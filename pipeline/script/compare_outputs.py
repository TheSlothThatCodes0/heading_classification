import json
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
EXPECTED_DIR = PIPELINE_DIR / "expected output"
OUTPUT_DIR   = PIPELINE_DIR / "output/infer_binary"


def load_outline(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(item.get('level',''), item.get('text','').strip()) for item in data.get('outline', [])]


def compare_files():
    if not EXPECTED_DIR.exists():
        print(f"Expected output directory not found: {EXPECTED_DIR}")
        return

    results = []
    for exp_file in EXPECTED_DIR.glob("*_expected.json"):
        # derive actual output file name
        base = exp_file.stem.replace('_expected', '')
        actual_file = OUTPUT_DIR / f"{base}.json"
        if not actual_file.exists():
            print(f"Skipping {base}: actual output missing")
            continue

        exp_outline = load_outline(exp_file)
        act_outline = load_outline(actual_file)

        # compute text-only metrics
        exp_text_set = set(text for _, text in exp_outline)
        act_text_set = set(text for _, text in act_outline)
        text_matches = exp_text_set & act_text_set
        recall_text = len(text_matches) / len(exp_text_set) if exp_text_set else 1.0
        precision_text = len(text_matches) / len(act_text_set) if act_text_set else 1.0
        f1_text = (2 * precision_text * recall_text / (precision_text + recall_text)) if (precision_text + recall_text) else 0.0

        # compute level-text pair metrics
        exp_pair_set = set(exp_outline)
        act_pair_set = set(act_outline)
        pair_matches = exp_pair_set & act_pair_set
        recall_pair = len(pair_matches) / len(exp_pair_set) if exp_pair_set else 1.0
        precision_pair = len(pair_matches) / len(act_pair_set) if act_pair_set else 1.0
        f1_pair = (2 * precision_pair * recall_pair / (precision_pair + recall_pair)) if (precision_pair + recall_pair) else 0.0

        results.append((base,
                        len(exp_pair_set), len(act_pair_set),
                        len(text_matches), recall_text, precision_text, f1_text,
                        len(pair_matches), recall_pair, precision_pair, f1_pair))

    # print summary
    if not results:
        print("No files compared.")
        return

    # header including text-only and level-text pair metrics
    print("File         Exp  Act TxtM TxtR TxtP TxtF1 PairM PairR PairP PairF1")
    for (base, exp_cnt, act_cnt,
         txt_mat, txt_rec, txt_prec, txt_f1,
         pair_mat, pair_rec, pair_prec, pair_f1) in results:
        print(f"{base:12s} {exp_cnt:4d} {act_cnt:4d} {txt_mat:4d} {txt_rec:6.2f} {txt_prec:6.2f} {txt_f1:6.2f} {pair_mat:4d} {pair_rec:6.2f} {pair_prec:6.2f} {pair_f1:6.2f}")

    # aggregate averages for both metrics
    avg_txt_rec = sum(r[4] for r in results) / len(results)
    avg_txt_prec = sum(r[5] for r in results) / len(results)
    avg_txt_f1 = sum(r[6] for r in results) / len(results)
    # pair metrics are at indices 8,9,10
    avg_pair_rec = sum(r[8] for r in results) / len(results)
    avg_pair_prec = sum(r[9] for r in results) / len(results)
    avg_pair_f1 = sum(r[10] for r in results) / len(results)
    print(f"\nAverage Txt  -    -    -    {avg_txt_rec:.2f} {avg_txt_prec:.2f} {avg_txt_f1:.2f}")
    print(f"Average Pair -    -    -    {avg_pair_rec:.2f} {avg_pair_prec:.2f} {avg_pair_f1:.2f}")


if __name__ == "__main__":
    compare_files()
