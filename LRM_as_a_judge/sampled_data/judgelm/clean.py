import json

# Filenames
file1 = 'judgelm_val_5k.jsonl'
file2 = 'judgelm_val_5k_gpt4.jsonl'
output = 'judgelm_merged.jsonl'

# Load the first file into a dict by question_id, dropping unwanted fields
records = {}
with open(file1, 'r', encoding='utf-8') as f1:
    for line in f1:
        entry = json.loads(line)
        qid = entry['question_id']
        # Remove the model-metrics and decoding metadata
        entry.pop('score', None)
        entry.pop('answer1_metadata', None)
        entry.pop('answer2_metadata', None)
        records[qid] = entry

# Read the second file, merge each record with the first, drop its review_id and metadata
with open(file2, 'r', encoding='utf-8') as f2, \
     open(output, 'w', encoding='utf-8') as out:

    for line in f2:
        entry2 = json.loads(line)
        qid = entry2['question_id']
        # Drop unwanted fields
        entry2.pop('review_id', None)
        entry2.pop('metadata', None)

        if qid not in records:
            # You could choose to warn or skip; here we skip
            continue

        # Merge: first-file fields take precedence, then second-file fields
        merged = {**records[qid], **entry2}
        out.write(json.dumps(merged, ensure_ascii=False) + '\n')

print(f'Merged file written to {output}')
