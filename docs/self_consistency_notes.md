# Self-Consistency Notes

Use consistent sampling settings when measuring the effect of self-consistency on AES predictions.

## Sampling Record

For each analysis, record:

- Checkpoint or adapter path.
- Prompt template version.
- Number of sampled responses per essay.
- Temperature, top-p, max tokens, and random seed.
- Voting rule, such as majority vote or average vote.
- Dataset split and sample count.

## Diagnostics

Save raw generations, parsed scores, vote summaries, and parse failures together. Parse failures are useful signal: they often indicate prompt drift or output-format instability.

## Reporting

Report self-consistency results next to the single-sample baseline from the same checkpoint and split. This avoids attributing gains to sampling when the underlying model or data changed.
