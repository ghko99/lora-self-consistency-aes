# Generation Audit Notes

Audit generated scoring outputs before using them for self-consistency or metric reporting.

## Checks

- Confirm every sampled response can be parsed into the expected rubric fields.
- Count missing, duplicated, or out-of-range scores.
- Save raw generations for a small review sample.
- Compare majority vote and average vote outputs on the same sample set.

## Failure Categories

Group failures into parser errors, prompt drift, invalid score ranges, repetitive responses, or model uncertainty. This makes follow-up changes easier to target.

## Reporting

Report sampling settings, parse failure count, aggregation rule, and split name next to self-consistency metrics.
