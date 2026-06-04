# Example operational insight brief

## Synthetic monitoring signal

The recent three-week monitor flags an increase in `delivery_delayed` contacts. The `unclear_or_out_of_scope` share also rises in the same period.

## Taxonomy-review finding

The unknown-topic workflow identifies a 27-contact pure out-of-scope cluster with terms such as `latest app`, `update`, and `open prescription`. Representative contacts indicate an app-update/app-crash issue rather than a standard login problem.

## Recommended operational review

1. Check whether an app release, platform incident, or navigation change occurred during the flagged period.
2. Create or refine an `app_crash_after_update` intent after human review.
3. Add a status message or knowledge-base entry if the issue is confirmed.
4. Monitor contact volume, repeat-contact rate, and negative-sentiment rate after the intervention.

## Safety boundary

Urgent medicine needs and possible adverse effects remain outside ordinary self-service automation and are routed to clinical escalation in the demonstration workflow.

## Interpretation limit

All findings come from generated data. They illustrate the analysis workflow and should not be interpreted as Pharmacy2U operational findings.
