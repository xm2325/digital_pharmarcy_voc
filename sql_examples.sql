-- Weekly contact-driver trend
SELECT
  strftime('%Y-%W', timestamp) AS year_week,
  intent,
  COUNT(*) AS contacts
FROM contacts_raw
GROUP BY year_week, intent
ORDER BY year_week, contacts DESC;

-- Patient-level link between contact patterns and synthetic CRM outcomes
SELECT
  c.pseudo_patient_id,
  COUNT(*) AS contact_count,
  SUM(CASE WHEN c.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_contacts,
  SUM(CASE WHEN c.repeat_contact_14d = 1 THEN 1 ELSE 0 END) AS repeat_contacts,
  MAX(CASE WHEN c.safety_sensitive = 1 THEN 1 ELSE 0 END) AS has_safety_sensitive_contact,
  crm.csat_score,
  crm.retained_90d,
  crm.crm_segment
FROM contacts_raw c
LEFT JOIN crm ON c.pseudo_patient_id = crm.pseudo_patient_id
GROUP BY c.pseudo_patient_id, crm.csat_score, crm.retained_90d, crm.crm_segment;

-- Contact categories that could be reviewed for proactive communication
SELECT
  intent,
  journey_stage,
  COUNT(*) AS contacts,
  SUM(CASE WHEN proactive_candidate = 1 THEN 1 ELSE 0 END) AS proactive_candidates
FROM contacts_raw
GROUP BY intent, journey_stage
ORDER BY proactive_candidates DESC, contacts DESC;

-- Generated out-of-taxonomy app-crash validation pattern
SELECT
  strftime('%Y-%W', timestamp) AS year_week,
  synthetic_issue_pattern,
  COUNT(*) AS contacts
FROM contacts_raw
WHERE synthetic_issue_pattern = 'novel_app_crash'
GROUP BY year_week, synthetic_issue_pattern
ORDER BY year_week;
