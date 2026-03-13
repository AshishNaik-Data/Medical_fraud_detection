// Rule 1: Too many claims per doctor
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)
WITH d, count(c) AS claim_count
WHERE claim_count > $min_claims
RETURN d.doctor_id AS doctor_id, claim_count
ORDER BY claim_count DESC;

// Rule 2: Doctor treating outside specialty
MATCH (d:Doctor)-[:SUBMITTED]->(c:Claim)
WHERE c.specialty_mismatch = true
RETURN d.doctor_id AS doctor_id, c.claim_event_id AS claim_event_id, c.code_group_title AS code_group;

// Rule 3: Multiple claims for the same patient on the same day
MATCH (p:Patient)-[:HAS_CLAIM]->(c:Claim)<-[:SUBMITTED]-(d:Doctor)
WITH p, date(c.claim_time) AS claim_day, count(c) AS claims_same_day, collect(c.claim_event_id) AS claim_ids
WHERE claims_same_day > 1
RETURN p.patient_id AS patient_id, claim_day, claims_same_day, claim_ids;

// Rule 4: Doctors concentrated in risky communities
MATCH (d:Doctor)
WHERE d.community_risk_score >= $community_risk_cutoff
RETURN d.doctor_id AS doctor_id, d.community_id AS community_id, d.community_risk_score
ORDER BY d.community_risk_score DESC;

