CREATE CONSTRAINT patient_id IF NOT EXISTS
FOR (p:Patient)
REQUIRE p.patient_id IS UNIQUE;

CREATE CONSTRAINT doctor_id IF NOT EXISTS
FOR (d:Doctor)
REQUIRE d.doctor_id IS UNIQUE;

CREATE CONSTRAINT claim_id IF NOT EXISTS
FOR (c:Claim)
REQUIRE c.claim_event_id IS UNIQUE;

CREATE CONSTRAINT specialty_name IF NOT EXISTS
FOR (s:Specialty)
REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT subspecialty_name IF NOT EXISTS
FOR (ss:Subspecialty)
REQUIRE ss.name IS UNIQUE;

CALL gds.graph.project(
  'medicalFraudGraph',
  ['Patient', 'Doctor', 'Claim', 'Specialty', 'Subspecialty'],
  {
    HAS_CLAIM: {orientation: 'NATURAL'},
    SUBMITTED: {orientation: 'NATURAL'},
    SPECIALIZED_IN: {orientation: 'NATURAL'},
    HAS_SUBSPECIALTY: {orientation: 'NATURAL'},
    TREATED: {orientation: 'UNDIRECTED'}
  }
);

