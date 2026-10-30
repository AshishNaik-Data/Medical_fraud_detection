"""
Data loading and merging: loads raw CMS-style CSVs, merges claims with
beneficiary data, and returns provider-level DataFrames ready for feature
engineering.

Key design decisions:
  - All merging is done at the claim level first, then aggregated to provider.
  - Train and test go through identical transformations (no leakage).
  - Date parsing is explicit to avoid silent coercion errors.
  - Memory-efficient dtypes are applied after loading.
"""
import pandas as pd
import numpy as np

from config import (
    CHRONIC_COLS, DATE_COLS_INPATIENT, DATE_COLS_OUTPATIENT,
    DIAGNOSIS_COLS, PHYSICIAN_COLS, PROCEDURE_COLS,
    TRAIN_BENEFICIARY, TRAIN_INPATIENT, TRAIN_LABELS, TRAIN_OUTPATIENT,
    TEST_BENEFICIARY, TEST_INPATIENT, TEST_LABELS, TEST_OUTPATIENT,
)
from utils import setup_logging

logger = setup_logging()


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save memory."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def _load_beneficiary(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    df["DOD"] = pd.to_datetime(df["DOD"], errors="coerce")
    df["IsDead"] = df["DOD"].notna().astype(np.int8)

    # Compute age as of a reference date (end of 2009, the data period)
    ref_date = pd.Timestamp("2009-12-31")
    df["Age"] = ((ref_date - df["DOB"]).dt.days / 365.25).round(1)

    # Recode chronic conditions: raw 1=Yes, 2=No → 1/0
    for col in CHRONIC_COLS:
        if col in df.columns:
            df[col] = (df[col] == 1).astype(np.int8)
    df["ChronicCondCount"] = df[CHRONIC_COLS].sum(axis=1).astype(np.int8)

    # Clean RenalDiseaseIndicator
    df["RenalDiseaseIndicator"] = (
        pd.to_numeric(df["RenalDiseaseIndicator"], errors="coerce")
        .fillna(0).astype(np.int8)
    )

    # Fill missing financial columns
    for col in ["IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
                 "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return _optimize_dtypes(df)


def _load_claims(inpatient_path, outpatient_path) -> pd.DataFrame:
    """Load and combine inpatient + outpatient claims with a type indicator."""
    inp = pd.read_csv(inpatient_path)
    out = pd.read_csv(outpatient_path)

    # Parse dates
    for col in DATE_COLS_INPATIENT:
        if col in inp.columns:
            inp[col] = pd.to_datetime(inp[col], errors="coerce")
    for col in DATE_COLS_OUTPATIENT:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    inp["ClaimType"] = "Inpatient"
    out["ClaimType"] = "Outpatient"

    # Inpatient has AdmissionDt/DischargeDt; outpatient doesn't
    if "AdmissionDt" not in out.columns:
        out["AdmissionDt"] = pd.NaT
    if "DischargeDt" not in out.columns:
        out["DischargeDt"] = pd.NaT
    if "DiagnosisGroupCode" not in out.columns:
        out["DiagnosisGroupCode"] = np.nan

    # Combine
    claims = pd.concat([inp, out], ignore_index=True, sort=False)

    # Claim duration (inpatient only, in days)
    claims["ClaimDuration"] = (
        (claims["DischargeDt"] - claims["AdmissionDt"]).dt.days
    )
    claims["ClaimDuration"] = claims["ClaimDuration"].clip(lower=0)

    # Number of diagnosis/procedure codes filled per claim
    claims["NumDiagCodes"] = claims[DIAGNOSIS_COLS].notna().sum(axis=1).astype(np.int8)
    claims["NumProcCodes"] = claims[PROCEDURE_COLS].notna().sum(axis=1).astype(np.int8)

    # Number of physicians on the claim
    claims["NumPhysicians"] = claims[PHYSICIAN_COLS].notna().sum(axis=1).astype(np.int8)

    return _optimize_dtypes(claims)


def _merge_claims_beneficiary(claims: pd.DataFrame, bene: pd.DataFrame) -> pd.DataFrame:
    """Left join claims → beneficiary on BeneID."""
    merged = claims.merge(bene, on="BeneID", how="left", suffixes=("", "_bene"))
    return merged


def load_dataset(split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge all data for a split.

    Returns:
        claims_merged: Claim-level DataFrame with beneficiary info joined
        labels: Provider-level DataFrame with PotentialFraud column (train only)
    """
    if split == "train":
        labels = pd.read_csv(TRAIN_LABELS)
        bene = _load_beneficiary(TRAIN_BENEFICIARY)
        claims = _load_claims(TRAIN_INPATIENT, TRAIN_OUTPATIENT)
    else:
        labels = pd.read_csv(TEST_LABELS)
        bene = _load_beneficiary(TEST_BENEFICIARY)
        claims = _load_claims(TEST_INPATIENT, TEST_OUTPATIENT)

    # Encode target for train
    if "PotentialFraud" in labels.columns:
        labels["PotentialFraud"] = (labels["PotentialFraud"] == "Yes").astype(np.int8)

    claims_merged = _merge_claims_beneficiary(claims, bene)

    logger.info(
        f"Loaded {split}: {len(claims_merged):,} claims, "
        f"{claims_merged['Provider'].nunique():,} providers, "
        f"{len(labels):,} label rows"
    )
    return claims_merged, labels


def build_provider_features(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate claim-level data to provider-level features.
    This is the core feature engineering step — no target info is used here.

    Produces ~60 features per provider covering:
    - Claim volume and type mix
    - Financial statistics
    - Beneficiary demographics
    - Diagnosis/procedure diversity
    - Physician patterns
    - Temporal patterns
    - Anomaly-style ratios
    """
    g = claims.groupby("Provider")

    feats = pd.DataFrame({"Provider": claims["Provider"].unique()})

    # ─── Claim Volume ────────────────────────────────────────────────────
    claim_counts = g.agg(
        TotalClaims=("ClaimID", "nunique"),
        TotalClaimRows=("ClaimID", "count"),
        InpatientClaims=("ClaimType", lambda x: (x == "Inpatient").sum()),
        OutpatientClaims=("ClaimType", lambda x: (x == "Outpatient").sum()),
    ).reset_index()
    feats = feats.merge(claim_counts, on="Provider", how="left")
    feats["InpatientRatio"] = feats["InpatientClaims"] / (feats["TotalClaims"] + 1)

    # ─── Financial ───────────────────────────────────────────────────────
    financial = g.agg(
        TotalReimbursed=("InscClaimAmtReimbursed", "sum"),
        MeanReimbursed=("InscClaimAmtReimbursed", "mean"),
        StdReimbursed=("InscClaimAmtReimbursed", "std"),
        MaxReimbursed=("InscClaimAmtReimbursed", "max"),
        MedianReimbursed=("InscClaimAmtReimbursed", "median"),
        TotalDeductible=("DeductibleAmtPaid", "sum"),
        MeanDeductible=("DeductibleAmtPaid", "mean"),
    ).reset_index()
    financial["StdReimbursed"] = financial["StdReimbursed"].fillna(0)
    feats = feats.merge(financial, on="Provider", how="left")

    # Reimbursement per claim
    feats["ReimbursedPerClaim"] = feats["TotalReimbursed"] / (feats["TotalClaims"] + 1)
    # Coefficient of variation
    feats["ReimbursedCV"] = feats["StdReimbursed"] / (feats["MeanReimbursed"] + 1)

    # ─── Beneficiary Patterns ────────────────────────────────────────────
    bene_stats = g.agg(
        UniqueBeneficiaries=("BeneID", "nunique"),
        MeanAge=("Age", "mean"),
        StdAge=("Age", "std"),
        FracDead=("IsDead", "mean"),
        MeanChronicCount=("ChronicCondCount", "mean"),
        MaxChronicCount=("ChronicCondCount", "max"),
        MeanIPReimbursement=("IPAnnualReimbursementAmt", "mean"),
        MeanOPReimbursement=("OPAnnualReimbursementAmt", "mean"),
    ).reset_index()
    bene_stats["StdAge"] = bene_stats["StdAge"].fillna(0)
    feats = feats.merge(bene_stats, on="Provider", how="left")

    # Claims per beneficiary (high → suspicious)
    feats["ClaimsPerBeneficiary"] = feats["TotalClaims"] / (feats["UniqueBeneficiaries"] + 1)

    # ─── Chronic Condition Aggregates ────────────────────────────────────
    for col in CHRONIC_COLS:
        if col in claims.columns:
            chronic_rate = g[col].mean().reset_index()
            chronic_rate.columns = ["Provider", f"{col}_Rate"]
            feats = feats.merge(chronic_rate, on="Provider", how="left")

    # Renal disease rate
    if "RenalDiseaseIndicator" in claims.columns:
        renal = g["RenalDiseaseIndicator"].mean().reset_index()
        renal.columns = ["Provider", "RenalRate"]
        feats = feats.merge(renal, on="Provider", how="left")

    # ─── Diagnosis and Procedure Diversity ───────────────────────────────
    diag_diversity = g.agg(
        MeanDiagCodes=("NumDiagCodes", "mean"),
        MaxDiagCodes=("NumDiagCodes", "max"),
    ).reset_index()
    feats = feats.merge(diag_diversity, on="Provider", how="left")

    proc_diversity = g.agg(
        MeanProcCodes=("NumProcCodes", "mean"),
        MaxProcCodes=("NumProcCodes", "max"),
    ).reset_index()
    feats = feats.merge(proc_diversity, on="Provider", how="left")

    # Count of distinct diagnosis codes used by provider
    def _count_unique_codes(sub, cols):
        vals = sub[cols].values.ravel()
        return pd.Series(vals).dropna().nunique()

    diag_unique = g.apply(lambda x: _count_unique_codes(x, DIAGNOSIS_COLS), include_groups=False).reset_index()
    diag_unique.columns = ["Provider", "UniqueDiagCodes"]
    feats = feats.merge(diag_unique, on="Provider", how="left")

    proc_unique = g.apply(lambda x: _count_unique_codes(x, PROCEDURE_COLS), include_groups=False).reset_index()
    proc_unique.columns = ["Provider", "UniqueProcCodes"]
    feats = feats.merge(proc_unique, on="Provider", how="left")

    # Diversity ratios (unique codes / total claims)
    feats["DiagDiversityRatio"] = feats["UniqueDiagCodes"] / (feats["TotalClaims"] + 1)
    feats["ProcDiversityRatio"] = feats["UniqueProcCodes"] / (feats["TotalClaims"] + 1)

    # ─── Physician Patterns ──────────────────────────────────────────────
    phys_stats = g.agg(
        MeanPhysicians=("NumPhysicians", "mean"),
    ).reset_index()
    feats = feats.merge(phys_stats, on="Provider", how="left")

    # Unique physicians per provider
    for col in PHYSICIAN_COLS:
        if col in claims.columns:
            nuniq = g[col].nunique().reset_index()
            nuniq.columns = ["Provider", f"Unique{col}"]
            feats = feats.merge(nuniq, on="Provider", how="left")

    feats["TotalUniquePhysicians"] = 0
    for col in PHYSICIAN_COLS:
        key = f"Unique{col}"
        if key in feats.columns:
            feats["TotalUniquePhysicians"] += feats[key].fillna(0)

    # ─── Temporal Patterns ───────────────────────────────────────────────
    temporal = g.agg(
        MeanClaimDuration=("ClaimDuration", "mean"),
        MaxClaimDuration=("ClaimDuration", "max"),
    ).reset_index()
    temporal["MeanClaimDuration"] = temporal["MeanClaimDuration"].fillna(0)
    temporal["MaxClaimDuration"] = temporal["MaxClaimDuration"].fillna(0)
    feats = feats.merge(temporal, on="Provider", how="left")

    # Claim date spread (max - min claim start)
    date_spread = g["ClaimStartDt"].agg(["min", "max"]).reset_index()
    date_spread["ClaimDateSpanDays"] = (date_spread["max"] - date_spread["min"]).dt.days
    date_spread = date_spread[["Provider", "ClaimDateSpanDays"]]
    feats = feats.merge(date_spread, on="Provider", how="left")

    # ─── State/County Concentration ──────────────────────────────────────
    state_count = g["State"].nunique().reset_index()
    state_count.columns = ["Provider", "UniqueStates"]
    feats = feats.merge(state_count, on="Provider", how="left")

    county_count = g["County"].nunique().reset_index()
    county_count.columns = ["Provider", "UniqueCounties"]
    feats = feats.merge(county_count, on="Provider", how="left")

    # Gender mix
    if "Gender" in claims.columns:
        gender_frac = g["Gender"].apply(lambda x: (x == 1).mean()).reset_index()
        gender_frac.columns = ["Provider", "MaleRatio"]
        feats = feats.merge(gender_frac, on="Provider", how="left")

    # Race diversity (entropy proxy)
    if "Race" in claims.columns:
        race_nuniq = g["Race"].nunique().reset_index()
        race_nuniq.columns = ["Provider", "UniqueRaces"]
        feats = feats.merge(race_nuniq, on="Provider", how="left")

    # ─── Anomaly Ratios ──────────────────────────────────────────────────
    # Zero-deductible claim fraction (suspicious if high)
    zero_ded = g["DeductibleAmtPaid"].apply(lambda x: (x == 0).mean()).reset_index()
    zero_ded.columns = ["Provider", "ZeroDeductibleFrac"]
    feats = feats.merge(zero_ded, on="Provider", how="left")

    # Same-day claim count per beneficiary (max across beneficiaries)
    claims["_claim_date_str"] = claims["ClaimStartDt"].dt.strftime("%Y-%m-%d")
    same_day = (
        claims.groupby(["Provider", "BeneID", "_claim_date_str"])["ClaimID"]
        .nunique().reset_index()
    )
    same_day.columns = ["Provider", "BeneID", "_date", "SameDayClaims"]
    same_day_max = same_day.groupby("Provider")["SameDayClaims"].max().reset_index()
    same_day_max.columns = ["Provider", "MaxSameDayClaimsPerBene"]
    feats = feats.merge(same_day_max, on="Provider", how="left")

    same_day_mean = same_day.groupby("Provider")["SameDayClaims"].mean().reset_index()
    same_day_mean.columns = ["Provider", "MeanSameDayClaimsPerBene"]
    feats = feats.merge(same_day_mean, on="Provider", how="left")

    # Drop temp column
    claims.drop("_claim_date_str", axis=1, inplace=True, errors="ignore")

    # Fill any remaining NaN
    feats = feats.fillna(0)
    feats = _optimize_dtypes(feats)

    logger.info(f"Built {feats.shape[1]-1} provider-level features for {len(feats)} providers")
    return feats
