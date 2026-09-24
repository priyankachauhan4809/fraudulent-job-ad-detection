import pandas as pd


def create_metadata_features(df):
    """
    Create structured metadata features from job advertisement data.
    """

    metadata = pd.DataFrame(index=df.index)

    metadata["telecommuting"] = (
        df["telecommuting"] == "t"
    ).astype(int)

    metadata["has_company_logo"] = (
        df["has_company_logo"] == "t"
    ).astype(int)

    metadata["has_questions"] = (
        df["has_questions"] == "t"
    ).astype(int)

    metadata["company_profile_missing"] = (
        df["company_profile"].isna()
    ).astype(int)

    metadata["requirements_missing"] = (
        df["requirements"].isna()
    ).astype(int)

    metadata["benefits_missing"] = (
        df["benefits"].isna()
    ).astype(int)

    metadata["salary_missing"] = (
        df["salary_range"].isna()
    ).astype(int)

    metadata["employment_type_missing"] = (
        df["employment_type"].isna()
    ).astype(int)

    metadata["education_missing"] = (
        df["required_education"].isna()
    ).astype(int)

    metadata["experience_missing"] = (
        df["required_experience"].isna()
    ).astype(int)

    metadata["industry_missing"] = (
        df["industry"].isna()
    ).astype(int)

    metadata["function_missing"] = (
        df["function"].isna()
    ).astype(int)

    metadata["location_missing"] = (
        df["location"].isna()
    ).astype(int)

    return metadata


def create_combined_text(df):
    """
    Combine relevant text fields into a single text column.
    """

    combined_text = (
        df["title"].fillna("") + " " +
        df["company_profile"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["requirements"].fillna("") + " " +
        df["benefits"].fillna("")
    )

    return combined_text