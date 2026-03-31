import pandas as pd
import pytest

from pipeline.features import FeatureEngineer


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "AGE": [25, 40, 55, 30, 65],
        "GENDER": [1, 0, 1, 0, 1],
        "CHILD_TOTAL": [0, 2, 1, 3, 0],
        "DEPENDANTS": [0, 2, 1, 3, 0],
        "PERSONAL_INCOME": [30000, 50000, 80000, 15000, 100000],
        "SOCSTATUS_WORK_FL": [1, 1, 0, 1, 0],
        "SOCSTATUS_PENS_FL": [0, 0, 1, 0, 1],
        "LOAN_NUM_TOTAL": [2, 5, 3, 1, 4],
        "LOAN_NUM_CLOSED": [1.0, 3.0, 3.0, 0.0, 2.0],
    })


@pytest.fixture
def fitted_engineer(sample_data):
    fe = FeatureEngineer()
    fe.fit(sample_data)
    return fe


class TestFeatureEngineer:

    def test_fit_sets_feature_names(self, fitted_engineer):
        assert fitted_engineer.feature_names_ is not None
        assert len(fitted_engineer.feature_names_) > 9

    def test_transform_returns_dataframe(self, fitted_engineer, sample_data):
        result = fitted_engineer.transform(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == sample_data.shape[0]

    def test_transform_creates_loan_close_ratio(self, sample_data):
        fe = FeatureEngineer()
        result = fe._create_features(sample_data.copy())
        assert "LOAN_CLOSE_RATIO" in result.columns
        assert result["LOAN_CLOSE_RATIO"].iloc[0] == pytest.approx(0.5)
        assert result["LOAN_CLOSE_RATIO"].iloc[2] == pytest.approx(1.0)

    def test_transform_creates_open_loans(self, sample_data):
        fe = FeatureEngineer()
        result = fe._create_features(sample_data.copy())
        assert "OPEN_LOANS" in result.columns
        assert result["OPEN_LOANS"].iloc[0] == 1
        assert result["OPEN_LOANS"].iloc[2] == 0

    def test_transform_creates_income_per_dependant(self, sample_data):
        fe = FeatureEngineer()
        result = fe._create_features(sample_data.copy())
        assert "INCOME_PER_DEPENDANT" in result.columns
        assert result["INCOME_PER_DEPENDANT"].iloc[0] == pytest.approx(30000.0)
        assert result["INCOME_PER_DEPENDANT"].iloc[1] == pytest.approx(50000 / 3)

    def test_transform_creates_age_groups(self, sample_data):
        fe = FeatureEngineer()
        result = fe._create_features(sample_data.copy())
        age_cols = [c for c in result.columns if c.startswith("AGE_")]
        assert len(age_cols) >= 3

    def test_transform_creates_income_groups(self, sample_data):
        fe = FeatureEngineer()
        result = fe._create_features(sample_data.copy())
        inc_cols = [c for c in result.columns if c.startswith("INC_")]
        assert len(inc_cols) >= 3

    def test_scaled_output_has_zero_mean(self, fitted_engineer, sample_data):
        result = fitted_engineer.transform(sample_data)
        means = result.mean()
        for col in result.columns:
            assert abs(means[col]) < 1e-10

    def test_single_row_prediction(self, fitted_engineer):
        single = pd.DataFrame([{
            "AGE": 35, "GENDER": 1, "CHILD_TOTAL": 1, "DEPENDANTS": 1,
            "PERSONAL_INCOME": 45000.0, "SOCSTATUS_WORK_FL": 1, "SOCSTATUS_PENS_FL": 0,
            "LOAN_NUM_TOTAL": 2, "LOAN_NUM_CLOSED": 1.0,
        }])
        result = fitted_engineer.transform(single)
        assert result.shape[0] == 1
        assert result.shape[1] == len(fitted_engineer.feature_names_)

    def test_zero_loans_ratio(self):
        fe = FeatureEngineer()
        df = pd.DataFrame([{
            "AGE": 30, "GENDER": 0, "CHILD_TOTAL": 0, "DEPENDANTS": 0,
            "PERSONAL_INCOME": 20000.0, "SOCSTATUS_WORK_FL": 1, "SOCSTATUS_PENS_FL": 0,
            "LOAN_NUM_TOTAL": 0, "LOAN_NUM_CLOSED": 0.0,
        }])
        result = fe._create_features(df.copy())
        assert result["LOAN_CLOSE_RATIO"].iloc[0] == 0.0

    def test_get_feature_names_out(self, fitted_engineer):
        names = fitted_engineer.get_feature_names_out()
        assert names == fitted_engineer.feature_names_
