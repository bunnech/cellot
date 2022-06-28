import numpy as np
import pandas as pd
import re
from anndata import AnnData
import random
import string

DDNAME = re.compile(
    r"(?:(?P<cell_part>Nuclei)_)?"
    r"(?P<measurement>Morphology|Intensity)_"
    r"(?:(?P<stat>sum|mean)_)?"
    r"(?P<feature>.*)"
)

ALPHA = string.ascii_lowercase + string.digits

CLEAN_INTENSITY = {
    "cd45": "CD45",
    "clcasp3": "ClCASP3",
    "dapi": "DAPI",
    "mela": "MelA",
    "tubulin": "Tubulin",
    "pakt": "pAKT",
    "pegfr": "pEGFR",
    "ps6k1": "pS6K1",
    "pmet": "pMET",
    "pmet1": "pMET",
    "pcna": "PCNA",
    "sox9": "Sox9",
    "totalprotein": "Total_protein",
    "totprotein": "Total_protein",
}

STANDARDIZE_DRUG = {
    "cl": "cellline",
    "sec.ab_ctrl": "sec.abctrl",
}


def clean_intensity_feature(name):
    return CLEAN_INTENSITY.get(name.lower().replace("_", ""), name)


def randid():
    return "".join(random.choice(ALPHA) for _ in range(8))


def parse_feature_name(name):

    extract = DDNAME.match(name).groupdict()

    if extract["cell_part"] is None:
        extract["cell_part"] = "cell"

    for key in extract:
        if extract[key] is None:
            continue

        if key == "feature" and extract["measurement"] == "intensity":
            extract[key] = clean_intensity_feature(extract[key])
            continue

        extract[key] = extract[key].lower()

    index = "-".join([extract["measurement"], extract["cell_part"], extract["feature"]])
    if extract["measurement"] == "intensity":
        index = index + "-" + extract["stat"]

    extract["index"] = index
    extract["original"] = name

    return extract


def parse_raw_dd_data(df):
    def clean_drug(drug):
        name = drug.lower().replace(" + ", "_")
        return STANDARDIZE_DRUG.get(name, name)

    df.index = [randid() for _ in range(len(df))]
    assert not df.index.duplicated().any()

    is_data_col = df.columns.map(lambda x: bool(DDNAME.match(x))).values.astype(bool)

    data = df.loc[:, is_data_col].copy()
    obs = df.loc[:, ~is_data_col].copy().rename(columns={"Condition": "drug"})
    assert "drug" in obs.columns
    obs["drug"] = obs["drug"].map(clean_drug)

    var = pd.DataFrame.from_records([parse_feature_name(name) for name in data.columns])

    column_mapper = var.set_index("original")["index"].to_dict()
    var = var.set_index("index")
    data.columns = data.columns.map(column_mapper.get)

    # merge duplicated columns
    if var.index.duplicated().any():
        data = data.groupby(level=0, axis=1).apply(lambda x: x.mean(1))
        var = var.loc[~var.index.duplicated()]
        var = var.loc[data.columns]

    assert not var.index.duplicated().any()
    assert data.columns.equals(var.index)

    return AnnData(data.astype(np.float64), obs=obs, var=var)
