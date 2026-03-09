import numpy as np
import yaml
from importlib import resources

from ..dh_param import Dh


def load_robot_config(name: str) -> dict:
    config_file = f"{name}.yaml"

    with resources.files("model.configs").joinpath(config_file).open() as f:
        cfg = yaml.safe_load(f)

    if cfg["param_type"] == "mat":
        param = [np.array(m) for m in cfg["mat"]]
    else:
        dh_type = cfg["param_type"].replace("_param", "")
        dh = Dh(cfg["dh"], type=dh_type)

        if cfg.get("dh_unit") == "millimeter":
            dh.millimeter_to_meter()

        param = dh

    if cfg.get("joint_unit") == "degree":
        upper = np.radians(cfg["upper"])
        lower = np.radians(cfg["lower"])
    else:
        upper = np.array(cfg["upper"])
        lower = np.array(cfg["lower"])

    return {
        "type": cfg["param_type"],
        "param": param,
        "upper": upper,
        "lower": lower,
    }
