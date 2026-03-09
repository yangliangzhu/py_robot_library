"""
验证测试数据 - 用于在修改代码后验证输出一致性
"""
import numpy as np
import sys

from model.robot_model_numpy import RobotModelNumpy
from model.configs.loader import load_robot_config
from model.ik_type import IkType


def validate_config(name, config, param_type, data_dir="test_data"):
    """验证一个配置的测试数据"""
    print(f"验证: {name}")
    
    data = np.load(f"{data_dir}/{name}.npz")
    joint_angles = data['joint_angles']
    fk_expected = data['fk_results']
    jac_expected = data['jacobian_results']
    jac_normal_expected = data['jac_normal_results']
    num_dof = data['num_dof']
    
    robot = RobotModelNumpy()
    
    if param_type == "mat":
        robot.build(type="mat", solver_type=IkType.IK_STANDARD, config=config)
    else:
        robot.build(type="dh", solver_type=IkType.IK_STANDARD, config=config)
    
    errors = []
    
    for i, q in enumerate(joint_angles):
        # FK验证
        T = robot.fk(q)
        if not np.allclose(T.flatten(), fk_expected[i], atol=1e-10):
            errors.append(f"  FK mismatch at sample {i}")
        
        # Jacobian验证
        jac = robot.jacobian(q)
        if not np.allclose(jac.flatten(), jac_expected[i], atol=1e-10):
            errors.append(f"  Jacobian mismatch at sample {i}")
        
        # Jacobian normal验证
        jac_normal = robot.jac_normal(q)
        if not np.allclose(jac_normal.flatten(), jac_normal_expected[i], atol=1e-10):
            errors.append(f"  Jac_normal mismatch at sample {i}")
    
    if errors:
        print(f"  FAILED: {len(errors)} errors")
        for e in errors[:5]:
            print(e)
        if len(errors) > 5:
            print(f"  ... and {len(errors)-5} more")
        return False
    else:
        print(f"  OK: {len(joint_angles)} samples verified")
        return True


def main():
    configs = [
        ("franka_mdh", load_robot_config("franka"), "mdh"),
        ("rokae_er3_mdh", load_robot_config("rokae_er3"), "mdh"),
        ("rokae_er3_sdh", load_robot_config("rokae_er3_sdh"), "sdh"),
        ("rokae_er3_plus_mdh", load_robot_config("rokae_er3_plus"), "mdh"),
        ("nerv_er3_mdh", load_robot_config("nerv_er3"), "mdh"),
        ("rokae_sr5_mat", load_robot_config("rokae_sr5"), "mat"),
        ("aubo_c5_mdh", load_robot_config("aubo_c5"), "mdh"),
    ]
    
    all_passed = True
    for name, config, param_type in configs:
        if not validate_config(name, config, param_type):
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("所有验证通过!")
    else:
        print("验证失败 - 代码可能被意外修改")
        sys.exit(1)


if __name__ == '__main__':
    main()
