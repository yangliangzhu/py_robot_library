"""
生成测试数据用于验证DH参数修改后的代码一致性
"""
import numpy as np
import os

from model.robot_model_numpy import RobotModelNumpy
from model.configs.loader import load_robot_config
from model.ik_type import IkType


def generate_test_configs():
    """生成所有测试配置"""
    configs = [
        ("franka_mdh", load_robot_config("franka"), "mdh"),
        ("rokae_er3_mdh", load_robot_config("rokae_er3"), "mdh"),
        ("rokae_er3_sdh", load_robot_config("rokae_er3_sdh"), "sdh"),
        ("rokae_er3_plus_mdh", load_robot_config("rokae_er3_plus"), "mdh"),
        ("nerv_er3_mdh", load_robot_config("nerv_er3"), "mdh"),
        ("rokae_sr5_mat", load_robot_config("rokae_sr5"), "mat"),
        ("aubo_c5_mdh", load_robot_config("aubo_c5"), "mdh"),
    ]
    return configs


def generate_random_joint_angles(num_dof, num_samples=10, seed=42):
    """生成随机关节角度用于测试"""
    np.random.seed(seed)
    # 生成在合理范围内的随机角度
    angles = []
    for _ in range(num_samples):
        # 使用-90度到+90度之间的随机角度
        q = np.random.uniform(-np.pi/2, np.pi/2, num_dof)
        angles.append(q)
    return np.array(angles)


def generate_grid_joint_angles(num_dof, num_points=3):
    """生成网格状关节角度组合"""
    # 简单网格: -45度, 0度, +45度
    angles = []
    for i in range(num_points):
        angle = -np.pi/4 + i * np.pi/4
        q = np.full(num_dof, angle)
        angles.append(q)
    return np.array(angles)


def generate_test_data_for_config(name, config, param_type, output_dir):
    """为一个配置生成测试数据"""
    print(f"生成测试数据: {name} (type={param_type})")
    
    robot = RobotModelNumpy()
    
    # 构建机器人模型
    if param_type == "mat":
        robot.build(type="mat", solver_type=IkType.IK_STANDARD, config=config)
    else:
        robot.build(type="dh", solver_type=IkType.IK_STANDARD, config=config)
    
    num_dof = robot.num_dof
    print(f"  DOF: {num_dof}")
    
    # 生成测试关节角度
    random_angles = generate_random_joint_angles(num_dof, num_samples=20)
    grid_angles = generate_grid_joint_angles(num_dof, num_points=5)
    
    # 合并所有测试角度
    all_angles = np.vstack([random_angles, grid_angles])
    
    # 计算FK和Jacobian
    fk_results = []
    jacobian_results = []
    jac_normal_results = []
    
    for q in all_angles:
        # Forward Kinematics
        T = robot.fk(q)
        fk_results.append(T.flatten())
        
        # Jacobian
        jac = robot.jacobian(q)
        jacobian_results.append(jac.flatten())
        
        # Jacobian (normal)
        jac_normal = robot.jac_normal(q)
        jac_normal_results.append(jac_normal.flatten())
    
    fk_results = np.array(fk_results)
    jacobian_results = np.array(jacobian_results)
    jac_normal_results = np.array(jac_normal_results)
    
    # 保存数据
    output_file = os.path.join(output_dir, f"{name}.npz")
    np.savez(
        output_file,
        joint_angles=all_angles,
        fk_results=fk_results,
        jacobian_results=jacobian_results,
        jac_normal_results=jac_normal_results,
        num_dof=num_dof,
        config_type=param_type,
    )
    print(f"  保存到: {output_file}")
    print(f"  样本数: {len(all_angles)}, FK shape: {fk_results.shape}")
    
    return {
        "name": name,
        "num_dof": num_dof,
        "num_samples": len(all_angles),
        "param_type": param_type,
    }


def generate_dh_matrix_test_data(output_dir):
    """
    专门测试DH参数转矩阵的测试数据
    """
    from model.dh_param import (
        Dh, mdh_to_matrix_list, sdh_to_matrix_list, get_matrix_list
    )
    
    print("\n生成DH矩阵转换测试数据...")
    
    # 测试不同的DH配置
    test_cases = []
    
    # 1. MDH基本测试
    dh_mdh = Dh([
        [100, 0, 0],           # d, alpha, a (theta=0)
        [50, -np.pi/2, 0],
        [200, np.pi/2, 0],
    ], type='mdh')
    test_cases.append(("mdh_basic", dh_mdh, "mdh"))
    
    # 2. SDH基本测试
    dh_sdh = Dh([
        [100, 0, 0],
        [50, np.pi/2, 0],
        [200, 0, 0],
    ], type='sdh')
    test_cases.append(("sdh_basic", dh_sdh, "sdh"))
    
    # 3. 不同order_map的MDH
    dh_mdh_reorder = Dh([
        [100, 0, 0],
        [50, -np.pi/2, 0],
        [200, np.pi/2, 0],
    ], type='mdh', ordermap=['d', 'a', 'alpha'])
    test_cases.append(("mdh_reorder", dh_mdh_reorder, "mdh"))
    
    # 4. 带theta的MDH (用于未来扩展测试)
    # 目前代码忽略theta，这个测试用例用于验证修改后的行为
    dh_mdh_theta = Dh([
        [100, 0, np.pi/4],    # 包含theta
        [50, -np.pi/2, 0],
        [200, np.pi/2, np.pi/6],
    ], type='mdh')
    test_cases.append(("mdh_with_theta", dh_mdh_theta, "mdh"))
    
    base = np.eye(4)
    ee = np.eye(4)
    
    all_matrix_results = {}
    
    for name, dh_params, dh_type in test_cases:
        # 转换为矩阵列表
        if dh_type == 'mdh':
            matrices = mdh_to_matrix_list(dh_params)
        else:
            matrices = sdh_to_matrix_list(dh_params)
        
        # 展平存储
        matrix_flat = [m.flatten() for m in matrices]
        
        # 同时测试get_matrix_list
        full_matrices = get_matrix_list(dh_params, base, ee)
        full_matrix_flat = [m.flatten() for m in full_matrices]
        
        all_matrix_results[name] = {
            "dh_params": np.array(dh_params.dh_list),
            "dh_type": dh_type,
            "order_map": dh_params.order_map,
            "matrix_list": np.array(matrix_flat),
            "full_matrix_list": np.array(full_matrix_flat),
        }
    
    output_file = os.path.join(output_dir, "dh_matrix_test.npz")
    np.savez(
        output_file,
        **{k: v for k, v in all_matrix_results.items()}
    )
    print(f"  保存到: {output_file}")
    print(f"  测试用例: {list(all_matrix_results.keys())}")


def main():
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成各配置的测试数据
    configs = generate_test_configs()
    
    summary = []
    for name, config, param_type in configs:
        result = generate_test_data_for_config(name, config, param_type, output_dir)
        summary.append(result)
    
    # 生成DH矩阵转换测试数据
    generate_dh_matrix_test_data(output_dir)
    
    # 生成测试汇总
    print("\n" + "="*60)
    print("测试数据生成完成!")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print("\n生成的测试配置:")
    for s in summary:
        print(f"  - {s['name']}: DOF={s['num_dof']}, samples={s['num_samples']}, type={s['param_type']}")
    
    print("\n使用示例:")
    print("```python")
    print("import numpy as np")
    print("data = np.load('test_data/franka_mdh.npz')")
    print("joint_angles = data['joint_angles']")
    print("fk_results = data['fk_results']  # shape: (n, 16)")
    print("")
    print("# 验证FK")
    print("robot = RobotModelNumpy()")
    print("robot.build(type='dh', solver_type=IkType.IK_STANDARD, config=...)")
    print("T = robot.fk(joint_angles[0])")
    print("assert np.allclose(T.flatten(), fk_results[0])")
    print("```")


if __name__ == '__main__':
    main()
