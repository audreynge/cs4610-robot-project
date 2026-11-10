from glob import glob

from setuptools import find_packages, setup

package_name = "clean_sweep_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/models", glob("models/*.pt")),
        *[(f"share/{package_name}/{d}", glob(f"{d}/*")) for d in glob("models/*_openvino_model")],
    ],
    install_requires=["setuptools", "pyyaml", "numpy"],
    zip_safe=True,
    maintainer="Clean Sweep Team",
    maintainer_email="noreply@example.com",
    description="Litter detection and 3D localization for Clean Sweep.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ground_truth_stub = clean_sweep_perception.ground_truth_stub_node:main",
            "litter_detector = clean_sweep_perception.litter_detector_node:main",
            "fake_head_camera = clean_sweep_perception.fake_head_camera_node:main",
            "litter_registry = clean_sweep_perception.litter_registry_node:main",
            "evaluate_perception = clean_sweep_perception.evaluate_node:main",
            "benchmark_perception = clean_sweep_perception.benchmark:main",
        ],
    },
)
