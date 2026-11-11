from setuptools import setup

package_name = "clean_sweep_manipulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Paarth",
    description="Clean Sweep manipulation for Stretch",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "demo_arm_gripper = clean_sweep_manipulation.demo_arm_gripper:main",
            "calibrate_gripper = clean_sweep_manipulation.calibrate_gripper:main",
            "demo_pick = clean_sweep_manipulation.demo_pick:main",
        ],
    },
)
