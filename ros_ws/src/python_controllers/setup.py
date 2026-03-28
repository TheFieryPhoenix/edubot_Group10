from setuptools import find_packages, setup

package_name = 'python_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anton',
    maintainer_email='a.bredenbeck@tudelft.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'example_pos_traj = python_controllers.example_pos_traj:main',
            'example_vel_traj = python_controllers.example_vel_traj:main',
            'group_pos_traj = python_controllers.group_pos_traj:main',
            'triangle_draw = python_controllers.triangle_draw:main',
            'test_ik_pos = python_controllers.test_ik_pos:main',
            'group_velocity_traj = python_controllers.group_velocity_traj:main',
            'group_vel_traj = python_controllers.group_vel_traj:main',
            'pickup_traj = python_controllers.pickup_traj:main',
            'pickup_traj_vel_control = python_controllers.pickup_traj_vel_control:main',
            'kinematics_utils = python_controllers.kinematics_utils:main',
            'square_trajectory_node = python_controllers.square_trajectory_node:main',
            'pick_and_place = python_controllers.pick_and_place:main',
            'ik_pick_and_place = python_controllers.ik_pick_and_place:main',
            'pick_and_place_copy = python_controllers.pick_and_place_copy:main'
        ],
    },
)
