from setuptools import setup
import os
from glob import glob

package_name = 'data_collector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'image_saver   = data_collector.image_saver_node:main',
            'gt_pose_saver = data_collector.gt_pose_saver_node:main',
        ],
    },
)
